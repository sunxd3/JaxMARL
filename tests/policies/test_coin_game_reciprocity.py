"""The Coin Game reciprocity family: reading the partner's moves, behaviour, and telling types apart."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmarl import make
from jaxmarl.policies import rollout, type_posterior
from jaxmarl.policies.base import Seen
from jaxmarl.policies.coin_game.reciprocity import make_population


@pytest.mark.parametrize(
    "egocentric, cnn", [(False, False), (True, False), (True, True)]
)
def test_grudge_counts_the_coins_the_partner_took(egocentric, cnn):
    """The grim player's grudge goes up exactly when the env counts a coin of its taken by the
    partner, except when both coins lay on one cell."""
    env = make("coin_game", egocentric=egocentric, cnn=cnn)
    population = make_population(env)
    seats = {seat: population for seat in env.agents}
    traj = jax.jit(lambda key: rollout(env, seats, key, 1000))(jax.random.PRNGKey(0))
    policy, grim = population.policy, population.get(population.names.index("grim"))

    start = jnp.concatenate([jnp.array([True]), traj.done[:-1]])
    first = np.asarray(start)[1:]
    shared = np.all(
        np.asarray(traj.state.red_coin_pos) == np.asarray(traj.state.blue_coin_pos), -1
    )[:-1]
    fresh = np.asarray(traj.state.step)[1:] == 0

    # seat 1 is blue, whose coins red takes, and the other way round
    for i, seat, taken in (
        (1, "1", traj.state.red_defect),
        (0, "0", traj.state.blue_defect),
    ):
        views = jax.vmap(lambda s, o: policy.view(env, s, o, seat))(
            traj.state, traj.obs
        )
        previous = jnp.concatenate([jnp.array([-1]), traj.action[:-1, i]])
        last = jnp.where(start, -1, previous).astype(jnp.int32)

        def f(mem, x):
            view, last_action, is_first = x
            mem = jax.tree.map(
                lambda a, b: jax.lax.select(is_first, a, b), policy.init(grim), mem
            )
            mem, _ = policy.step(grim, mem, Seen(view, jnp.ones(5, bool), last_action))
            return mem, mem.grudge

        _, grudge = jax.lax.scan(f, policy.init(grim), (views, last, start))
        detected = (np.diff(np.asarray(grudge)) > 0) & ~first
        counted = np.diff(np.asarray(taken.sum(-1)).astype(int)) > 0
        truth = counted & ~fresh & ~first & ~shared
        assert truth.sum() > 100
        np.testing.assert_array_equal(detected, truth)


def _returns(env, a, b, episodes=128):
    """Mean return of each seat over one 100-step meta-episode."""
    population = make_population(env)
    seats = {"0": population.only(a), "1": population.only(b)}
    run = jax.jit(jax.vmap(lambda key: rollout(env, seats, key, 100).reward.sum(0)))
    return np.asarray(run(jax.random.split(jax.random.PRNGKey(1), episodes))).mean(0)


@pytest.mark.parametrize(
    "pair, at_least, at_most",
    [
        # the nice types cooperate with each other
        (("cooperator", "cooperator"), (40, 40), None),
        (("tit_for_tat", "tit_for_tat"), (35, 35), None),
        (("cooperator", "tit_for_tat"), (40, 40), None),
        # a defector exploits a cooperator, but not a retaliator
        (("defector", "cooperator"), (45, None), (None, 0)),
        (("defector", "tit_for_tat"), None, (15, None)),
        (("defector", "grim"), None, (-20, None)),
    ],
)
def test_who_gains_depends_on_the_partner(pair, at_least, at_most):
    returns = _returns(make("coin_game"), *pair)
    for r, lo, hi in zip(returns, at_least or (None, None), at_most or (None, None)):
        assert lo is None or r >= lo, f"{pair}: {returns}"
        assert hi is None or r <= hi, f"{pair}: {returns}"


def test_only_a_defecting_partner_reveals_the_type():
    """Against a defector every type shows itself; against a cooperator the nice types act
    alike, so a learner has to take coins to find out who it plays.

    Measured over 256 episodes when this was written, posterior on the true type after 100
    moves: 1.00 against a defector; against a cooperator 1.00 for a defector and about 0.5
    for cooperator and tit-for-tat.
    """
    env = make("coin_game")
    population = make_population(env)
    names = population.names

    def final_posterior(partner):
        seats = {"0": population.only(partner), "1": population}

        def run(key):
            traj = rollout(env, seats, key, 100)
            return type_posterior(population, env, traj, "1")[-1], traj.type_idx[-1, 1]

        log_post, truth = jax.jit(jax.vmap(run))(
            jax.random.split(jax.random.PRNGKey(2), 128)
        )
        post, truth = np.exp(np.asarray(log_post)), np.asarray(truth)
        return {n: post[truth == k, k].mean() for k, n in enumerate(names)}

    against_defector = final_posterior("defector")
    assert min(against_defector.values()) > 0.95, against_defector

    against_cooperator = final_posterior("cooperator")
    assert against_cooperator["defector"] > 0.95, against_cooperator
    assert against_cooperator["cooperator"] < 0.7, against_cooperator
    assert against_cooperator["tit_for_tat"] < 0.7, against_cooperator
