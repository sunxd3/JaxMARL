"""The Overcooked-V2 soup family: decoding, behaviour, and whether its types can be told apart."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmarl import make
from jaxmarl.environments.overcooked_v2.common import DynamicObject
from jaxmarl.policies import rollout, type_posterior, uniform_population
from jaxmarl.policies.overcooked_v2.common import decode
from jaxmarl.policies.overcooked_v2.soup import make_population


def _env(max_steps=400):
    return make("overcooked_v2", layout="cramped_room", max_steps=max_steps)


def test_decode_recovers_the_state():
    env = _env(max_steps=200)
    population = make_population(env)
    seats = {seat: population for seat in env.agents}
    traj = jax.jit(lambda key: rollout(env, seats, key, 200))(jax.random.PRNGKey(0))
    n = env.layout.num_ingredients
    for i, seat in enumerate(env.agents):
        view = jax.vmap(lambda o: decode(o, n))(traj.obs[seat])
        agents = traj.state.agents
        np.testing.assert_array_equal(view.pos[:, 0], agents.pos.y[:, i])
        np.testing.assert_array_equal(view.pos[:, 1], agents.pos.x[:, i])
        np.testing.assert_array_equal(view.dir, agents.dir[:, i])
        np.testing.assert_array_equal(view.other_pos[:, 0], agents.pos.y[:, 1 - i])
        inventory = np.asarray(agents.inventory[:, i])
        np.testing.assert_array_equal(view.held[:, 0], inventory & DynamicObject.PLATE)
        np.testing.assert_array_equal(
            view.held[:, 1], (inventory & DynamicObject.COOKED) >> 1
        )
        np.testing.assert_array_equal(view.held[:, 2], (inventory >> 2) & 0x3)
    # the run should have exercised more than empty hands
    assert len(np.unique(np.asarray(traj.state.agents.inventory))) >= 4


def _soups(env, seats, episodes=8):
    """Mean soups delivered in 400 steps."""
    run = jax.jit(
        jax.vmap(lambda key: rollout(env, seats, key, 400).reward[:, 0].sum())
    )
    returns = run(jax.random.split(jax.random.PRNGKey(0), episodes))
    return float(np.asarray(returns).mean()) / 20


@pytest.mark.parametrize(
    "pair, at_least, at_most",
    [
        (("runner", "plater"), 6, None),
        (("plater", "runner"), 6, None),
        (("flex", "flex"), 3, None),
        (("flex", "lazy"), 3, None),
        # two players with the same role leave the other errand undone
        (("runner", "runner"), None, 1),
        (("plater", "plater"), None, 1),
    ],
)
def test_who_delivers_depends_on_the_partner(pair, at_least, at_most):
    env = _env()
    population = make_population(env)
    seats = {seat: population.only(name) for seat, name in zip(env.agents, pair)}
    soups = _soups(env, seats)
    assert at_least is None or soups >= at_least, f"{pair}: {soups:.2f} soups"
    assert at_most is None or soups <= at_most, f"{pair}: {soups:.2f} soups"


@pytest.mark.parametrize("w_ingredient, w_plate", [(1.0, 4.0), (2.5, 2.5)])
def test_a_player_with_an_idle_partner_still_delivers(w_ingredient, w_plate):
    """A partner that never moves leaves every errand to the player.  A player that leans
    towards plates fetches one before the pot is full; it used to hold it for the rest of the
    episode instead of putting it down to fetch ingredients."""
    env = _env()
    flex = make_population(env).only("flex")

    def one(**changes):
        return dataclasses.replace(
            flex,
            params=flex.params.replace(
                **{k: jnp.full((1,), v) for k, v in changes.items()}
            ),
        )

    seats = {
        env.agents[0]: one(rho=1.0, eps=0.0),
        env.agents[1]: one(w_ingredient=w_ingredient, w_plate=w_plate),
    }
    assert _soups(env, seats) >= 3


def test_uniform_players_do_not():
    env = _env()
    seats = {seat: uniform_population(env, seat) for seat in env.agents}
    run = jax.jit(
        jax.vmap(lambda key: rollout(env, seats, key, 400).reward[:, 0].sum())
    )
    assert np.asarray(run(jax.random.split(jax.random.PRNGKey(0), 8))).mean() < 20


def test_posterior_finds_the_type():
    """The types can be told apart from play, and more surely the longer the episode.

    Measured over 96 episodes when this was written: the exact posterior puts 0.81 on the true
    type after 10 steps, 0.91 after 25 and 1.00 after 400 (chance is 0.25).
    """
    env = _env(max_steps=400)
    population = make_population(env)
    seats = {seat: population for seat in env.agents}
    seat, i = env.agents[1], 1

    def run(key):
        traj = rollout(env, seats, key, 400)
        return type_posterior(population, env, traj, seat), traj.type_idx[-1, i]

    log_post, truth = jax.jit(jax.vmap(run))(
        jax.random.split(jax.random.PRNGKey(0), 48)
    )
    post, truth = np.exp(np.asarray(log_post)), np.asarray(truth)  # [episodes, T, K]
    np.testing.assert_allclose(post.sum(-1), 1.0, atol=1e-5)
    assert len(np.unique(truth)) == len(population)

    on_truth = post[np.arange(len(truth)), :, truth]  # [episodes, T]
    early, late = on_truth[:, 9].mean(), on_truth[:, -1].mean()
    assert early < 0.95, f"one move should not give the type away: {early:.2f}"
    assert late > 0.95, f"mean posterior on the true type after 400 steps: {late:.2f}"
    assert (post[:, -1].argmax(-1) == truth).mean() > 0.9
