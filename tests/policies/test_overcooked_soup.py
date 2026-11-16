"""The Overcooked-V2 soup family: decoding, behaviour, and whether its types can be told apart."""

import jax
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


@pytest.mark.parametrize(
    "pair",
    [
        ("runner", "plater"),
        ("runner", "runner"),
        ("plater", "plater"),
        ("flex", "flex"),
    ],
)
def test_soup_players_deliver(pair):
    env = _env()
    population = make_population(env)
    seats = {seat: population.only(name) for seat, name in zip(env.agents, pair)}
    run = jax.jit(
        jax.vmap(lambda key: rollout(env, seats, key, 400).reward[:, 0].sum())
    )
    returns = np.asarray(run(jax.random.split(jax.random.PRNGKey(0), 8)))
    assert returns.mean() >= 3 * 20, (
        f"{pair} delivered {returns / 20} soups in 400 steps"
    )


def test_uniform_players_do_not():
    env = _env()
    seats = {seat: uniform_population(env, seat) for seat in env.agents}
    run = jax.jit(
        jax.vmap(lambda key: rollout(env, seats, key, 400).reward[:, 0].sum())
    )
    assert np.asarray(run(jax.random.split(jax.random.PRNGKey(0), 8))).mean() < 20


def test_posterior_finds_the_type():
    """The types can be told apart from play, and more surely the longer the episode.

    Measured over 96 episodes when this was written: the exact posterior puts 0.47 on the true
    type after 25 steps, 0.65 after 100 and 0.88 after 400 (chance is 0.33).  Runner and
    plater are not confused with each other; flex is sometimes taken for one of them.
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
    early, late = on_truth[:, 24].mean(), on_truth[:, -1].mean()
    assert late > 0.75, f"mean posterior on the true type after 400 steps: {late:.2f}"
    assert late > early + 0.2, f"posterior should sharpen: {early:.2f} -> {late:.2f}"
    guess = post[:, -1].argmax(-1)
    assert (guess == truth).mean() > 0.75

    runner, plater = population.names.index("runner"), population.names.index("plater")
    crossed = ((truth == runner) & (guess == plater)) | (
        (truth == plater) & (guess == runner)
    )
    assert not crossed.any()
