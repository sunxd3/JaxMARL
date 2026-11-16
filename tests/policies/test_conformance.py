"""What every policy family and env must satisfy; runs on each case in ``conftest.py``."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.policies import FixedSeats, replay, rollout
from jaxmarl.wrappers.baselines import LogWrapper

STEPS = 100


def _rollout(env, seats, seed=0):
    return jax.jit(lambda key: rollout(env, seats, key, STEPS))(
        jax.random.PRNGKey(seed)
    )


def test_recorded_probabilities_are_distributions(case):
    env, seats = case
    traj = _rollout(env, seats)
    logp, avail = np.asarray(traj.logp), np.asarray(traj.avail)
    assert logp.shape[:2] == (STEPS, len(env.agents))
    assert np.isfinite(logp[avail]).all()
    assert np.isneginf(logp[~avail]).all()
    np.testing.assert_allclose(np.exp(logp).sum(-1), 1.0, atol=1e-5)
    taken = np.take_along_axis(avail, np.asarray(traj.action)[..., None], axis=-1)
    assert taken.all()
    assert np.asarray(traj.done).any(), "the case should include an episode boundary"


def test_replay_reproduces_recorded_probabilities(case):
    """The memory rule: memory can be recomputed from what the seat saw and did."""
    env, seats = case
    traj = _rollout(env, seats)
    for i, seat in enumerate(env.agents):
        pop = seats[seat]
        for k in range(len(pop)):
            replayed = jax.jit(lambda p: replay(pop.policy, p, env, traj, seat))(
                pop.get(k)
            )
            played = np.asarray(traj.type_idx[:, i]) == k
            np.testing.assert_allclose(
                np.asarray(replayed)[played],
                np.asarray(traj.logp[:, i])[played],
                atol=1e-5,
            )


def test_fixed_seats_matches_the_raw_env(case):
    """Given the same moves, the env behaves the same with or without policies in seats."""
    env, seats = case
    open_seat = env.agents[0]
    fixed = {seat: pop for seat, pop in seats.items() if seat != open_seat}
    wrapped = FixedSeats(env, fixed)
    assert wrapped.agents == [open_seat] and wrapped.num_agents == 1

    key = jax.random.PRNGKey(1)
    obs, state = wrapped.reset(key)
    raw_obs, raw_state = env.reset(key)
    for t in range(60):
        np.testing.assert_array_equal(obs[open_seat], raw_obs[open_seat])
        for a, b in zip(jax.tree.leaves(state.env_state), jax.tree.leaves(raw_state)):
            np.testing.assert_array_equal(a, b)

        key, key_act, key_step = jax.random.split(key, 3)
        mine = jax.random.randint(key_act, (), 0, env.action_space(open_seat).n)
        # the documented key rule: policy seat i draws with fold_in(env key, i + 1)
        env_key = jax.random.split(key_step)[0]
        moves = {open_seat: mine}
        for i, seat in enumerate(wrapped.fixed_agents):
            logp = wrapped.seat_state(state, seat).logp
            moves[seat] = jax.random.categorical(
                jax.random.fold_in(env_key, i + 1), logp
            )
        moves = {seat: moves[seat] for seat in env.agents}

        obs, state, reward, done, _ = wrapped.step(key_step, state, {open_seat: mine})
        raw_obs, raw_state, raw_reward, raw_done, _ = env.step(
            key_step, raw_state, moves
        )
        assert set(reward) == {open_seat}
        np.testing.assert_array_equal(reward[open_seat], raw_reward[open_seat])
        np.testing.assert_array_equal(done["__all__"], raw_done["__all__"])


def test_fixed_seats_trains_like_an_env(case):
    """It runs under LogWrapper and vmap, and its info reshapes the way training scripts do."""
    env, seats = case
    open_seat = env.agents[0]
    wrapped = LogWrapper(
        FixedSeats(env, {s: p for s, p in seats.items() if s != open_seat})
    )
    num_envs = 4
    keys = jax.random.split(jax.random.PRNGKey(2), num_envs)
    obs, state = jax.vmap(wrapped.reset)(keys)
    actions = {open_seat: jnp.zeros((num_envs,), dtype=jnp.int32)}
    step = jax.jit(jax.vmap(wrapped.step))
    for _ in range(3):
        obs, state, reward, done, info = step(keys, state, actions)
    assert set(obs) == {open_seat} and obs[open_seat].shape[0] == num_envs
    jax.tree.map(lambda x: x.reshape((num_envs * wrapped.num_agents,)), info)
