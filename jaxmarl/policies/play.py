"""Running policies in an env.

``rollout``     every seat is played by a population; returns the recorded trajectory.
``FixedSeats``  an env in which some seats are played by populations and the rest are left
                open for a learner.  It is a ``MultiAgentEnv``, so wrappers and training
                scripts written for JaxMARL envs run on it.

Both are built on the same two functions, ``_reset`` and ``_transition``, so there is one
implementation of "draw a type, keep its memory, sample its move".

Timing.  A policy is stepped as soon as a state exists, and its move probabilities for that
state are stored (``SeatState.logp``).  The move is sampled from them at the next transition.
So the probabilities of a move are known before it is made.

Random keys.  The env gets the key it would have got without us; policy keys are derived from
it with ``fold_in``.  Given the same moves, the env therefore behaves identically with or
without policies in its seats.
"""

from typing import Any, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.policies.base import (
    Population,
    Seen,
    avail_actions,
    masked_log_probs,
    num_actions,
)


@struct.dataclass
class SeatState:
    """Runtime data for one policy-played seat."""

    type_idx: jax.Array  # int32; which type of the population is playing this episode
    params: Any  # that type's params
    mem: Any  # the policy's memory, already updated for the current state
    avail: jax.Array  # bool[A] for the current state
    logp: (
        jax.Array
    )  # [A]; exact log-probabilities of the next move, for the current state
    last_action: jax.Array  # int32; -1 at the start of an episode


@struct.dataclass
class PlayState:
    env_state: Any
    seats: Tuple[SeatState, ...]  # one per policy-played seat, in ``env.agents`` order


class Traj(NamedTuple):
    """A recorded run of ``T`` steps.  Arrays over seats follow ``env.agents`` order."""

    state: Any  # [T, ...]   env state s_t, before move t
    obs: Dict  # seat -> [T, ...]   observations emitted for s_t
    avail: jax.Array  # [T, N, A] bool
    action: jax.Array  # [T, N] int32   the moves made in s_t
    logp: jax.Array  # [T, N, A]   exact log-probabilities of each seat's move in s_t
    reward: jax.Array  # [T, N]
    done: jax.Array  # [T] bool   the episode ended with move t
    type_idx: jax.Array  # [T, N] int32   the type playing each seat


def _ordered(env, seats: Dict[Any, Population]):
    """``seats`` as a tuple of (seat, population) in ``env.agents`` order."""
    unknown = [s for s in seats if s not in env.agents]
    if unknown:
        raise ValueError(f"not seats of this env: {unknown}; seats are {env.agents}")
    return tuple((s, seats[s]) for s in env.agents if s in seats)


def _advance(pop, env, env_state, obs, seat, type_idx, params, mem, last_action):
    """Step one seat's policy on a newly produced state."""
    masks = avail_actions(env, env_state)
    if masks is None:
        avail = jnp.ones((num_actions(env, seat),), dtype=bool)
    else:
        avail = masks[seat].astype(bool)
    view = pop.policy.view(env, env_state, obs, seat)
    mem, logits = pop.policy.step(params, mem, Seen(view, avail, last_action))
    return SeatState(
        type_idx=type_idx,
        params=params,
        mem=mem,
        avail=avail,
        logp=masked_log_probs(logits, avail),
        last_action=last_action,
    )


def _reset(env, fixed, key):
    """Reset the env and start every policy seat with a freshly drawn type."""
    obs, env_state = env.reset(key)
    seats = []
    for i, (seat, pop) in enumerate(fixed):
        type_idx, params = pop.sample(jax.random.fold_in(key, i + 1))
        seats.append(
            _advance(
                pop,
                env,
                env_state,
                obs,
                seat,
                type_idx,
                params,
                pop.policy.init(params),
                jnp.array(-1, dtype=jnp.int32),
            )
        )
    return obs, PlayState(env_state=env_state, seats=tuple(seats))


def _transition(env, fixed, key, st: PlayState, open_actions):
    """One env transition without auto-reset.

    Samples the policy seats' moves from their stored probabilities, steps the env with them
    and ``open_actions``, and steps each policy on the new state.  Returns the env's outputs
    for all seats, the new state, and the moves of all seats.
    """
    chosen = dict(open_actions)
    for i, ((seat, _), s) in enumerate(zip(fixed, st.seats)):
        chosen[seat] = jax.random.categorical(
            jax.random.fold_in(key, i + 1), s.logp
        ).astype(jnp.int32)
    # some envs read the actions dict by position, so keep env.agents order
    actions = {seat: chosen[seat] for seat in env.agents}
    obs, env_state, reward, done, info = env.step_env(key, st.env_state, actions)
    seats = tuple(
        _advance(
            pop, env, env_state, obs, seat, s.type_idx, s.params, s.mem, actions[seat]
        )
        for (seat, pop), s in zip(fixed, st.seats)
    )
    return obs, PlayState(env_state=env_state, seats=seats), reward, done, info, actions


def _select(pred, on_true, on_false):
    return jax.tree.map(lambda x, y: jax.lax.select(pred, x, y), on_true, on_false)


def rollout(env, seats: Dict[Any, Population], key, num_steps: int) -> Traj:
    """Play ``num_steps`` steps with every seat played by a population.

    Episodes that end are followed by a new one (a new type is drawn for every seat), as with
    ``env.step``.  Wrap the call in ``jax.jit`` for speed; it can be ``vmap``-ed over ``key``.
    """
    fixed = _ordered(env, seats)
    if len(fixed) != len(env.agents):
        missing = [s for s in env.agents if s not in seats]
        raise ValueError(
            f"rollout needs a population for every seat; missing {missing}"
        )

    def body(carry, key_t):
        obs, st = carry
        key_step, key_reset = jax.random.split(key_t)  # as MultiAgentEnv.step does
        obs_st, st_st, reward, done, _, actions = _transition(
            env, fixed, key_step, st, {}
        )
        obs_re, st_re = _reset(env, fixed, key_reset)
        ended = done["__all__"]
        record = Traj(
            state=st.env_state,
            obs=obs,
            avail=jnp.stack([s.avail for s in st.seats]),
            action=jnp.stack([actions[seat] for seat in env.agents]),
            logp=jnp.stack([s.logp for s in st.seats]),
            reward=jnp.stack([reward[seat] for seat in env.agents]),
            done=ended,
            type_idx=jnp.stack([s.type_idx for s in st.seats]),
        )
        return (_select(ended, obs_re, obs_st), _select(ended, st_re, st_st)), record

    key, key_reset = jax.random.split(key)
    _, traj = jax.lax.scan(
        body, _reset(env, fixed, key_reset), jax.random.split(key, num_steps)
    )
    return traj


class FixedSeats(MultiAgentEnv):
    """``env`` with some seats played by populations; only the other seats are exposed.

        env = FixedSeats(make("overcooked_v2"), {"agent_1": population})

    ``agents`` are the open seats.  The state is a ``PlayState``; the ground truth for a fixed
    seat (its type, and the exact probabilities of its next move) is in ``state.seats`` and
    can be read with ``seat_state``.  It is kept out of ``info`` because training scripts
    reshape every ``info`` entry.

    ``step`` is inherited from ``MultiAgentEnv``: when an episode ends it calls ``reset``,
    which draws a new type and clears the memory of every fixed seat.
    """

    def __init__(self, env: MultiAgentEnv, fixed: Dict[Any, Population]):
        self._env = env
        self._fixed = _ordered(env, fixed)
        self.fixed_agents = [seat for seat, _ in self._fixed]
        self.agents = [seat for seat in env.agents if seat not in fixed]
        self.num_agents = len(self.agents)
        self.observation_spaces = self._keep(getattr(env, "observation_spaces", {}))
        self.action_spaces = self._keep(getattr(env, "action_spaces", {}))

    def __getattr__(self, name):
        # only reached for attributes not defined here: layout, max_steps, ...
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._env, name)

    def _keep(self, per_seat: dict) -> dict:
        return {seat: v for seat, v in per_seat.items() if seat in self.agents}

    def _keep_info(self, info: dict) -> dict:
        def is_per_seat(v):
            return (
                isinstance(v, dict)
                and len(v) > 0
                and all(k in self._env.agents for k in v)
            )

        return {k: self._keep(v) if is_per_seat(v) else v for k, v in info.items()}

    def reset(self, key):
        obs, st = _reset(self._env, self._fixed, key)
        return self._keep(obs), st

    def step_env(self, key, state: PlayState, actions):
        obs, st, reward, done, info, _ = _transition(
            self._env,
            self._fixed,
            key,
            state,
            {seat: actions[seat] for seat in self.agents},
        )
        done = {**self._keep(done), "__all__": done["__all__"]}
        return self._keep(obs), st, self._keep(reward), done, self._keep_info(info)

    def get_obs(self, state: PlayState):
        return self._keep(self._env.get_obs(state.env_state))

    def get_avail_actions(self, state: PlayState):
        masks = avail_actions(self._env, state.env_state)
        if masks is None:
            raise NotImplementedError
        return self._keep(masks)

    def seat_state(self, state: PlayState, seat) -> SeatState:
        """The ground truth for a fixed seat: its type, memory and next-move probabilities."""
        return state.seats[self.fixed_agents.index(seat)]

    def observation_space(self, *args, **kwargs):
        return self._env.observation_space(*args, **kwargs)

    def action_space(self, *args, **kwargs):
        return self._env.action_space(*args, **kwargs)

    @property
    def name(self) -> str:
        return self._env.name

    @property
    def agent_classes(self) -> dict:
        return self._env.agent_classes
