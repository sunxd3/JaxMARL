"""Policies: programmed players for JaxMARL environments.

A policy is a deterministic function from (type, memory, what the seat sees) to the
distribution over the seat's next move.  The type is data (``params``), so a set of types is
one stacked pytree; the memory (``mem``) is a fixed-shape pytree.  The code that runs a policy
(``jaxmarl.policies.play``) does all the sampling, so the exact probability of every move is
always on record.

Rules every policy follows:

1. No randomness.  ``step`` returns logits; the runner makes the only random draw.
2. Memory depends only on what the seat saw and did (the ``Seen`` values passed to ``step``).
3. Any hidden choice (a goal, an intent) is summed over inside ``step``, never sampled.
4. A policy reads only its ``view``.  If the env's own observation is not enough, override
   ``view`` so that what the policy reads is one explicit function.

A "seat" is an entry of ``env.agents``, whatever type the env uses for it.
"""

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp


class Seen(NamedTuple):
    """What a policy is given for one seat at one step."""

    view: Any  # ``policy.view(...)``; by default the env's own observation for the seat
    avail: jax.Array  # bool[A]; all True when the env has no action mask
    last_action: (
        jax.Array
    )  # int32; this seat's previous move, -1 at the start of an episode


class Policy:
    """Code only.  Build one per env; keep static things (layout, tables) on ``self``."""

    def view(self, env, state, obs, seat):
        """What this policy reads.  Default: the observation the env emitted for the seat."""
        return obs[seat]

    def init(self, params):
        """Memory at the start of an episode."""
        return ()

    def step(self, params, mem, seen: Seen):
        """Return ``(new mem, logits[A])``."""
        raise NotImplementedError


class UniformPolicy(Policy):
    """Every available move with equal probability.  Works in any discrete-action env."""

    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def step(self, params, mem, seen):
        return mem, jnp.zeros((self.num_actions,))


@dataclass(frozen=True, eq=False)
class Population:
    """One policy, a finite set of named types for it, and a prior over them.

    ``params`` is stacked: every leaf has a leading axis of length ``len(names)``.  A single
    type is a population of size one.  This is a static object, not a pytree.
    """

    policy: Policy
    params: Any
    probs: jax.Array
    names: Tuple[str, ...]

    def __post_init__(self):
        k = len(self.names)
        if jnp.shape(self.probs) != (k,):
            raise ValueError(
                f"probs must have shape ({k},), got {jnp.shape(self.probs)}"
            )
        for leaf in jax.tree.leaves(self.params):
            if jnp.shape(leaf)[:1] != (k,):
                raise ValueError(
                    f"every params leaf needs a leading axis of {k}, got {jnp.shape(leaf)}"
                )

    def __len__(self) -> int:
        return len(self.names)

    def get(self, idx):
        """The params of type ``idx``."""
        return jax.tree.map(lambda x: x[idx], self.params)

    def only(self, name: str) -> "Population":
        """The population of one that always plays the type called ``name``."""
        k = self.names.index(name)
        return Population(
            policy=self.policy,
            params=jax.tree.map(lambda x: x[k : k + 1], self.params),
            probs=jnp.ones((1,)),
            names=(name,),
        )

    def sample(self, key):
        """Draw a type from the prior: ``(type index, its params)``."""
        idx = jax.random.categorical(key, jnp.log(self.probs)).astype(jnp.int32)
        return idx, self.get(idx)


def uniform_population(env, seat) -> Population:
    """The uniform-random player for ``seat``, as a population of one."""
    return Population(
        policy=UniformPolicy(num_actions(env, seat)),
        params=(),
        probs=jnp.ones((1,)),
        names=("uniform",),
    )


def num_actions(env, seat) -> int:
    return int(env.action_space(seat).n)


def avail_actions(env, state) -> Optional[dict]:
    """The env's action masks keyed by seat, or None if the env has none.

    This is the one place that knows the names envs use for their mask.
    """
    for name in ("get_avail_actions", "get_legal_moves"):
        fn = getattr(env, name, None)
        if fn is None:
            continue
        try:
            return fn(state)
        except NotImplementedError:
            continue
    return None


def masked_log_probs(logits, avail):
    """Normalised log-probabilities with unavailable moves at -inf."""
    return jax.nn.log_softmax(jnp.where(avail, logits, -jnp.inf))
