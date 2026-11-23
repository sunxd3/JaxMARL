"""Replaying a policy over a recorded trajectory, and the exact posterior over its type.

Both rely on the memory rule in ``jaxmarl.policies.base``: a policy's memory is a
deterministic function of what its seat saw and did, so it can be recomputed from the record.
"""

import jax
import jax.numpy as jnp

from jaxmarl.policies.base import Population, Seen, masked_log_probs
from jaxmarl.policies.play import Traj


def _episode_starts(traj: Traj):
    """[T] bool: step t is the first of an episode."""
    return jnp.concatenate([jnp.array([True]), traj.done[:-1]])


def replay(policy, params, env, traj: Traj, seat):
    """The move log-probabilities ``[T, A]`` that ``policy`` with ``params`` gives for
    ``seat`` at every step of ``traj``, had it been sitting there and made the recorded moves.
    """
    i = env.agents.index(seat)
    start = _episode_starts(traj)
    previous = jnp.concatenate([jnp.array([-1], dtype=jnp.int32), traj.action[:-1, i]])
    last_action = jnp.where(start, -1, previous).astype(jnp.int32)
    views = jax.vmap(lambda s, o: policy.view(env, s, o, seat))(traj.state, traj.obs)

    def f(mem, x):
        view, avail, last, first = x
        fresh = policy.init(params)
        mem = jax.tree.map(lambda a, b: jax.lax.select(first, a, b), fresh, mem)
        mem, logits = policy.step(params, mem, Seen(view, avail, last))
        return mem, masked_log_probs(logits, avail)

    _, logp = jax.lax.scan(
        f, policy.init(params), (views, traj.avail[:, i], last_action, start)
    )
    return logp


def type_posterior(pop: Population, env, traj: Traj, seat):
    """Exact log-posterior ``[T, K]`` over which type of ``pop`` is playing ``seat``.

    Row ``t`` is the posterior after the seat's moves from the start of the current episode up
    to and including move ``t``; it restarts from the prior at every episode, because a new
    type is drawn then.

    This is the omniscient observer: it conditions on the seat's own views, which another
    player may not be able to see.  What a learner in another seat could infer is a different
    quantity.
    """
    i = env.agents.index(seat)
    taken = traj.action[:, i]

    def loglik(k):
        logp = replay(pop.policy, pop.get(k), env, traj, seat)
        return jnp.take_along_axis(logp, taken[:, None], axis=1)[:, 0]

    ll = jax.vmap(loglik)(jnp.arange(len(pop)))  # [K, T]

    def accumulate(total, x):
        step_ll, first = x
        total = jnp.where(first, 0.0, total) + step_ll
        return total, total

    _, cum = jax.lax.scan(
        accumulate, jnp.zeros(len(pop)), (ll.T, _episode_starts(traj))
    )
    return jax.nn.log_softmax(jnp.log(pop.probs) + cum, axis=-1)
