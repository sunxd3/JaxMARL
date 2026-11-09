"""Teammate registry for the typed-intent partner families.

``make_teammate`` builds a typed-intent partner for an env, ``make_theta_set`` / ``sample_theta``
enumerate a family's types.

Families (``TypedIntentTheta`` grids):

* ``typed_intent``   - 12 types: role {runner, plater, flex} x staging x rho {lo, hi};
* ``typed_intent_x`` - 8 types: exclusive roles {runner_x, plater_x} x staging x rho;
* ``typed_intent_g`` - 8 types: graded exclusivity {runner_g, plater_g} x staging x rho,
  with the per-step masking probability given as ``extra={"excl_q": q}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp

from partner_modeling.partners.typed_intent import (
    OvercookedV2TypedIntentPolicyWrapper,
    TypedIntentConfig,
    TypedIntentTheta,
)

_STAGING = ("pot_direct", "counter_stager")
_RHO = ("lo", "hi")

THETA_PRESETS: Dict[str, Tuple[str, ...]] = {
    "typed_intent": ("default",) + tuple(
        f"{r}_{s}_{rho}" for r in ("runner", "plater", "flex") for s in _STAGING for rho in _RHO),
    "typed_intent_x": tuple(
        f"{r}_{s}_{rho}" for r in ("runner_x", "plater_x") for s in _STAGING for rho in _RHO),
    "typed_intent_g": tuple(
        f"{r}_{s}_{rho}" for r in ("runner_g", "plater_g") for s in _STAGING for rho in _RHO),
}

THETA_GRIDS: Dict[str, Callable[[], List[TypedIntentTheta]]] = {
    "typed_intent": TypedIntentTheta.grid,
    "typed_intent_x": TypedIntentTheta.grid_x,
    "typed_intent_g": TypedIntentTheta.grid_g,
}


@dataclass
class HeuristicTeammateSpec:
    """Specification of a typed-intent teammate.

    ``theta`` is a preset name of ``family``, a dict of ``TypedIntentTheta`` fields, or a
    ``TypedIntentTheta``.  ``extra`` holds ``TypedIntentConfig`` options (beta, eps, excl_q, ...).
    ``use_log_wrapper`` and ``reseed_on_reset`` are accepted for compatibility and have no
    effect: the partner unwraps any state wrapper and always uses the rng passed per step.
    """

    family: str
    theta: Union[str, Dict[str, Any], TypedIntentTheta] = "default"
    use_log_wrapper: bool = True
    start_cooking_interaction: bool = False
    reseed_on_reset: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeammatePolicy:
    """``init(rng) -> carry``; ``act(carry, obs, done, rng, env_state) -> (carry, action)``;
    ``dist_fn(carry, env_state) -> (probs[6], info)`` is the exact action distribution."""

    name: str
    init: Callable[..., Any]
    act: Callable[..., Tuple[Any, Any]]
    _policy: Any = None
    _params: Any = None
    dist_fn: Optional[Callable[..., Any]] = None


def _check_family(family: str) -> None:
    if family not in THETA_PRESETS:
        raise ValueError(f"Unknown family '{family}'. Valid families: {sorted(THETA_PRESETS)}")


def resolve_theta(family: str, theta_input: Union[str, Dict[str, Any], TypedIntentTheta]) -> TypedIntentTheta:
    _check_family(family)
    if isinstance(theta_input, TypedIntentTheta):
        return theta_input
    if isinstance(theta_input, str):
        if theta_input not in THETA_PRESETS[family]:
            raise ValueError(f"Unknown theta preset '{theta_input}' for family '{family}'. "
                             f"Valid presets: {sorted(THETA_PRESETS[family])}")
        return TypedIntentTheta.from_preset(theta_input)
    if isinstance(theta_input, dict):
        default = TypedIntentTheta.default()
        expected = set(vars(default))
        unknown = set(theta_input) - expected
        if unknown:
            raise ValueError(f"Unknown theta fields for family '{family}': {sorted(unknown)}. "
                             f"Valid fields are: {sorted(expected)}")
        kw = {k: (jnp.array(theta_input[k], dtype=getattr(default, k).dtype) if k in theta_input
                  else getattr(default, k)) for k in expected}
        return TypedIntentTheta(**kw)
    raise ValueError(f"Invalid theta_input type '{type(theta_input).__name__}'.")


def _theta_id(theta_input: Any) -> str:
    if isinstance(theta_input, str):
        return theta_input
    if isinstance(theta_input, dict):
        return f"custom_{hash(tuple(sorted(theta_input.items()))) % 10000:04d}"
    return "custom"


def make_teammate(spec: HeuristicTeammateSpec, env, teammate_agent_id: Union[str, int] = "agent_1") -> TeammatePolicy:
    """Build a typed-intent teammate for ``env`` (any object with a ``layout``, possibly wrapped)."""
    theta = resolve_theta(spec.family, spec.theta)
    e = env
    while not hasattr(e, "layout"):
        e = getattr(e, "_env", None) or getattr(e, "env")
    layout = e.layout
    if isinstance(teammate_agent_id, str) and teammate_agent_id.startswith("agent_"):
        agent_idx = int(teammate_agent_id.split("_")[1])
    else:
        agent_idx = int(teammate_agent_id)

    policy = OvercookedV2TypedIntentPolicyWrapper(
        layout=layout,
        theta=theta,
        using_log_wrapper=spec.use_log_wrapper,
        start_cooking_interaction=spec.start_cooking_interaction,
        agent_id=agent_idx,
        config=TypedIntentConfig(**spec.extra) if spec.extra else None,
    )

    def init_fn(rng=None):
        return policy.init_hstate(batch_size=1, aux_info={"agent_id": agent_idx})

    def act_fn(carry, obs, done, rng=None, env_state=None, avail_actions=None):
        done = jnp.asarray(done)
        if done.ndim == 0:
            done = done.reshape(1)
        if avail_actions is None:
            avail_actions = jnp.ones((6,), dtype=jnp.float32)
        action, new_carry = policy.get_action(params=None, obs=obs, done=done,
                                              avail_actions=avail_actions, hstate=carry,
                                              rng=rng, env_state=env_state)
        return new_carry, action

    def dist_fn(carry, env_state):
        """Exact pi_theta(. | s, m) for the current carry (host-side NumPy)."""
        return policy.dist(env_state, carry)

    return TeammatePolicy(name=f"{spec.family}[{_theta_id(spec.theta)}]", init=init_fn, act=act_fn,
                          _policy=policy, _params=None, dist_fn=dist_fn)


# ---------------------------------------------------------------------------
# Enumerated theta sets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThetaSpec:
    """One member of an enumerated family: ``theta_id`` indexes the family's grid."""

    family: str
    theta_id: int
    split: str = "study"
    base_seed: int = 0
    preset: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = field(default=None, hash=False)

    def __post_init__(self):
        _check_family(self.family)


def sample_theta(spec: ThetaSpec) -> TypedIntentTheta:
    grid = THETA_GRIDS[spec.family]()
    if spec.theta_id >= len(grid):
        raise ValueError(f"theta_id {spec.theta_id} out of range for {spec.family} grid ({len(grid)})")
    theta = TypedIntentTheta.from_preset(spec.preset) if spec.preset else grid[spec.theta_id]
    if spec.overrides:
        kw = {k: getattr(theta, k) for k in vars(theta)}
        for k, v in spec.overrides.items():
            kw[k] = jnp.array(v, dtype=kw[k].dtype)
        theta = TypedIntentTheta(**kw)
    return theta


def make_theta_set(family: str, split: str, n: int, base_seed: int = 0) -> List[ThetaSpec]:
    """The first ``n`` members of ``family``'s grid (all of it if ``n`` is 0 or too large).
    The typed-intent families are enumerated, so only ``split="study"`` exists."""
    _check_family(family)
    if split != "study":
        raise ValueError(f"family '{family}' only supports split='study'")
    if n is not None and n < 0:
        raise ValueError("n must be non-negative")
    n_grid = len(THETA_GRIDS[family]())
    n = n_grid if n is None or n == 0 or n > n_grid else n
    return [ThetaSpec(family=family, theta_id=i, split=split, base_seed=base_seed) for i in range(n)]
