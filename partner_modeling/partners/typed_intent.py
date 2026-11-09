"""Typed-intent partner for ideal-observer studies (Overcooked-V2).

A three-layer stochastic policy whose action likelihood is available in closed
form (see ``partner_modeling/README.md``):

1. **Intent layer** (stochastic, carries all type information)::

       p(i | s, m, theta) = softmax_i( beta * (w_theta[i] * v_i(s) - dw * d_i(s)
                                               + inertia * [i == m]) )

   over the intents that are valid in ``s``.  ``YIELD`` is an explicit intent
   with a fixed probability ``yield_prob`` whenever the other agent is adjacent
   (the remaining mass is spread by the softmax over the other valid intents).
2. **Controller** (deterministic, shared by all types): intent + state ->
   target -> action, built on the shortest-path tables of ``GridNav``.
3. **Noise floor** (theta-indexed)::

       pi(a) = (1 - rho - eps) * sum_i p(i) [a == ctrl_i(s)] + rho [a == stay] + eps / 6

The policy is exposed through

* ``dist(theta, view, mem) -> (probs[6], info)``   exact pi_theta(. | s, m)
* ``sample(theta, view, mem, key) -> (a, mem', info)``  a ~ Categorical(dist)
* ``update_memory(view, mem, a) -> mem'``   deterministic, theta-independent

Memory is ``last_intent`` only and is a deterministic function of the observable
history: it is set to an intent only when the
observed action is explained by exactly one "sticky" intent's controller
action.  Because the rule is theta-independent, one memory track is shared by
every hypothesis, so the observer's filter stays exact even with a switching
kernel over theta.

The partner reads only ``PartnerView`` (grid, agent positions / directions /
inventories, recipe, time), all of which appear in the env observation layers.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import chex
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np

from partner_modeling.env_compat import Actions, Direction, DynamicObject, StaticObject, state_time

from .grid_nav import GridNav


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLE_RUNNER = 0
ROLE_PLATER = 1
ROLE_FLEX = 2
# Exclusive roles:
# runner_x has w_plate = 0 and plater_x has w_fetch = 0.  A role weight of exactly 0 means
# the intent is *unavailable* to that type ("never fetches plates" / "never fetches
# ingredients"), not merely unweighted -- see ``TypedIntentModel.excluded_mask``.
ROLE_RUNNER_X = 3
ROLE_PLATER_X = 4
# Graded-exclusivity roles: the soft runner / plater weights,
# with the matching exclusive role's off-role intents masked with probability
# ``TypedIntentConfig.excl_q`` independently at every step (a per-step mixture folded into
# ``dist``).  excl_q = 0 -> soft runner / plater exactly; excl_q = 1 -> runner_x / plater_x exactly.
ROLE_RUNNER_G = 5
ROLE_PLATER_G = 6
ROLE_NAMES = ("runner", "plater", "flex", "runner_x", "plater_x", "runner_g", "plater_g")
ROLE_SHORT = ("R", "P", "F", "Rx", "Px", "Rg", "Pg")
EXCLUSIVE_ROLES = (ROLE_RUNNER_X, ROLE_PLATER_X)
GRADED_ROLES = (ROLE_RUNNER_G, ROLE_PLATER_G)
# graded role -> soft role whose weights it uses, and -> exclusive role whose exclusion set it masks
GRADED_BASE_ROLE = {ROLE_RUNNER_G: ROLE_RUNNER, ROLE_PLATER_G: ROLE_PLATER}
GRADED_EXCL_ROLE = {ROLE_RUNNER_G: ROLE_RUNNER_X, ROLE_PLATER_G: ROLE_PLATER_X}

STAGING_POT_DIRECT = 0
STAGING_COUNTER = 1
STAGING_NAMES = ("pot_direct", "counter_stager")

RELIABILITY_LEVELS = (0.05, 0.25)

INTENT_DELIVER = 0
INTENT_PICKUP_SOUP = 1      # holding plate: take cooked soup / hover near cooking pot
INTENT_GET_PLATE = 2
INTENT_ADD_INGREDIENT = 3
INTENT_FETCH_PILE = 4
INTENT_FETCH_STAGED = 5     # pick up an ingredient someone left on a counter
INTENT_STAGE = 6            # put the held item on a counter
INTENT_YIELD = 7            # stay to let the other agent pass
INTENT_IDLE = 8             # fallback when nothing else is valid
NUM_INTENTS = 9
INTENT_NAMES = (
    "deliver", "pickup_soup", "get_plate", "add_ingredient", "fetch_pile",
    "fetch_staged", "stage", "yield", "idle",
)
NON_STICKY_INTENTS = (INTENT_YIELD, INTENT_IDLE)

NUM_ACTIONS = 6
ACTION_NAMES = ("right", "down", "left", "up", "stay", "interact")

# Action -> (dy, dx)
_MOVE_DELTAS = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}


# ---------------------------------------------------------------------------
# Theta
# ---------------------------------------------------------------------------

@flax.struct.dataclass
class TypedIntentTheta:
    """Discrete partner type.  12 types = 3 roles x 2 staging styles x 2 rho."""

    role: jnp.ndarray          # int32 in {ROLE_RUNNER, ROLE_PLATER, ROLE_FLEX, ROLE_RUNNER_X, ROLE_PLATER_X}
    staging: jnp.ndarray       # int32 in {STAGING_POT_DIRECT, STAGING_COUNTER}
    reliability: jnp.ndarray   # float32 rho = pause (stay) probability

    @classmethod
    def make(cls, role: int, staging: int, rho: float) -> "TypedIntentTheta":
        return cls(
            role=jnp.array(role, dtype=jnp.int32),
            staging=jnp.array(staging, dtype=jnp.int32),
            reliability=jnp.array(rho, dtype=jnp.float32),
        )

    @classmethod
    def default(cls) -> "TypedIntentTheta":
        return cls.make(ROLE_FLEX, STAGING_POT_DIRECT, RELIABILITY_LEVELS[0])

    @staticmethod
    def grid() -> List["TypedIntentTheta"]:
        """Enumerated study grid in canonical order (role, staging, rho)."""
        return [
            TypedIntentTheta.make(r, s, rho)
            for r, s, rho in itertools.product(range(3), range(2), RELIABILITY_LEVELS)
        ]

    @staticmethod
    def grid_g() -> List["TypedIntentTheta"]:
        """Graded-exclusivity family: {runner_g, plater_g} x staging x rho = 8 types, in the same
        order as ``grid_x`` (and as the first 8 types of ``grid``).  The exclusivity level q is a
        family-level setting, ``TypedIntentConfig.excl_q``, not a type field."""
        return [
            TypedIntentTheta.make(r, s, rho)
            for r, s, rho in itertools.product(GRADED_ROLES, range(2), RELIABILITY_LEVELS)
        ]

    @staticmethod
    def grid_x() -> List["TypedIntentTheta"]:
        """Redesign family: exclusive roles {runner_x, plater_x} x staging x rho = 8 types."""
        return [
            TypedIntentTheta.make(r, s, rho)
            for r, s, rho in itertools.product(EXCLUSIVE_ROLES, range(2), RELIABILITY_LEVELS)
        ]

    def key(self) -> Tuple[int, int, float]:
        return (int(self.role), int(self.staging), round(float(self.reliability), 4))

    def preset_name(self) -> str:
        r, s, rho = self.key()
        return f"{ROLE_NAMES[r]}_{STAGING_NAMES[s]}_{'lo' if rho < 0.1 else 'hi'}"

    @classmethod
    def preset_names(cls) -> List[str]:
        return ["default"] + [t.preset_name() for t in cls.grid()]

    @classmethod
    def preset_names_x(cls) -> List[str]:
        return [t.preset_name() for t in cls.grid_x()]

    @classmethod
    def preset_names_g(cls) -> List[str]:
        return [t.preset_name() for t in cls.grid_g()]

    @classmethod
    def from_preset(cls, name: str) -> "TypedIntentTheta":
        if name == "default":
            return cls.default()
        for t in cls.grid() + cls.grid_x() + cls.grid_g():
            if t.preset_name() == name:
                return t
        raise ValueError(f"Unknown typed_intent preset {name!r}; valid: {cls.preset_names()}")

    def name(self) -> str:
        r, s, rho = self.key()
        return f"{ROLE_NAMES[r]}/{STAGING_NAMES[s]}/rho={rho:g}"

    def short(self) -> str:
        r, s, rho = self.key()
        return f"{ROLE_SHORT[r]}{'D' if s == 0 else 'S'}{'lo' if rho < 0.1 else 'hi'}"


def theta_grid_names() -> List[str]:
    return [t.short() for t in TypedIntentTheta.grid()]


def theta_grid_x_names() -> List[str]:
    return [t.short() for t in TypedIntentTheta.grid_x()]


def theta_grid_g_names() -> List[str]:
    return [t.short() for t in TypedIntentTheta.grid_g()]


THETA_FAMILIES = {"typed_intent": TypedIntentTheta.grid, "typed_intent_x": TypedIntentTheta.grid_x,
                  "typed_intent_g": TypedIntentTheta.grid_g}


# ---------------------------------------------------------------------------
# Observable state snapshot
# ---------------------------------------------------------------------------

def unwrap_env_state(env_state):
    """LogEnvState -> WrappedEnvState -> overcooked State."""
    s = env_state
    while hasattr(s, "env_state"):
        s = s.env_state
    return s


@dataclass
class PartnerView:
    """Everything the partner policy is allowed to read (all in the obs layers).

    Positions are (y, x) = (row, col).
    """

    grid: np.ndarray        # (H, W, 3) int32: static, dynamic, extra (pot timer)
    pos: np.ndarray         # (n_agents, 2) int32 (y, x)
    dir: np.ndarray         # (n_agents,) int32
    inv: np.ndarray         # (n_agents,) int32
    recipe: int
    time: int

    @classmethod
    def from_env_state(cls, env_state) -> "PartnerView":
        s = unwrap_env_state(env_state)
        pos = np.stack([np.asarray(s.agents.pos.y), np.asarray(s.agents.pos.x)], axis=-1)
        return cls(
            grid=np.asarray(s.grid, dtype=np.int32).copy(),
            pos=pos.astype(np.int32).reshape(-1, 2),
            dir=np.asarray(s.agents.dir, dtype=np.int32).reshape(-1).copy(),
            inv=np.asarray(s.agents.inventory, dtype=np.int32).reshape(-1).copy(),
            recipe=int(np.asarray(s.recipe)),
            time=state_time(s),
        )

    def key(self) -> bytes:
        return b"".join([
            self.grid.tobytes(), self.pos.tobytes(), self.dir.tobytes(),
            self.inv.tobytes(), np.int64(self.recipe).tobytes(),
        ])


@dataclass
class IntentFeatures:
    """Theta-independent per-state quantities (computed once, shared by all types)."""

    valid: np.ndarray       # (K,) bool
    value: np.ndarray       # (K,) float
    dist: np.ndarray        # (K,) float, path length to the target's access cell
    ctrl: np.ndarray        # (K,) int, controller action per intent (stay if invalid)
    comp: np.ndarray        # (K,) complementarity feature (+1 = fills the gap left by the other agent)
    forced_stage: bool      # STAGE is a forced drop (no pot can use the item)
    adjacent: bool          # other agent is adjacent
    decision_point: bool    # >= 2 non-yield intents valid with differing controls


# ---------------------------------------------------------------------------
# Hyper-parameters shared by all types
# ---------------------------------------------------------------------------

@dataclass
class TypedIntentConfig:
    beta: float = 1.0               # softmax temperature (inverse)
    eps: float = 0.03               # uniform floor over the 6 actions
    inertia: float = 1.0            # bonus for m.last_intent
    dist_weight: float = 0.5
    yield_prob: float = 0.15        # P(YIELD) when a move may be contested
    yield_radius: int = 2           # YIELD valid when the other agent is within this distance
    anticipate: bool = True         # plates also count soups the other agent is about to complete
    flex_complement: float = 0.0    # flex role: bonus for the complementary task (tried 3.0: no identifiability gain)
    controller: str = "aware"       # "aware" (routes around the other agent) | "shortest" (static shortest paths)
    blocked_penalty: float = 4.0    # aware: extra distance when every access cell is occupied
    make_way_station_only: bool = False   # YIELD steps aside only from station-access cells (redesign fix)
    # Graded exclusivity (runner_g / plater_g only; ignored by every other role): per-step
    # probability that the type's off-role intents are masked.  Must be set for graded types.
    excl_q: Optional[float] = None

    # Role weights: (w_fetch, w_plate) for runner, plater, flex, runner_x, plater_x.
    # A weight of exactly 0 makes the intent unavailable to that role (exclusive roles).
    w_role: Tuple[Tuple[float, float], ...] = ((6.0, 1.5), (1.5, 6.0), (4.0, 4.0), (6.0, 0.0), (0.0, 6.0))
    # Staging weights: (w_add, w_stage, fetch_staged_offset).  The offset is added to the
    # role's w_fetch for FETCH_STAGED.  (A multiplicative 0.1 discount was tried first: for
    # the plater role, w_fetch = 1.5, it left a gap of only 1.35, so plater/stagers juggled
    # their own staged onion between hand and counter.)
    w_staging: Tuple[Tuple[float, float, float], ...] = ((6.0, 1.0, 0.0), (2.0, 6.0, -4.0))
    w_deliver: float = 10.0
    w_pickup: float = 8.0
    w_forced: float = 5.0           # weight for forced drops / idle (only option anyway)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class TypedIntentModel(GridNav):
    """Exact-likelihood partner, built on the shortest-path tables of ``GridNav``."""

    def __init__(self, layout, agent_id: int = 1, config: Optional[TypedIntentConfig] = None,
                 start_cooking_interaction: bool = False):
        super().__init__(layout)
        if start_cooking_interaction:
            raise NotImplementedError("typed_intent assumes auto-cooking pots")
        self.agent_id = int(agent_id)
        self.cfg = config if config is not None else TypedIntentConfig()
        self._feat_cache: Dict[bytes, IntentFeatures] = {}
        self._precompute_typed()

    # ------------------------------------------------------------ precompute
    def _precompute_typed(self):
        H, W = self.height, self.width
        walk = self.walkable_mask_np
        n_walk_nb = np.zeros((H, W), dtype=np.int32)
        for y in range(H):
            for x in range(W):
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ay, ax = y + dy, x + dx
                    if 0 <= ay < H and 0 <= ax < W and walk[ay, ax]:
                        n_walk_nb[y, x] += 1
        # Staging counters: "hub" counters touching >= 3 walkable cells (hand-off points
        # between the ingredient side and the pot side).  Fall back to pot-adjacent
        # counters only if the layout has no hub.  (v1 also used pot-adjacent counters
        # in coord_ring; stagers then walked past the pot to stage next to it.)
        hub = self.counter_mask_np & (n_walk_nb >= 3)
        self.staging_counter_mask_np = hub if np.any(hub) else (
            self.counter_mask_np & self.pot_adjacent_counter_mask_np & (n_walk_nb >= 1))
        self.accessible_counter_mask_np = self.counter_mask_np & (n_walk_nb >= 1)
        stations = self.pot_mask_np | self.ingredient_pile_mask_np | self.plate_pile_mask_np | self.goal_mask_np
        self.station_access_np = np.zeros((H, W), dtype=bool)
        for (y, x) in map(tuple, np.argwhere(stations)):
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ay, ax = y + dy, x + dx
                if 0 <= ay < H and 0 <= ax < W and walk[ay, ax]:
                    self.station_access_np[ay, ax] = True

        # Connected components of the walkable graph (split layouts: forced_coord,
        # two_rooms, fun_symmetries*, ...).  ``comp_reach[c]`` = non-walkable cells adjacent
        # to component c; ``counter_comps[y, x]`` = set of components touching a counter.
        # Used only by the multi-room generalisation rules below; on single-component
        # layouts (coord_ring, cramped_room, counter_circuit) none of them fire.
        self.comp_id_np = np.full((H, W), -1, dtype=np.int32)
        nc = 0
        for (sy, sx) in map(tuple, np.argwhere(walk)):
            if self.comp_id_np[sy, sx] >= 0:
                continue
            q = [(sy, sx)]
            self.comp_id_np[sy, sx] = nc
            while q:
                cy, cx = q.pop()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and walk[ny, nx] and self.comp_id_np[ny, nx] < 0:
                        self.comp_id_np[ny, nx] = nc
                        q.append((ny, nx))
            nc += 1
        self.n_comps = nc
        self.comp_adj_np = np.zeros((max(nc, 1), H, W), dtype=bool)   # cell adjacent to comp c
        for (y, x) in map(tuple, np.argwhere(~walk)):
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ay, ax = y + dy, x + dx
                if 0 <= ay < H and 0 <= ax < W and walk[ay, ax]:
                    self.comp_adj_np[self.comp_id_np[ay, ax], y, x] = True
        # Parking fallback classes (lower = better): 0 non-station cell, 1 station access
        # cell not adjacent to a pot or goal, 2 anything else.  Class 1 is only used when
        # no class-0 cell is free (cramped_room: the single non-station cell is taken).
        pg = self.pot_mask_np | self.goal_mask_np
        pg_access = np.zeros((H, W), dtype=bool)
        for (y, x) in map(tuple, np.argwhere(pg)):
            for (ay, ax) in self._access_cells(y, x):
                pg_access[ay, ax] = True
        self.park_class_np = np.where(~self.station_access_np, 0, np.where(~pg_access, 1, 2))

        # Agent-aware path tables: for every possible blocked walkable cell b,
        # BFS on the walkable graph with b removed (used by controller="aware").
        self._aware_dist = {}
        self._aware_next = {}
        walk_cells = [tuple(c) for c in np.argwhere(walk)]
        for b in walk_cells:
            d, nxt = self._bfs_tables(blocked=b)
            self._aware_dist[b] = d
            self._aware_next[b] = nxt

    def _bfs_tables(self, blocked):
        H, W = self.height, self.width
        walk = self.walkable_mask_np.copy()
        walk[blocked] = False
        INF = 10000
        n = len(self.idx_to_pos)
        dist = np.full((n, H, W), INF, dtype=np.int32)
        nxt = np.full((n, H, W), Actions.stay, dtype=np.int32)
        for si, (sy, sx) in enumerate(self.idx_to_pos):
            if (sy, sx) == tuple(blocked):
                continue
            d = np.full((H, W), INF, dtype=np.int32)
            fs = np.full((H, W), -1, dtype=np.int32)
            d[sy, sx] = 0
            q = [(sy, sx)]
            h = 0
            while h < len(q):
                cy, cx = q[h]
                h += 1
                for a, (dy, dx) in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0))):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and walk[ny, nx] and d[ny, nx] == INF:
                        d[ny, nx] = d[cy, cx] + 1
                        fs[ny, nx] = a if (cy, cx) == (sy, sx) else fs[cy, cx]
                        q.append((ny, nx))
            dist[si] = d
            ok = fs >= 0
            nxt[si][ok] = fs[ok]
        return dist, nxt

    # ------------------------------------------------------------ state helpers
    def _pot_info(self, grid):
        _, pot_non_full, pot_cooking, pot_cooked, _, counts = self.pot_states(grid)
        return pot_non_full, pot_cooking, pot_cooked, counts

    @staticmethod
    def _inv_flags(inv: int):
        dish = bool(inv & DynamicObject.COOKED) and bool(inv & DynamicObject.PLATE)
        plate = inv == DynamicObject.PLATE
        ingredient = ((inv >> 2) != 0) and ((inv & DynamicObject.PLATE) == 0)
        empty = inv == DynamicObject.EMPTY
        return empty, ingredient, plate, dish

    def _access_cells(self, ty, tx):
        out = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ay, ax = ty + dy, tx + dx
            if 0 <= ay < self.height and 0 <= ax < self.width and self.walkable_mask_np[ay, ax]:
                out.append((ay, ax))
        return out

    def _face_action(self, py, px, ty, tx):
        dy, dx = ty - py, tx - px
        if dy == -1:
            return Actions.up
        if dy == 1:
            return Actions.down
        if dx == 1:
            return Actions.right
        return Actions.left

    @staticmethod
    def _dir_of_action(a):
        return {Actions.up: Direction.UP, Actions.down: Direction.DOWN,
                Actions.right: Direction.RIGHT, Actions.left: Direction.LEFT}[a]

    # ------------------------------------------------------------ controllers
    def _ctrl_to_target(self, mask, py, px, pdir, other, hover=False):
        """Deterministic action toward the closest cell of ``mask`` (non-walkable target).

        Returns (action, path_length) or (None, inf) when unreachable.
        """
        reach = self.reachable_targets(mask, py, px)
        if not np.any(reach):
            return None, np.inf
        dist = self.min_dist_to_target(reach, py, px)
        if self.cfg.controller == "shortest":
            ty, tx, ok = self.closest_target(reach, py, px)
            if not ok:
                return None, np.inf
            if hover:
                a = self.step_next_to(py, px, pdir, ty, tx, other)
            else:
                a = self.step_to_station(py, px, pdir, ty, tx, other)
            return int(a), float(dist)
        return self._aware_ctrl(reach, py, px, pdir, other, hover), self._aware_dist_to(reach, py, px, other)

    def _aware_dist_to(self, reach, py, px, other) -> float:
        """Path length to the nearest free access cell, routing around the other agent.
        If every access cell is occupied/unreachable: static distance + ``blocked_penalty``."""
        oy, ox = int(other[0][0]), int(other[0][1])
        idx = self.pos_to_idx[py, px]
        dtab = self._aware_dist[(oy, ox)]
        best, best_static = 10000, 10000
        for (ty, tx) in map(tuple, np.argwhere(reach)):
            for (ay, ax) in self._access_cells(ty, tx):
                best_static = min(best_static, self.dist_matrix[idx, ay, ax])
                if (ay, ax) != (oy, ox):
                    best = min(best, dtab[idx, ay, ax])
        if best < 10000:
            return float(best)
        return float(best_static + self.cfg.blocked_penalty)

    def _aware_ctrl(self, reach, py, px, pdir, other, hover):
        """Agent-aware controller (``controller="aware"``).

        * plans on the walkable graph with the other agent's cell removed (routes
          around it instead of bumping into it);
        * never targets an access cell the other agent stands on; if every goal cell
          is occupied (or only reachable through the other agent) it *parks* on the
          nearest free non-station cell and waits there.  (Earlier versions approached
          to distance 1 and stepped aside when standing on a station cell; against a
          static agent sitting on the only access cell this produced a deterministic
          period-2 loop.)
        * ``hover`` (holding a plate while the soup cooks): waits on a free non-station
          cell next to the access cell instead of on it, so the access cell stays usable.
        """
        oy, ox = int(other[0][0]), int(other[0][1])
        idx = self.pos_to_idx[py, px]
        targets = [tuple(t) for t in np.argwhere(reach)]
        if not hover:
            for (ty, tx) in targets:
                if abs(ty - py) + abs(tx - px) == 1:
                    fa = self._face_action(py, px, ty, tx)
                    return Actions.interact if pdir == self._dir_of_action(fa) else int(fa)
        access = sorted({c for (ty, tx) in targets for c in self._access_cells(ty, tx)})
        if hover:
            goal_cells = sorted({(ny, nx) for (ay, ax) in access for (ny, nx) in self._walk_nbrs(ay, ax)
                                 if (ny, nx) not in access and not self.station_access_np[ny, nx]})
            if not goal_cells:
                goal_cells = access
            if (py, px) in goal_cells:
                return Actions.stay
        else:
            goal_cells = access
        dtab = self._aware_dist[(oy, ox)]
        ntab = self._aware_next[(oy, ox)]
        best, best_d = None, 10000
        for (ay, ax) in goal_cells:
            if (ay, ax) == (oy, ox):
                continue
            d = dtab[idx, ay, ax]
            if d < best_d:
                best, best_d = (ay, ax), d
        if best is not None and best_d < 10000:
            if (py, px) == best:
                return Actions.stay
            return int(ntab[idx, best[0], best[1]])
        return self._park_action(py, px, other, force=True)

    def _walk_nbrs(self, y, x):
        out = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width and self.walkable_mask_np[ny, nx]:
                out.append((ny, nx))
        return out

    def _step_aside(self, py, px, oy, ox):
        """Move to a free neighbouring cell, preferring non-station cells (deterministic)."""
        cands = []
        for a, (dy, dx) in _MOVE_DELTAS.items():
            ny, nx = py + dy, px + dx
            if (0 <= ny < self.height and 0 <= nx < self.width and self.walkable_mask_np[ny, nx]
                    and (ny, nx) != (oy, ox)):
                cands.append((int(self.station_access_np[ny, nx]), a))
        if not cands:
            return Actions.stay
        return int(min(cands)[1])

    def _park_action(self, py, px, other, force: bool = False):
        """Go to the nearest free non-station cell and stay there (IDLE, blocked goals)."""
        if (self.cfg.controller != "aware" and not force) or not self.station_access_np[py, px]:
            return Actions.stay
        oy, ox = int(other[0][0]), int(other[0][1])
        idx = self.pos_to_idx[py, px]
        dtab, ntab = self._aware_dist[(oy, ox)], self._aware_next[(oy, ox)]
        best, bd = None, 10000
        for (y, x) in map(tuple, np.argwhere(self.walkable_mask_np & ~self.station_access_np)):
            if (y, x) != (oy, ox) and dtab[idx, y, x] < bd:
                best, bd = (y, x), dtab[idx, y, x]
        if best is None or bd >= 10000:
            # Generalisation (cramped_room): no free non-station cell is reachable.  Rather
            # than wait on a pot/goal access cell (which deadlocks against an agent waiting
            # for that cell), move to the nearest free cell of class 1 (station access not
            # next to a pot/goal); stay if already on one.
            if self.park_class_np[py, px] <= 1:
                return Actions.stay
            for (y, x) in map(tuple, np.argwhere(self.walkable_mask_np & (self.park_class_np == 1))):
                if (y, x) != (oy, ox) and dtab[idx, y, x] < bd:
                    best, bd = (y, x), dtab[idx, y, x]
            if best is None or bd >= 10000:
                return Actions.stay
        return int(ntab[idx, best[0], best[1]])

    # ------------------------------------------------------------ multi-room helpers
    def _comp_of(self, py, px) -> int:
        return int(self.comp_id_np[py, px])

    def _usable_here(self, mask, py, px) -> bool:
        """Some cell of ``mask`` is adjacent to my component."""
        return bool(np.any(mask & self.comp_adj_np[self._comp_of(py, px)]))

    def _handoff_counters(self, empty_counters, use_mask, py, px):
        """Empty counters adjacent to my component AND to another component from which
        ``use_mask`` is reachable (where the item I hold can be used)."""
        me = self._comp_of(py, px)
        out = np.zeros_like(empty_counters)
        for c in range(self.n_comps):
            if c != me and np.any(use_mask & self.comp_adj_np[c]):
                out |= self.comp_adj_np[c]
        return out & self.comp_adj_np[me] & empty_counters

    def _needed_pile_mask(self, grid, recipe, pot_non_full):
        """Piles of ingredient types still missing from some idle, non-full pot.

        (Counted per idle pot: summing over all pots, including cooking ones, would stop
        fetching for the second pot while the first one cooks.)
        """
        need_types = np.zeros(4, dtype=np.int32)
        recipe_counts = self.decode_recipe(int(recipe))
        for (y, x) in map(tuple, np.argwhere(pot_non_full)):
            content = int(grid[y, x, 1]) >> 2
            have = np.array([(content >> (2 * i)) & 0x3 for i in range(4)])
            need_types = np.maximum(need_types, recipe_counts - have)
        mask = np.zeros((self.height, self.width), dtype=bool)
        static = grid[:, :, 0]
        for i in range(4):
            if need_types[i] > 0:
                mask |= static == (StaticObject.INGREDIENT_PILE_BASE + i)
        return mask

    # ------------------------------------------------------------ features
    def features(self, view: PartnerView) -> IntentFeatures:
        """theta-independent features of the *current* state.

        Path tables are keyed on the current static grid (``view.grid[..., 0]``): if the
        static layout differs from the one this model was built for (e.g. a wall added
        mid-episode), features are computed by a child model whose path tables were built
        from that grid, so the controller reroutes and path costs (hence intent scores and
        ``dist``) change.  Children are cached per static layout.
        """
        static = view.grid[:, :, 0]
        if static.shape != self.static_objects_np.shape or not np.array_equal(static, self.static_objects_np):
            return self._model_for_static(static).features(view)
        k = view.key()
        f = self._feat_cache.get(k)
        if f is None:
            f = self._compute_features(view)
            if len(self._feat_cache) > 200_000:
                self._feat_cache.clear()
            self._feat_cache[k] = f
        return f

    def _model_for_static(self, static: np.ndarray) -> "TypedIntentModel":
        if not hasattr(self, "_static_children"):
            self._static_children = {}
        key = static.astype(np.int32).tobytes() + bytes(str(static.shape), "ascii")
        child = self._static_children.get(key)
        if child is None:
            from types import SimpleNamespace
            child = TypedIntentModel(SimpleNamespace(static_objects=np.asarray(static, dtype=np.int32)),
                                     agent_id=self.agent_id, config=self.cfg)
            self._static_children[key] = child
        return child

    def _compute_features(self, view: PartnerView) -> IntentFeatures:
        K = NUM_INTENTS
        valid = np.zeros(K, dtype=bool)
        value = np.zeros(K, dtype=np.float64)
        dist = np.zeros(K, dtype=np.float64)
        ctrl = np.full(K, Actions.stay, dtype=np.int64)

        me = self.agent_id
        others = [i for i in range(view.pos.shape[0]) if i != me]
        py, px = int(view.pos[me, 0]), int(view.pos[me, 1])
        pdir = int(view.dir[me])
        inv = int(view.inv[me])
        other_pos = np.array([[int(view.pos[o, 0]), int(view.pos[o, 1])] for o in others],
                             dtype=np.int32).reshape(-1, 2)
        grid = view.grid
        empty, ingredient, plate, dish = self._inv_flags(inv)

        pot_non_full, pot_cooking, pot_cooked, counts = self._pot_info(grid)
        n_active = int(np.sum(pot_cooking | pot_cooked))
        any_cooked = bool(np.any(pot_cooked))
        any_cooking = bool(np.any(pot_cooking))

        dyn = grid[:, :, 1]
        counter_items = self.counter_mask_np & (dyn != DynamicObject.EMPTY)
        counter_ingredient = counter_items & ((dyn >> 2) != 0) & ((dyn & DynamicObject.PLATE) == 0)
        counter_plate = counter_items & (dyn == DynamicObject.PLATE)
        empty_counters = self.accessible_counter_mask_np & (dyn == DynamicObject.EMPTY)
        empty_staging = self.staging_counter_mask_np & (dyn == DynamicObject.EMPTY)

        other_inv = [int(view.inv[o]) for o in others]
        other_ing = sum(1 for v in other_inv if self._inv_flags(v)[1])
        other_plates = sum(1 for v in other_inv if self._inv_flags(v)[2])
        need_ing = int(np.sum(np.where(pot_non_full, 3 - counts, 0))) - other_ing
        # "Soup coming": an idle pot that the other agent's carried ingredient will complete.
        pot_coming = pot_non_full & (counts + other_ing >= 3) & (counts > 0)
        if other_ing == 0 or not self.cfg.anticipate:
            pot_coming = np.zeros_like(pot_non_full)
        any_coming = bool(np.any(pot_coming))
        plates_needed = n_active + int(np.sum(pot_coming)) - other_plates

        # Multi-room generalisation (only when the walkable graph has > 1 component).
        # An agent whose room cannot reach a pot (two_rooms, forced_coord supply side)
        # fetches for hand-off only: it does not re-collect items already staged on
        # counters (that was a pick-up/drop juggling loop), counts staged items as
        # supplied, and drops held items on a counter shared with a room where they can
        # be used.  Dishes are handed off the same way when no goal is reachable.
        multi = self.n_comps > 1
        plate_src = self.plate_pile_mask_np | counter_plate
        supply_only = multi and not self._usable_here(self.pot_mask_np, py, px)
        can_goal = (not multi) or self._usable_here(self.goal_mask_np, py, px)
        if supply_only:
            plate_src = self.plate_pile_mask_np
            plates_needed -= int(np.sum(counter_plate))
            need_ing -= int(np.sum(counter_ingredient))
            # keep >= 1 hand-off counter free for dishes coming back (two_rooms_both: the
            # supply side filled every divider counter and the dish was dropped elsewhere)
            if int(np.sum(self._handoff_counters(empty_counters, self.pot_mask_np, py, px))) < 2:
                plates_needed = 0
                need_ing = 0

        def setk(i, a_d, v):
            a, d = a_d
            if a is None:
                return
            valid[i] = True
            ctrl[i] = int(a)
            dist[i] = d
            value[i] = v

        counter_dish = counter_items & ((dyn & DynamicObject.COOKED) != 0) & ((dyn & DynamicObject.PLATE) != 0)
        if dish:
            setk(INTENT_DELIVER, self._ctrl_to_target(self.goal_mask_np, py, px, pdir, other_pos), 1.0)
        elif empty and np.any(counter_dish) and can_goal:
            # a dish left on a counter (e.g. dropped by a noise "interact"): go pick it up
            setk(INTENT_DELIVER, self._ctrl_to_target(counter_dish, py, px, pdir, other_pos), 1.0)

        if plate:
            if any_cooked:
                setk(INTENT_PICKUP_SOUP, self._ctrl_to_target(pot_cooked, py, px, pdir, other_pos), 1.0)
            elif any_cooking or any_coming:
                setk(INTENT_PICKUP_SOUP,
                     self._ctrl_to_target(pot_cooking | pot_coming, py, px, pdir, other_pos, hover=True), 1.0)

        if empty and plates_needed > 0:
            v_plate = 1.5 if any_cooked else (1.0 if any_cooking else 0.6)
            setk(INTENT_GET_PLATE,
                 self._ctrl_to_target(plate_src, py, px, pdir, other_pos),
                 v_plate)

        if ingredient and np.any(pot_non_full):
            setk(INTENT_ADD_INGREDIENT, self._ctrl_to_target(pot_non_full, py, px, pdir, other_pos), 1.0)

        if empty and need_ing > 0:
            piles = self._needed_pile_mask(grid, view.recipe, pot_non_full)
            if np.any(piles):
                setk(INTENT_FETCH_PILE, self._ctrl_to_target(piles, py, px, pdir, other_pos), 1.0)
            if np.any(counter_ingredient) and not supply_only:
                setk(INTENT_FETCH_STAGED,
                     self._ctrl_to_target(counter_ingredient, py, px, pdir, other_pos), 1.0)

        forced = False
        if ingredient and valid[INTENT_ADD_INGREDIENT] and np.any(empty_staging):
            setk(INTENT_STAGE, self._ctrl_to_target(empty_staging, py, px, pdir, other_pos), 1.0)
        elif ((ingredient and not valid[INTENT_ADD_INGREDIENT]) or (plate and not valid[INTENT_PICKUP_SOUP])
              or (dish and not valid[INTENT_DELIVER])):
            # Item is useless right now: drop it on the closest free counter.  Multi-room:
            # if it cannot be used from this room at all, drop it on a hand-off counter
            # shared with a room where it can (fallback: closest free counter).
            target = empty_counters
            if multi:
                use = self.goal_mask_np if dish else self.pot_mask_np
                if not self._usable_here(use, py, px):
                    ho = self._handoff_counters(empty_counters, use, py, px)
                    if np.any(ho) or dish:
                        target = ho      # a dish is held rather than dropped out of reach
            if np.any(target):
                forced = True
                setk(INTENT_STAGE, self._ctrl_to_target(target, py, px, pdir, other_pos), 1.0)

        dmin = min((abs(int(o[0]) - py) + abs(int(o[1]) - px) for o in other_pos), default=99)
        adjacent = dmin == 1
        # YIELD is available whenever a move could be contested (other agent within 2
        # cells, i.e. both could step into the same cell).  Adjacent-only yield (v1) left head-on collisions over a
        # shared cell unresolved.
        contested = dmin <= self.cfg.yield_radius
        if contested:
            valid[INTENT_YIELD] = True
            ctrl[INTENT_YIELD] = Actions.stay
            value[INTENT_YIELD] = 1.0

        non_yield = valid.copy()
        non_yield[INTENT_YIELD] = False
        # IDLE's controller (park) is always computed: an exclusive-role type whose every
        # valid intent is unavailable to it falls back to IDLE (``_intent_probs_batch``).
        # For the original 12 types IDLE only gets probability when it is valid here, so
        # filling ctrl/value while it is invalid changes nothing for them.
        ctrl[INTENT_IDLE] = self._park_action(py, px, other_pos)
        value[INTENT_IDLE] = 1.0
        if not np.any(non_yield):
            valid[INTENT_IDLE] = True

        # YIELD = make way: if the adjacent other agent faces our cell (it may be trying
        # to move into us), yielding means stepping aside; otherwise it means staying.
        # This is stochastic (YIELD has probability yield_prob), unlike the deterministic
        # make-way override tried earlier, which looped against static agents whose
        # facing direction carries no intent.
        if valid[INTENT_YIELD] and adjacent and self.cfg.controller == "aware" and len(others) == 1:
            o = others[0]
            od = int(view.dir[o])
            fy, fx = int(view.pos[o, 0]) + (-1, 1, 0, 0)[od], int(view.pos[o, 1]) + (0, 0, 1, -1)[od]
            # Only make way when standing on a station-access cell (the case this exists
            # for: unblocking a pot / pile / goal the other agent needs).  In a 1-wide
            # corridor of a ring (coord_ring, counter_circuit) the only free neighbour is
            # behind us, so "step aside" equalled the agent-aware detour move and YIELD could
            # never break a head-on period-2 livelock (redesign round, 2026-11-08: a 55-step
            # loop to the end of the episode on counter_circuit).  With
            # ``make_way_station_only`` YIELD stays there instead.  Off by default so the N1 /
            # Phase-0 runs of the original family stay bit-reproducible.
            if (fy, fx) == (py, px) and (self.station_access_np[py, px] or not self.cfg.make_way_station_only):
                ctrl[INTENT_YIELD] = self._step_aside(py, px, int(view.pos[o, 0]), int(view.pos[o, 1]))

        ny = valid.copy()
        ny[INTENT_YIELD] = False
        decision = int(np.sum(ny)) >= 2 and len(set(ctrl[ny].tolist())) >= 2
        # Complementarity (used by the flex role): plates are the complementary task when
        # the other agent carries an ingredient; ingredients when it carries a plate/dish.
        comp = np.zeros(K, dtype=np.float64)
        o_ing = other_ing > 0
        o_pl = any(self._inv_flags(v)[2] or self._inv_flags(v)[3] for v in other_inv)
        comp[INTENT_GET_PLATE] = float(o_ing) - float(o_pl)
        comp[INTENT_FETCH_PILE] = comp[INTENT_FETCH_STAGED] = float(o_pl) - float(o_ing)
        return IntentFeatures(valid=valid, value=value, dist=dist, ctrl=ctrl, comp=comp,
                              forced_stage=forced, adjacent=adjacent, decision_point=decision)

    # ------------------------------------------------------------ theta weights
    def weight_vector(self, theta) -> np.ndarray:
        r, s, _ = _theta_tuple(theta)
        c = self.cfg
        w_fetch, w_plate = c.w_role[GRADED_BASE_ROLE.get(r, r)]
        w_add, w_stage, staged_offset = c.w_staging[s]
        w = np.zeros(NUM_INTENTS, dtype=np.float64)
        w[INTENT_DELIVER] = c.w_deliver
        w[INTENT_PICKUP_SOUP] = c.w_pickup
        w[INTENT_GET_PLATE] = w_plate
        w[INTENT_ADD_INGREDIENT] = w_add
        w[INTENT_FETCH_PILE] = w_fetch
        w[INTENT_FETCH_STAGED] = w_fetch + staged_offset
        w[INTENT_STAGE] = w_stage
        w[INTENT_IDLE] = c.w_forced
        return w

    def excluded_mask(self, theta) -> np.ndarray:
        """(K,) bool: intents this type never chooses.  A role weight of exactly 0 (the
        exclusive roles runner_x / plater_x) removes the intent instead of leaving it at
        score -dist (which would still be chosen whenever it is the only valid intent).
        All-False for the original 12 types and for the graded types (see ``graded_mask``)."""
        r = _theta_tuple(theta)[0]
        if r in GRADED_ROLES:
            return np.zeros(NUM_INTENTS, dtype=bool)
        w_fetch, w_plate = self.cfg.w_role[r]
        x = np.zeros(NUM_INTENTS, dtype=bool)
        if w_plate == 0.0:
            x[INTENT_GET_PLATE] = True
        if w_fetch == 0.0:
            x[INTENT_FETCH_PILE] = True
            x[INTENT_FETCH_STAGED] = True
        return x

    def graded_mask(self, theta) -> np.ndarray:
        """(K,) bool: intents masked with per-step probability ``cfg.excl_q`` (graded roles only:
        the exclusion set of the matching exclusive role).  All-False for every other role."""
        r = _theta_tuple(theta)[0]
        if r not in GRADED_ROLES:
            return np.zeros(NUM_INTENTS, dtype=bool)
        return self.excluded_mask((GRADED_EXCL_ROLE[r], 0, 0.0))

    def excl_q(self) -> float:
        q = self.cfg.excl_q
        if q is None or not (0.0 <= float(q) <= 1.0):
            raise ValueError(f"graded types need TypedIntentConfig.excl_q in [0, 1], got {q!r}")
        return float(q)

    def _intent_probs_graded(self, Wmat: np.ndarray, f: IntentFeatures, mem: int, ccoef: np.ndarray,
                             excl: np.ndarray, gmask: np.ndarray) -> np.ndarray:
        """Intent probs with graded exclusivity: rows with a graded mask get
        q * p(. | off-role intents masked) + (1 - q) * p(. | unmasked), the exact marginal of a
        per-step Bernoulli(q) mask.  q = 0 / q = 1 return the unmasked / masked rows unchanged
        (bit-identical to the soft / exclusive roles).  Rows without a graded mask, and every call
        for the original families, take the unchanged ``_intent_probs_batch`` path."""
        P = self._intent_probs_batch(Wmat, f, mem, ccoef, excl)
        if gmask is None or not np.any(gmask):
            return P
        q = self.excl_q()
        if q == 0.0:
            return P
        rows = np.any(gmask, axis=1)
        Pm = self._intent_probs_batch(Wmat[rows], f, mem, ccoef[rows], excl[rows] | gmask[rows])
        P = P.copy()
        P[rows] = Pm if q == 1.0 else q * Pm + (1.0 - q) * P[rows]
        return P

    def comp_coef(self, theta) -> float:
        """Weight on the complementarity feature: only the flex role uses it."""
        return self.cfg.flex_complement if _theta_tuple(theta)[0] == ROLE_FLEX else 0.0

    def intent_probs_from_features(self, theta, f: IntentFeatures, mem: int) -> np.ndarray:
        return self._intent_probs_graded(self.weight_vector(theta)[None], f, mem,
                                         np.array([self.comp_coef(theta)]),
                                         self.excluded_mask(theta)[None], self.graded_mask(theta)[None])[0]

    def _intent_probs_batch(self, Wmat: np.ndarray, f: IntentFeatures, mem: int,
                            ccoef: Optional[np.ndarray] = None,
                            excl: Optional[np.ndarray] = None) -> np.ndarray:
        """Wmat: (T, K) weights, ccoef: (T,) complementarity coefs, excl: (T, K) intents
        unavailable to each type -> (T, K) intent probs."""
        c = self.cfg
        T = Wmat.shape[0]
        main = f.valid.copy()
        main[INTENT_YIELD] = False
        main = np.broadcast_to(main, (T, NUM_INTENTS)).copy()
        if excl is not None and np.any(excl):
            main &= ~excl
            none_left = ~np.any(main, axis=1)
            main[none_left, INTENT_IDLE] = True     # nothing this type would do: park / idle
        W = Wmat.copy()
        if f.forced_stage:
            W[:, INTENT_STAGE] = c.w_forced
        scores = W * f.value[None] - c.dist_weight * f.dist[None]
        if ccoef is not None:
            scores = scores + ccoef[:, None] * f.comp[None]
        if 0 <= mem < NUM_INTENTS:
            scores[:, mem] += np.where(main[:, mem], c.inertia, 0.0)
        scores = np.where(main, c.beta * scores, -np.inf)
        mx = np.max(scores, axis=1, keepdims=True)
        e = np.where(main, np.exp(scores - mx), 0.0)
        p = e / np.sum(e, axis=1, keepdims=True)
        if f.valid[INTENT_YIELD]:
            p = p * (1.0 - c.yield_prob)
            p[:, INTENT_YIELD] = c.yield_prob
        return p

    def _action_probs_batch(self, P: np.ndarray, rho: np.ndarray, f: IntentFeatures) -> np.ndarray:
        eps = self.cfg.eps
        C = np.zeros((NUM_INTENTS, NUM_ACTIONS))
        C[np.arange(NUM_INTENTS), f.ctrl] = 1.0
        A = (1.0 - rho - eps)[:, None] * (P @ C)
        A[:, Actions.stay] += rho
        A += eps / NUM_ACTIONS
        return A

    # ------------------------------------------------------------ public API
    def dist(self, theta, view: PartnerView, mem: int):
        """Exact pi_theta(a | s, m).  Returns (probs[6], info dict)."""
        f = self.features(view)
        X = self.excluded_mask(theta)
        G = self.graded_mask(theta)
        P = self._intent_probs_graded(self.weight_vector(theta)[None], f, mem,
                                      np.array([self.comp_coef(theta)]), X[None], G[None])
        rho = np.array([_theta_tuple(theta)[2]])
        A = self._action_probs_batch(P, rho, f)[0]
        valid = f.valid.copy()
        if np.any(G):
            # graded: q = 1 -> availability as for runner_x / plater_x; 0 < q < 1 -> every valid
            # intent stays reachable, plus the masked component's IDLE fallback; q = 0 -> unchanged
            q = self.excl_q()
            if q == 1.0:
                X = X | G
            elif q > 0.0:
                valid = valid | (P[0] > 0)
        if np.any(X):      # per-type availability (exclusive roles)
            valid = (valid & ~X) | (P[0] > 0)
        info = {"intent_probs": P[0], "ctrl": f.ctrl.copy(), "valid": valid,
                "decision_point": f.decision_point, "adjacent": f.adjacent}
        return A, info

    def dist_all(self, thetas: Sequence, view: PartnerView, mem: int):
        """Vectorised over a theta list: (A[T,6], P_intent[T,K], features)."""
        f = self.features(view)
        Wmat = np.stack([self.weight_vector(t) for t in thetas])
        rho = np.array([_theta_tuple(t)[2] for t in thetas])
        X = np.stack([self.excluded_mask(t) for t in thetas])
        G = np.stack([self.graded_mask(t) for t in thetas])
        P = self._intent_probs_graded(Wmat, f, mem, np.array([self.comp_coef(t) for t in thetas]), X, G)
        A = self._action_probs_batch(P, rho, f)
        return A, P, f

    def update_memory(self, view: PartnerView, mem: int, action: int, f: Optional[IntentFeatures] = None) -> int:
        """Deterministic, theta-independent memory update from (s, m, a)."""
        if f is None:
            f = self.features(view)
        action = int(action)
        if 0 <= mem < NUM_INTENTS and f.valid[mem] and mem not in NON_STICKY_INTENTS and f.ctrl[mem] == action:
            return int(mem)
        expl = [i for i in range(NUM_INTENTS)
                if f.valid[i] and i not in NON_STICKY_INTENTS and f.ctrl[i] == action]
        if len(expl) == 1:
            return int(expl[0])
        if len(expl) > 1:
            return -1
        return int(mem)

    def sample(self, theta, view: PartnerView, mem: int, key):
        """a ~ Categorical(dist(theta, view, mem)); the only sampler."""
        probs, info = self.dist(theta, view, mem)
        u = uniform_from_key(key)
        a = int(np.searchsorted(np.cumsum(probs), u * probs.sum(), side="right"))
        a = min(a, NUM_ACTIONS - 1)
        new_mem = self.update_memory(view, mem, a)
        info["probs"] = probs
        return a, new_mem, info


def _theta_tuple(theta) -> Tuple[int, int, float]:
    if isinstance(theta, tuple):
        return int(theta[0]), int(theta[1]), float(theta[2])
    return int(theta.role), int(theta.staging), float(theta.reliability)


def uniform_from_key(key) -> float:
    """Map a JAX PRNG key (raw uint32[2] or typed key) to U[0,1).

    The key bits are hashed through numpy's SeedSequence (not used raw), so distinct
    keys give independent uniforms and the same key always gives the same draw.
    """
    if hasattr(key, "dtype") and jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        key = jax.random.key_data(key)
    k = np.asarray(key, dtype=np.uint32).ravel().tolist()
    return float(np.random.default_rng(k).random())


# ---------------------------------------------------------------------------
# Policy wrapper (registry / jit-compatible via pure_callback)
# ---------------------------------------------------------------------------

@chex.dataclass
class TypedIntentAgentState:
    agent_id: jnp.ndarray     # int32
    last_intent: jnp.ndarray  # int32, -1 = none


class OvercookedV2TypedIntentPolicyWrapper:
    """AgentPolicy-style wrapper.  Uses the ``rng`` passed by the env loop on every
    step (no key is stored in the carry) and resets memory on ``done``."""

    def __init__(self, layout, theta: TypedIntentTheta = None, using_log_wrapper: bool = False,
                 start_cooking_interaction: bool = False, agent_id: int = 1,
                 config: Optional[TypedIntentConfig] = None):
        self.theta = theta if theta is not None else TypedIntentTheta.default()
        self.layout = layout
        self.using_log_wrapper = using_log_wrapper
        self.config = config if config is not None else TypedIntentConfig()
        self.agent_id = agent_id
        self.model = TypedIntentModel(layout, agent_id=agent_id, config=self.config,
                                      start_cooking_interaction=start_cooking_interaction)
        self.action_dim = NUM_ACTIONS
        self._step_jit = jax.jit(self._step)

    def init_hstate(self, batch_size: int = 1, aux_info=None):
        agent_id = self.agent_id if aux_info is None else aux_info.get("agent_id", self.agent_id)
        if int(agent_id) != self.model.agent_id:
            self.model = TypedIntentModel(self.layout, agent_id=int(agent_id), config=self.config)
        return TypedIntentAgentState(agent_id=jnp.array(agent_id, dtype=jnp.int32),
                                     last_intent=jnp.array(-1, dtype=jnp.int32))

    def dist(self, env_state, hstate):
        view = PartnerView.from_env_state(env_state)
        return self.model.dist(self.theta, view, int(hstate.last_intent))

    def _host_sample(self, env_state, last_intent, rng):
        view = PartnerView.from_env_state(env_state)
        a, m, _ = self.model.sample(self.theta, view, int(last_intent), np.asarray(rng))
        return np.int32(a), np.int32(m)

    def _step(self, env_state, hstate, rng, done):
        action, new_mem = jax.pure_callback(
            self._host_sample,
            (jax.ShapeDtypeStruct((), jnp.int32), jax.ShapeDtypeStruct((), jnp.int32)),
            env_state, hstate.last_intent, rng,
            vmap_method="sequential",
        )
        reset = TypedIntentAgentState(agent_id=hstate.agent_id,
                                      last_intent=jnp.array(-1, dtype=jnp.int32))
        new_hstate = TypedIntentAgentState(agent_id=hstate.agent_id,
                                           last_intent=jnp.where(done, reset.last_intent, new_mem))
        return action, new_hstate

    def get_action(self, params, obs, done, avail_actions, hstate, rng, env_state=None,
                   aux_obs=None, test_mode=False):
        if rng is None:
            raise ValueError("typed_intent partner needs the per-step rng from the env loop")
        rng = jnp.asarray(rng)
        if jax.dtypes.issubdtype(rng.dtype, jax.dtypes.prng_key):
            rng = jax.random.key_data(rng)
        done = jnp.asarray(done).reshape(-1)[0]
        return self._step_jit(env_state, hstate, rng, done)
