"""Two-agent rollout harness for the typed-intent partners (coord_ring by default).

The env is JaxMARL's ``OvercookedV2`` with the study settings (``make_study_env``): 2-cell view
radius, random agent positions, negative rewards and recipe resampling on delivery.  The partner and the ideal observer read the full env state regardless.

The partner is agent_1 and is driven through ``TeammatePolicy.act`` (``partner_modeling.teammates``);
the focal agent is agent_0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments.overcooked_v2 import OvercookedV2

from partner_modeling.partners.typed_intent import (
    INTENT_ADD_INGREDIENT, INTENT_DELIVER, INTENT_FETCH_PILE, INTENT_FETCH_STAGED,
    INTENT_GET_PLATE, INTENT_IDLE, INTENT_PICKUP_SOUP, INTENT_STAGE, INTENT_YIELD,
    NUM_INTENTS, PartnerView, TypedIntentConfig, TypedIntentModel, uniform_from_key,
    unwrap_env_state,
)
from partner_modeling.env_compat import Actions, DynamicObject, StaticObject

ACTION_CHARS = "RDLUSI"   # right, down, left, up, stay, interact

# Env settings for every study run; only the layout and max_steps vary.
STUDY_SETTINGS = dict(
    agent_view_size=2,                 # partial observability: 2 cells in each direction
    observation_type="default",
    op_ingredient_permutations=None,
    random_reset=False,
    random_agent_positions=True,
    negative_rewards=True,
    sample_recipe_on_delivery=True,
    start_cooking_interaction=False,
)
TEST_TIME_LAYOUTS = {"test_time_simple", "test_time_wide"}


def make_study_env(layout: str = "coord_ring", max_steps: int = 100, full_obs: bool = False):
    """JaxMARL ``OvercookedV2`` with ``STUDY_SETTINGS``.

    ``indicate_successful_delivery`` is on for the test_time layouts only.
    ``full_obs=True`` changes exactly one setting: ``agent_view_size`` becomes ``None`` (the whole
    grid).  Dynamics are unaffected; only the ``obs`` arrays change.  The partner, the scripted
    focals and the ideal observer read the env state, not ``obs``, so their behaviour is
    identical under both settings.
    """
    kw = dict(STUDY_SETTINGS)
    if full_obs:
        kw["agent_view_size"] = None
    return OvercookedV2(layout=layout, max_steps=max_steps,
                        indicate_successful_delivery=layout in TEST_TIME_LAYOUTS, **kw)


VIEW_RADIUS = 2   # STUDY_SETTINGS["agent_view_size"]


def view_hidden(pos: np.ndarray, pot_cells: np.ndarray, radius: int = VIEW_RADIUS):
    """Per step, whether the focal's (agent_0) (2r+1)^2 egocentric window hides the
    partner / any pot / all pots.  ``pos``: (T, 2, 2) (y, x); ``pot_cells``: (P, 2)."""
    d = np.abs(pos[:, 1] - pos[:, 0])
    partner = np.max(d, axis=1) > radius
    dp = np.abs(pot_cells[None, :, :] - pos[:, 0][:, None, :]).max(-1) > radius   # (T, P)
    return partner, dp.any(1), dp.all(1)


def env_layout(env):
    e = env
    while not hasattr(e, "layout"):
        e = getattr(e, "_env", None) or getattr(e, "env")
    return e.layout


# ---------------------------------------------------------------------------
# Focal policies (agent_0)
# ---------------------------------------------------------------------------

class FocalPolicy:
    name = "focal"

    def reset(self, key):
        pass

    def act(self, view: PartnerView, env_state, key) -> int:
        raise NotImplementedError


class StaticFocal(FocalPolicy):
    name = "static"

    def act(self, view, env_state, key):
        return int(Actions.stay)


class RandomFocal(FocalPolicy):
    name = "random"

    def act(self, view, env_state, key):
        return min(int(uniform_from_key(key) * 6), 5)


class IntentFocal(FocalPolicy):
    """Deterministic scripted focal built on the typed-intent feature extractor.

    ``weights`` scores intents (argmax over valid & allowed intents, distance-penalised).
    Idle behaviour parks the agent on a non-station cell so it does not block access.
    """

    def __init__(self, layout, weights: Dict[int, float], name: str, controller: str = "aware",
                 park_when_idle: bool = True, yield_prob: float = 0.15, ctrl_cfg: Optional[Dict] = None,
                 lock_L: int = 0, init_role: Optional[str] = None, partition: str = "fp"):
        # Role-commitment lock.  lock_L = 0 -> no lock (code
        # path identical to the unlocked focal).  lock_L > 0: commit to init_role at t = 0 and
        # re-commit at t = L, 2L, ... to the role of the focal's own highest-scoring valid intent.
        # ``partition`` picks the role groups: "fp" = ROLE_GROUPS (F / P, the lock round), "relay" =
        # RELAY_GROUPS (A / B, the relay split below).
        self.role_masks = PARTITION_MASKS[partition]
        assert lock_L == 0 or init_role in self.role_masks, (lock_L, init_role, partition)
        self.partition = partition
        self.lock_L = int(lock_L)
        self.init_role = init_role
        self.model = TypedIntentModel(layout, agent_id=0,
                                      config=TypedIntentConfig(controller=controller, **(ctrl_cfg or {})))
        self.w = np.full(NUM_INTENTS, -np.inf)
        for k, v in weights.items():
            self.w[k] = v
        self.name = name
        self.park = park_when_idle
        self.yield_prob = yield_prob
        m = self.model
        station = m.pot_mask_np | m.ingredient_pile_mask_np | m.plate_pile_mask_np | m.goal_mask_np
        self.station_access = np.zeros_like(station)
        for (y, x) in map(tuple, np.argwhere(station)):
            for (ay, ax) in m._access_cells(y, x):
                self.station_access[ay, ax] = True
        self.parking = m.walkable_mask_np & ~self.station_access

    def reset(self, key):
        self._prev = None            # (inv, was_interact) at the previous step
        self._from_counter = False   # holding an item taken from a counter
        self._t = 0                  # steps acted this episode (lock epochs)
        self._role = self.init_role  # committed role (lock only)
        self.role_log = []           # (t, role) at every commitment (lock only)

    def _observe(self, view):
        """Update the from-counter flag from the current state (before acting)."""
        inv = int(view.inv[0])
        if inv == 0:
            self._from_counter = False
        elif self._prev is not None and self._prev[0] == 0 and self._prev[1]:
            fy, fx = _fwd(view, 0)
            self._from_counter = int(view.grid[fy, fx, 0]) == StaticObject.WALL
        return inv

    def act(self, view, env_state, key):
        if not hasattr(self, "_prev"):
            self.reset(key)
        inv = self._observe(view)
        a = self._act(view, key)
        self._prev = (inv, a == int(Actions.interact))
        self._t += 1
        return a

    def _act(self, view, key):
        f = self.model.features(view)
        if self.lock_L > 0:
            if self._t == 0:
                self._role = self.init_role
                self.role_log = [(0, self._role)]
            elif self._t % self.lock_L == 0:
                self._recommit(f)
        # same stochastic yield as the partner: stay with prob yield_prob when a move may
        # be contested (two deterministic reactive agents on a 1-wide ring can livelock)
        if f.valid[INTENT_YIELD] and uniform_from_key(key) < self.yield_prob:
            return int(f.ctrl[INTENT_YIELD])
        ok = f.valid & np.isfinite(self.w)
        ok[INTENT_YIELD] = False
        ok[INTENT_IDLE] = False
        if self._from_counter and not f.forced_stage:
            # never voluntarily re-stage an item just taken from a counter (juggling loop
            # in rooms whose own piles are useless, e.g. fun_symmetries, two_rooms)
            ok[INTENT_STAGE] = False
        if self.lock_L > 0:
            keep = self.role_masks[self._role].copy()
            if f.forced_stage:
                keep[INTENT_STAGE] = True      # forced drop is always allowed
            ok &= keep
        if np.any(ok):
            w = self.w.copy()
            if f.forced_stage:
                w[INTENT_STAGE] = max(w[INTENT_STAGE], 1.0)
            s = np.full(NUM_INTENTS, -np.inf)
            s[ok] = w[ok] * f.value[ok] - 0.5 * f.dist[ok]
            return int(f.ctrl[int(np.argmax(s))])
        # forced drop is always allowed so the agent never gets stuck holding junk
        if f.valid[INTENT_STAGE] and f.forced_stage:
            return int(f.ctrl[INTENT_STAGE])
        return self._park(view)

    def _scores(self, f):
        """Unlocked intent scores (exactly what the unlocked focal argmaxes over)."""
        ok = f.valid & np.isfinite(self.w)
        ok[INTENT_YIELD] = False
        ok[INTENT_IDLE] = False
        if self._from_counter and not f.forced_stage:
            ok[INTENT_STAGE] = False
        w = self.w.copy()
        if f.forced_stage:
            w[INTENT_STAGE] = max(w[INTENT_STAGE], 1.0)
        s = np.full(NUM_INTENTS, -np.inf)
        s[ok] = w[ok] * f.value[ok] - 0.5 * f.dist[ok]
        return s

    def _recommit(self, f):
        """Lock boundary: commit to the role of the own highest-scoring valid intent.  Role-neutral
        intents (YIELD, IDLE, a forced drop) do not decide; if no role intent is valid, keep the role."""
        s = self._scores(f)
        if f.forced_stage:
            s[INTENT_STAGE] = -np.inf
        if np.any(np.isfinite(s)):
            i = int(np.argmax(s))
            r0, r1 = self.role_masks       # two roles; every role-bearing intent is in exactly one
            self._role = r0 if self.role_masks[r0][i] else r1
        self.role_log.append((self._t, self._role))

    def _park(self, view):
        m = self.model
        py, px = int(view.pos[0, 0]), int(view.pos[0, 1])
        if not self.park or self.parking[py, px]:
            return int(Actions.stay)
        oy, ox = int(view.pos[1, 0]), int(view.pos[1, 1])
        idx = m.pos_to_idx[py, px]
        best, bd = None, 10_000
        for (y, x) in map(tuple, np.argwhere(self.parking)):
            if (y, x) == (oy, ox):
                continue
            d = m.dist_matrix[idx, y, x]
            if d < bd:
                best, bd = (y, x), d
        if best is None or bd >= 10000:
            # no free non-station cell (cramped_room): use the partner's fallback class 1
            # (station access not next to a pot/goal) instead of blocking a pot/goal cell
            if m.park_class_np[py, px] <= 1:
                return int(Actions.stay)
            for (y, x) in map(tuple, np.argwhere(m.walkable_mask_np & (m.park_class_np == 1))):
                if (y, x) != (oy, ox) and m.dist_matrix[idx, y, x] < bd:
                    best, bd = (y, x), m.dist_matrix[idx, y, x]
            if best is None or bd >= 10000:
                return int(Actions.stay)
        return int(m.first_step[idx, best[0], best[1]])


# Role groups for the role-commitment lock
ROLE_GROUPS = {
    "F": (INTENT_ADD_INGREDIENT, INTENT_FETCH_PILE, INTENT_FETCH_STAGED, INTENT_STAGE),
    "P": (INTENT_GET_PLATE, INTENT_PICKUP_SOUP, INTENT_DELIVER),
}
ROLE_MASK = {r: np.isin(np.arange(NUM_INTENTS), g) for r, g in ROLE_GROUPS.items()}

# Relay partition: a split that gives the
# second role the counter-to-pot work.  A ("feeder") = take an ingredient from a pile and put
# it on a counter; B ("finisher") = take staged ingredients to the pot, plate, pick up, deliver.
# Per soup: A makes 3 pile->counter trips; B makes 3 counter->pot trips + 1 plate trip + delivery.
RELAY_GROUPS = {
    "A": (INTENT_FETCH_PILE, INTENT_STAGE),
    "B": (INTENT_FETCH_STAGED, INTENT_ADD_INGREDIENT, INTENT_GET_PLATE, INTENT_PICKUP_SOUP, INTENT_DELIVER),
}
PARTITION_MASKS = {
    "fp": ROLE_MASK,
    "relay": {r: np.isin(np.arange(NUM_INTENTS), g) for r, g in RELAY_GROUPS.items()},
}
# Under the relay partition STAGE is A's only way to hand an ingredient on, so every relay
# strategy gets STAGE weight >= this floor (the forced-drop floor used in IntentFocal._act).
RELAY_STAGE_FLOOR = 1.0


# Controller options that are shared by the partner and the focal library (the focal is built
# on the same feature extractor / controller).  Only these are forwarded to the focals.
FOCAL_CTRL_KEYS = ("make_way_station_only",)


def strategy_family(layout, controller: str = "aware", ctrl_cfg: Optional[Dict] = None) -> Dict[str, IntentFocal]:
    """Focal strategy library used to approximate BR(theta) for the cross-best-response matrix.

    12 competent generalists = plate preference {fetch-first 1.5, balanced 4, plate-first 9}
    x staged-item preference {pile 3.0, staged 4.5} x own staging {never, stage on hub 7}
    plus the three specialists (fetcher, plater, shuttle).  Defined before looking at
    any cross-BR result; not pruned afterwards.
    """
    I = dict
    fam = {}
    for pname, wp in (("ff", 1.5), ("bal", 4.0), ("pf", 9.0)):
        for sname, ws in (("pile", 3.0), ("stg", 4.5)):
            for gname, wg in (("nostage", None), ("stage", 7.0)):
                w = I({INTENT_DELIVER: 10, INTENT_PICKUP_SOUP: 8, INTENT_GET_PLATE: wp,
                       INTENT_ADD_INGREDIENT: 6, INTENT_FETCH_PILE: 4, INTENT_FETCH_STAGED: ws})
                if wg is not None:
                    w[INTENT_STAGE] = wg
                    # Same -4 FETCH_STAGED offset the partner's counter-stager uses.  Without
                    # it (STAGE 7 vs FETCH_STAGED 3/4.5 at distance 0) the focal picked up its
                    # own staged onion and re-staged it every step (~20 juggles/episode in
                    # coord_ring; g_ff_stg_stage scored 0.04 soups).  Fixed 2026-11-08.
                    w[INTENT_FETCH_STAGED] = ws - 4.0
                name = f"g_{pname}_{sname}_{gname}"
                fam[name] = IntentFocal(layout, w, name, controller, ctrl_cfg=ctrl_cfg)
    base = scripted_strategies(layout, controller, ctrl_cfg)
    for k in ("fetcher", "plater", "shuttle"):
        fam[k] = base[k]
    return fam


def locked_strategy_family(layout, lock_L: int, controller: str = "aware",
                           ctrl_cfg: Optional[Dict] = None, partition: str = "fp") -> Dict[str, IntentFocal]:
    """Role-lock library: the 15 strategies of ``strategy_family`` x initial role {F, P} = 30,
    named ``<strategy>@F`` / ``<strategy>@P``.  Same weights; only the lock differs.
    ``partition="relay"``: roles {A, B} (``RELAY_GROUPS``), names ``<strategy>@A`` / ``@B``, and
    STAGE weight raised to at least ``RELAY_STAGE_FLOOR``."""
    out = {}
    roles = tuple(PARTITION_MASKS[partition])
    for name, foc in strategy_family(layout, controller, ctrl_cfg).items():
        w = {k: float(v) for k, v in enumerate(foc.w) if np.isfinite(v)}
        if partition == "relay":
            w[INTENT_STAGE] = max(w.get(INTENT_STAGE, -np.inf), RELAY_STAGE_FLOOR)
        for r in roles:
            out[f"{name}@{r}"] = IntentFocal(layout, w, f"{name}@{r}", controller, ctrl_cfg=ctrl_cfg,
                                             lock_L=lock_L, init_role=r, partition=partition)
    return out


def scripted_strategies(layout, controller: str = "aware", ctrl_cfg: Optional[Dict] = None) -> Dict[str, IntentFocal]:
    """Fixed focal strategies used for the decision-relevance check."""
    I = dict
    out = {
        "coop": IntentFocal(layout, I({INTENT_DELIVER: 10, INTENT_PICKUP_SOUP: 8, INTENT_GET_PLATE: 4,
                                       INTENT_ADD_INGREDIENT: 6, INTENT_FETCH_PILE: 4,
                                       INTENT_FETCH_STAGED: 4.5}), "coop", controller, ctrl_cfg=ctrl_cfg),
        "fetcher": IntentFocal(layout, I({INTENT_ADD_INGREDIENT: 6, INTENT_FETCH_PILE: 4,
                                          INTENT_FETCH_STAGED: 4}), "fetcher", controller, ctrl_cfg=ctrl_cfg),
        "plater": IntentFocal(layout, I({INTENT_DELIVER: 10, INTENT_PICKUP_SOUP: 8,
                                         INTENT_GET_PLATE: 4}), "plater", controller, ctrl_cfg=ctrl_cfg),
        "shuttle": IntentFocal(layout, I({INTENT_ADD_INGREDIENT: 6, INTENT_FETCH_STAGED: 8,
                                          INTENT_FETCH_PILE: 1}), "shuttle", controller, ctrl_cfg=ctrl_cfg),
        # competent generalists that differ only in a soft plate-vs-fetch preference
        "coop_platefirst": IntentFocal(layout, I({INTENT_DELIVER: 10, INTENT_PICKUP_SOUP: 8, INTENT_GET_PLATE: 9,
                                                  INTENT_ADD_INGREDIENT: 6, INTENT_FETCH_PILE: 4,
                                                  INTENT_FETCH_STAGED: 4.5}), "coop_platefirst", controller, ctrl_cfg=ctrl_cfg),
        "coop_fetchfirst": IntentFocal(layout, I({INTENT_DELIVER: 10, INTENT_PICKUP_SOUP: 8, INTENT_GET_PLATE: 1.5,
                                                  INTENT_ADD_INGREDIENT: 6, INTENT_FETCH_PILE: 4,
                                                  INTENT_FETCH_STAGED: 4.5}), "coop_fetchfirst", controller, ctrl_cfg=ctrl_cfg),
    }
    return out


# ---------------------------------------------------------------------------
# Episode logging
# ---------------------------------------------------------------------------

@dataclass
class EpisodeLog:
    views: List[PartnerView]            # state before step t
    partner_actions: np.ndarray         # (T,)
    focal_actions: np.ndarray           # (T,)
    mem_before: np.ndarray              # (T,) partner carry last_intent before acting (typed_intent)
    mem_after: np.ndarray               # (T,)
    rewards: np.ndarray                 # (T,)
    rng_keys: Optional[np.ndarray] = None   # (T, 2) partner carry rng_key (original families)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self):
        return len(self.partner_actions)

    @property
    def soups(self) -> int:
        return int(np.sum(self.rewards > 0))

    @property
    def wrong_soups(self) -> int:
        return int(np.sum(self.rewards < 0))


def run_episode(env, partner, focal: FocalPolicy, key, max_steps: int = 100,
                fixed_reset_key=None, record_rng_key: bool = False) -> EpisodeLog:
    """One episode, partner = agent_1 via ``TeammatePolicy.act``."""
    key, reset_key, init_key, focal_key = jax.random.split(key, 4)
    if fixed_reset_key is not None:
        reset_key = fixed_reset_key
    obs, state = env.reset(reset_key)
    carry = partner.init(init_key)
    focal.reset(focal_key)
    done = jnp.array(False)
    views, pa, fa, mb, ma, rw, keys = [], [], [], [], [], [], []
    for t in range(max_steps):
        key, kf, kp, ks = jax.random.split(key, 4)
        view = PartnerView.from_env_state(state)
        a_f = focal.act(view, state, np.asarray(kf))
        m0 = int(getattr(carry, "last_intent", -1)) if hasattr(carry, "last_intent") else -1
        if record_rng_key and hasattr(carry, "rng_key"):
            keys.append(np.asarray(carry.rng_key).copy())
        carry, a_p = partner.act(carry, obs["agent_1"], done, kp, env_state=state)
        a_p = int(np.asarray(a_p).squeeze())
        m1 = int(getattr(carry, "last_intent", -1)) if hasattr(carry, "last_intent") else -1
        obs, state, rew, dones, info = env.step(ks, state, {"agent_0": a_f, "agent_1": a_p})
        views.append(view)
        pa.append(a_p)
        fa.append(a_f)
        mb.append(m0)
        ma.append(m1)
        rw.append(float(rew["agent_0"]))
        if bool(dones["__all__"]):
            break
    return EpisodeLog(views=views, partner_actions=np.array(pa), focal_actions=np.array(fa),
                      mem_before=np.array(mb), mem_after=np.array(ma), rewards=np.array(rw),
                      rng_keys=np.array(keys) if keys else None,
                      meta={"partner": partner.name, "focal": focal.name})


# ---------------------------------------------------------------------------
# Events and rendering
# ---------------------------------------------------------------------------

def _inv_kind(inv: int) -> str:
    if inv == 0:
        return "-"
    if inv & DynamicObject.COOKED and inv & DynamicObject.PLATE:
        return "d"
    if inv == DynamicObject.PLATE:
        return "p"
    return "o"


def _fwd(view: PartnerView, agent: int):
    y, x = int(view.pos[agent, 0]), int(view.pos[agent, 1])
    d = int(view.dir[agent])
    dy, dx = [(-1, 0), (1, 0), (0, 1), (0, -1)][d]  # UP, DOWN, RIGHT, LEFT
    return y + dy, x + dx


def partner_events(log: EpisodeLog, agent: int = 1) -> List[tuple]:
    """(t, event) for inventory transitions of ``agent`` between consecutive views."""
    ev = []
    for t in range(len(log.views) - 1):
        v0, v1 = log.views[t], log.views[t + 1]
        i0, i1 = _inv_kind(int(v0.inv[agent])), _inv_kind(int(v1.inv[agent]))
        if i0 == i1:
            continue
        fy, fx = _fwd(v1, agent)
        st = int(v0.grid[fy, fx, 0]) if 0 <= fy < v0.grid.shape[0] and 0 <= fx < v0.grid.shape[1] else -1
        where = {StaticObject.POT: "pot", StaticObject.WALL: "counter", StaticObject.GOAL: "goal",
                 StaticObject.PLATE_PILE: "plate_pile"}.get(st, "pile" if st >= StaticObject.INGREDIENT_PILE_BASE else "?")
        ev.append((t, f"{i0}->{i1}@{where}"))
    return ev


def render_grid(view: PartnerView) -> str:
    H, W = view.grid.shape[:2]
    rows = []
    for y in range(H):
        row = ""
        for x in range(W):
            st, dyn, ext = (int(v) for v in view.grid[y, x])
            ag = [i for i in range(view.pos.shape[0]) if tuple(view.pos[i]) == (y, x)]
            if ag:
                row += f"{ag[0]}{_inv_kind(int(view.inv[ag[0]]))} "
                continue
            if st == StaticObject.POT:
                n = sum(((dyn >> 2) >> (2 * i)) & 3 for i in range(4))
                c = "c" if dyn & DynamicObject.COOKED else (f"{ext:02d}"[-1] if ext > 0 else str(n))
                row += f"P{c} "
            elif st == StaticObject.WALL:
                row += ("W" + _inv_kind(dyn) + " ") if dyn else "## "
            elif st == StaticObject.GOAL:
                row += "XX "
            elif st == StaticObject.PLATE_PILE:
                row += "BB "
            elif st >= StaticObject.INGREDIENT_PILE_BASE:
                row += "OO "
            else:
                row += ".. "
        rows.append(row)
    return "\n".join(rows)


def render_trace(log: EpisodeLog, t0: int = 0, t1: Optional[int] = None,
                 intents: Optional[np.ndarray] = None) -> str:
    """Compact one-line-per-step trace: t | focal pos inv act | partner pos inv act | pots | counters."""
    from partner_modeling.partners.typed_intent import INTENT_NAMES
    t1 = log.T if t1 is None else t1
    out = []
    for t in range(t0, min(t1, log.T)):
        v = log.views[t]
        pots = []
        for (y, x) in map(tuple, np.argwhere(v.grid[:, :, 0] == StaticObject.POT)):
            dyn, ext = int(v.grid[y, x, 1]), int(v.grid[y, x, 2])
            n = sum(((dyn >> 2) >> (2 * i)) & 3 for i in range(4))
            pots.append("C" if dyn & DynamicObject.COOKED else (f"{n}t{ext}" if ext else str(n)))
        ctr = int(np.sum((v.grid[:, :, 0] == StaticObject.WALL) & (v.grid[:, :, 1] != 0)))
        f = f"F{tuple(int(c) for c in v.pos[0])}{_inv_kind(int(v.inv[0]))}:{ACTION_CHARS[log.focal_actions[t]]}"
        p = f"P{tuple(int(c) for c in v.pos[1])}{_inv_kind(int(v.inv[1]))}:{ACTION_CHARS[log.partner_actions[t]]}"
        it = ""
        if intents is not None:
            it = f" [{INTENT_NAMES[int(np.argmax(intents[t]))]} {np.max(intents[t]):.2f}]"
        r = " +20" if log.rewards[t] > 0 else ""
        out.append(f"t={t:3d} {f:18s} {p:18s} pots={','.join(pots):9s} ctr={ctr}{it}{r}")
    return "\n".join(out)


def stuck_stats(log: EpisodeLog, agent: int = 1, window: int = 10) -> Dict[str, Any]:
    """Heuristics for deadlocks / loops of ``agent``.

    * ``bump_steps``: steps where the agent chose a move but its position did not change
      while the move target was a walkable cell (blocked by the other agent).
    * ``max_no_move``: longest run of steps without a position change.
    * ``osc2``: longest run of period-2 position oscillation (A,B,A,B,...).
    * ``max_no_progress``: longest gap (steps) between inventory events of either agent.
    """
    T = len(log.views)
    pos = np.array([tuple(v.pos[agent]) for v in log.views])
    acts = log.partner_actions if agent == 1 else log.focal_actions
    walk = log.views[0].grid[:, :, 0] == StaticObject.EMPTY
    bumps = 0
    for t in range(T - 1):
        a = int(acts[t])
        if a < 4:
            dy, dx = [(0, 1), (1, 0), (0, -1), (-1, 0)][a]
            ny, nx = pos[t][0] + dy, pos[t][1] + dx
            if 0 <= ny < walk.shape[0] and 0 <= nx < walk.shape[1] and walk[ny, nx] and tuple(pos[t + 1]) == tuple(pos[t]):
                bumps += 1
    run = best = 0
    for t in range(1, T):
        run = run + 1 if tuple(pos[t]) == tuple(pos[t - 1]) else 0
        best = max(best, run)
    osc = obest = 0
    for t in range(2, T):
        if tuple(pos[t]) == tuple(pos[t - 2]) and tuple(pos[t]) != tuple(pos[t - 1]):
            osc += 1
            obest = max(obest, osc)
        else:
            osc = 0
    ev_t = sorted({t for t, _ in partner_events(log, 0)} | {t for t, _ in partner_events(log, 1)})
    gaps = np.diff([0] + ev_t + [T]) if T else np.array([0])
    return {"bump_steps": bumps, "max_no_move": best, "osc2": obest,
            "max_no_progress": int(np.max(gaps)) if len(gaps) else T}
