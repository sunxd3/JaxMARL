"""A focal agent that reasons and plans with probabilistic programs (``ppl.py``, ``pp_models.py``).

Every step it

1. **updates its belief** about what is hidden in the partner (the recipe the partner knows,
   its type, its memory) by conditioning the partner program on what the partner was just seen
   to do (``pp_models.belief_step``, enumerated exactly);
2. **plans**, at decision points, by imagining the future: for each of its own options it runs
   the world forward under hypotheses drawn from the belief, with the same partner program now
   simulating the partner, and keeps the option with the highest expected return.

Its options are short policies, not single moves: wait, handle plates only (needs no recipe),
and "cook / serve / fetch for recipe r" for each recipe r it finds plausible.  Cooking and
serving for r set a soup that is not r aside on a counter instead of delivering it.  A policy
that assumes recipe r is imagined in worlds where the recipe is something else too, so acting
on a guess is priced by what it would cost if the guess were wrong.

The agent never reads the true recipe when ``hide_recipe`` is set: its own policies and the
partner program always run on a kitchen whose recipe has been replaced by a hypothesis.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from partner_modeling.env_compat import Actions, DynamicObject, StaticObject
from partner_modeling.llm_focal import ingredients_of, recipe_code
from partner_modeling.partners.typed_intent import (
    INTENT_ADD_INGREDIENT, INTENT_DELIVER, INTENT_FETCH_PILE, INTENT_FETCH_STAGED, INTENT_GET_PLATE,
    INTENT_IDLE, INTENT_PICKUP_SOUP, INTENT_STAGE, INTENT_YIELD, NUM_ACTIONS, NUM_INTENTS, THETA_FAMILIES,
    PartnerView, TypedIntentConfig, TypedIntentModel, _theta_tuple, unwrap_env_state,
)
from partner_modeling.pp_models import (
    Delivery, Hypothesis, PartnerModel, encode_recipe, know_recipe, partner_action, prior, update, with_recipe,
)
from partner_modeling.ppl import Dist, factor, flip, infer, sample, simulate
from partner_modeling.rollouts import FOCAL_CTRL_KEYS, FocalPolicy, _fwd, env_layout

STAY = int(Actions.stay)
_INF = float("inf")


def _weights(d: Dict[int, float]) -> np.ndarray:
    w = np.full(NUM_INTENTS, -_INF)
    for k, v in d.items():
        w[k] = v
    return w


# The scripted focal strategies' intent weights (rollouts.scripted_strategies).
COOP = _weights({INTENT_DELIVER: 10, INTENT_PICKUP_SOUP: 8, INTENT_GET_PLATE: 4, INTENT_ADD_INGREDIENT: 6,
                 INTENT_FETCH_PILE: 4, INTENT_FETCH_STAGED: 4.5})
FETCHER = _weights({INTENT_ADD_INGREDIENT: 6, INTENT_FETCH_PILE: 4, INTENT_FETCH_STAGED: 4})
PLATER = _weights({INTENT_DELIVER: 10, INTENT_PICKUP_SOUP: 8, INTENT_GET_PLATE: 4})


class Option(NamedTuple):
    """Something the agent can commit to for a while: intent weights (None = wait), the recipe
    the policy assumes (None = the policy does not depend on the recipe), and whether it checks
    soups against that recipe: a checking policy sets a soup that does not match down on a
    counter instead of delivering it, and leaves dishes that are lying on counters alone."""
    name: str
    weights: Optional[Tuple[float, ...]]
    recipe: Optional[int]
    checks: bool = False


def my_action(me: TypedIntentModel, view: PartnerView, option: Option, any_recipe: int, yield_prob: float) -> int:
    """The agent's own next move under ``option``, as a program: the same intent scoring the
    scripted focals use, on a kitchen whose recipe is the one the option assumes."""
    f = me.features(with_recipe(view, option.recipe if option.recipe is not None else any_recipe))
    if f.valid[INTENT_YIELD] and flip(yield_prob):
        return int(f.ctrl[INTENT_YIELD])
    if option.weights is None:
        return int(f.ctrl[INTENT_IDLE])
    w = np.array(option.weights)
    ok = f.valid & np.isfinite(w)
    ok[INTENT_YIELD] = ok[INTENT_IDLE] = False
    if option.checks:
        held = int(view.inv[me.agent_id])
        if not (held & DynamicObject.PLATE and held & DynamicObject.COOKED):
            ok[INTENT_DELIVER] = False                    # do not collect dishes that were set aside
        elif held & ~int(DynamicObject.PLATE | DynamicObject.COOKED) != option.recipe:
            free = me.accessible_counter_mask_np & (view.grid[:, :, 1] == 0)     # a soup that is not the recipe
            other = np.array([[int(view.pos[1 - me.agent_id, 0]), int(view.pos[1 - me.agent_id, 1])]], dtype=np.int32)
            a, _ = me._ctrl_to_target(free, int(view.pos[me.agent_id, 0]), int(view.pos[me.agent_id, 1]),
                                      int(view.dir[me.agent_id]), other)
            return int(f.ctrl[INTENT_IDLE]) if a is None else int(a)
    if ok.any():
        if f.forced_stage:
            w[INTENT_STAGE] = max(w[INTENT_STAGE], 1.0)
        score = np.full(NUM_INTENTS, -_INF)
        score[ok] = w[ok] * f.value[ok] - 0.5 * f.dist[ok]
        return int(f.ctrl[int(np.argmax(score))])
    if f.valid[INTENT_STAGE] and f.forced_stage:      # holding something useless: put it down
        return int(f.ctrl[INTENT_STAGE])
    return int(f.ctrl[INTENT_IDLE])


def in_progress(view: PartnerView) -> float:
    """What the work under way in a kitchen is worth, on the scale of the 20 a delivered soup
    scores: ingredients that fit the recipe in force, soups cooking, soups plated.  Ingredients
    and soups that do not fit count against it.  Used to value where an imagined future ends."""
    recipe = Counter(ingredients_of(view.recipe))

    def fits(contents) -> bool:
        return not (Counter(contents) - recipe)

    total = 0.0
    for (y, x) in map(tuple, np.argwhere(view.grid[:, :, 0] == StaticObject.POT)):
        dyn = int(view.grid[y, x, 1])
        contents = ingredients_of(dyn)
        if not contents:
            continue
        started = bool(dyn & DynamicObject.COOKED) or view.grid[y, x, 2] > 0
        total += (4.0 * len(contents) + (4.0 if started else 0.0)) if fits(contents) else -4.0 * len(contents)
    held = [(int(v), -10.0) for v in view.inv]                # a wrong soup in hand is likely to be delivered
    aside = [(int(v), -2.0) for v in view.grid[:, :, 1][view.grid[:, :, 0] == StaticObject.WALL]]
    for v, wrong in held + aside:
        if v & DynamicObject.PLATE and v & DynamicObject.COOKED:
            total += 18.0 if Counter(ingredients_of(v)) == recipe else wrong
    return total


def rollout_return(agent: "PPFocal", state, option: Option, h: Hypothesis, horizon: int, to_the_end: bool) -> float:
    """Imagine ``horizon`` steps in a world where hypothesis ``h`` is true and return the
    discounted reward, plus the worth of what is in progress if the episode goes on after that.
    The partner is the partner program; the kitchen is the real simulator."""
    s = state.replace(recipe=jnp.array(h.recipe, dtype=state.recipe.dtype))
    total, mem = 0.0, h.mem
    for k in range(horizon):
        view = PartnerView.from_env_state(s)
        a_me = my_action(agent.me, view, option, agent.any_recipe, agent.yield_prob)
        hk = h._replace(recipe=int(view.recipe), mem=mem)        # the partner knows the recipe in force
        a_partner = agent.model.program(agent.model.pm, view, hk)
        mem = agent.model.next_memory(view, hk, a_partner)
        s, reward = agent.kitchen_step(s, a_me, a_partner, sample(agent.key_ids))
        total += agent.gamma ** k * reward
        if reward > 0:
            option = agent.after_delivery                         # a new recipe was drawn; I would not know it
    if not to_the_end:
        total += agent.gamma ** horizon * in_progress(PartnerView.from_env_state(s))
    return total


def choose(options: Sequence[Option], value: Dict[Option, float], alpha: float) -> Option:
    """Planning as inference: options are weighted by exp(alpha * expected return)."""
    o = sample(Dist.uniform(options))
    factor(alpha * value[o])
    return o


class PPFocal(FocalPolicy):
    """Focal agent (agent_0): exact belief over the partner, expected-return planning by rollouts.

    ``program``: the partner program used for both belief and planning (default: the reference
    ``pp_models.partner_action``; pass another to study a mis-specified or machine-written model).
    ``thetas``: partner types the agent considers (default: the whole family).
    ``horizon`` / ``n_rollouts`` / ``replan``: imagination length, rollouts per (option, recipe
    hypothesis), and the longest gap between plans.
    """

    name = "pp"

    def __init__(self, env, family: str = "typed_intent", cfg: Optional[Dict] = None, hide_recipe: bool = True,
                 program: Callable = partner_action, use_memory: bool = True,
                 thetas: Optional[Sequence] = None, horizon: int = 50,
                 n_rollouts: int = 2, replan: int = 4, gamma: float = 0.98, alpha: float = 1.0,
                 margin: float = 1.0, min_recipe_prob: float = 0.1, yield_prob: float = 0.15, seed: int = 0,
                 verbose: bool = False):
        cfg = dict(cfg or {})
        self.env = env
        lay = env_layout(env)
        self.max_steps = int(env.max_steps)
        self.model = PartnerModel(TypedIntentModel(lay, agent_id=1, config=TypedIntentConfig(**cfg)), program,
                                  use_memory=use_memory)
        self.me = TypedIntentModel(lay, agent_id=0, config=TypedIntentConfig(
            controller="aware", **{k: v for k, v in cfg.items() if k in FOCAL_CTRL_KEYS}))
        self.recipes = [encode_recipe(r) for r in np.asarray(lay.possible_recipes).tolist()]
        self.any_recipe = self.recipes[0]
        self.recipe_prior = Dist.uniform(self.recipes)
        self.thetas = [_theta_tuple(t) for t in (thetas if thetas is not None else THETA_FAMILIES[family]())]
        self.hide_recipe = bool(hide_recipe and len(self.recipes) > 1)
        self.horizon, self.n_rollouts, self.replan = int(horizon), int(n_rollouts), int(replan)
        self.gamma, self.alpha, self.margin = gamma, alpha, margin
        self.min_recipe_prob, self.yield_prob, self.seed, self.verbose = min_recipe_prob, yield_prob, seed, verbose
        self.wait = Option("wait", None, None)
        self.plates = Option("plates", tuple(PLATER), None)
        self.after_delivery = self.plates
        self._keys = jax.random.split(jax.random.PRNGKey(seed), 64)
        self.key_ids = Dist.uniform(range(64))
        self.reset(None)

    def reset(self, key):
        self.belief = prior(self.recipes, self.thetas)
        self.rng = np.random.default_rng([self.seed, 1])
        self._t = 0
        self._prev_state = self._prev_view = self._last_a = None
        self._blocked = 0
        self.impossible = 0                         # steps the partner model could not explain at all
        self.deliveries: List[Dict[str, Any]] = []  # every soup handed in, by either cook
        self._option = self.wait
        self._t_plan = -10 ** 6
        self._sig: Any = None
        self.trace: List[Dict[str, Any]] = []       # one entry per step
        self.plans: List[Dict[str, Any]] = []       # one entry per plan

    # ------------------------------------------------------------ the kitchen as a simulator
    def kitchen_step(self, state, a_me: int, a_partner: int, key_id: int = 0):
        _, s, rew, _, _ = self.env.step(self._keys[key_id], state, {"agent_0": a_me, "agent_1": a_partner})
        return s, float(rew["agent_0"])

    @staticmethod
    def _visible(state) -> Tuple[bytes, ...]:
        """What everyone can see of a state: positions, facing, inventories and the grid."""
        a = state.agents
        return tuple(np.asarray(x).tobytes() for x in (a.pos.x, a.pos.y, a.dir, a.inventory, state.grid))

    def _seen_actions(self, prev_state, my_a: int, state) -> List[int]:
        """Partner actions that would have turned ``prev_state`` into what is now visible."""
        now = self._visible(state)
        seen = [a for a in range(NUM_ACTIONS) if self._visible(self.kitchen_step(prev_state, my_a, a)[0]) == now]
        return seen or list(range(NUM_ACTIONS))

    @staticmethod
    def _delivery(prev: PartnerView, view: PartnerView, state) -> Optional[Delivery]:
        for ag in range(prev.inv.shape[0]):
            inv = int(prev.inv[ag])
            if inv & DynamicObject.PLATE and inv & DynamicObject.COOKED and int(view.inv[ag]) == 0:
                fy, fx = _fwd(view, ag)
                if (0 <= fy < prev.grid.shape[0] and 0 <= fx < prev.grid.shape[1]
                        and prev.grid[fy, fx, 0] == StaticObject.GOAL):
                    contents = inv & ~int(DynamicObject.PLATE | DynamicObject.COOKED)
                    return Delivery(contents, bool(np.asarray(state.new_correct_delivery)), ag)
        return None

    # ------------------------------------------------------------ acting
    def act(self, view: PartnerView, env_state, key) -> int:
        state = unwrap_env_state(env_state)
        delivery = None
        if self._prev_state is not None:
            moved = not np.array_equal(self._prev_view.pos[0], view.pos[0])
            self._blocked = self._blocked + 1 if (self._last_a < 4 and not moved) else 0
            delivery = self._delivery(self._prev_view, view, state)
            try:
                self.belief = update(self.model, self.belief, self._prev_view,
                                     self._seen_actions(self._prev_state, self._last_a, state), delivery,
                                     self.recipe_prior)
            except ValueError:      # the model gave what happened probability zero under every hypothesis
                self.impossible += 1
                if delivery is not None and delivery.correct:
                    self.belief = prior(self.recipes, self.thetas)
        if delivery is not None:    # for analysis: who handed in what, under which option and belief
            before = self.trace[-1]
            self.deliveries.append({"t": self._t - 1, "who": delivery.who, "correct": delivery.correct,
                                    "soup": recipe_code(ingredients_of(delivery.contents)),
                                    "option": before["option"],
                                    "p_recipe": {recipe_code(ingredients_of(r)): round(p, 3)
                                                 for r, p in before["recipe"].items()}})
        if not self.hide_recipe:
            self.belief = know_recipe(self.belief, int(view.recipe))
        p_recipe = self.belief.marginal(lambda h: h.recipe)
        sig = (tuple(int(v) for v in view.grid[:, :, 1][view.grid[:, :, 0] == StaticObject.POT]),
               tuple(bool(v) for v in (view.grid[:, :, 2] > 0)[view.grid[:, :, 0] == StaticObject.POT]),
               int(view.inv[0]), p_recipe.argmax())
        if delivery is not None or sig != self._sig or self._t - self._t_plan >= self.replan:
            self._plan(state, p_recipe)
            self._sig = sig
        a = simulate(my_action, self.me, view, self._option, self.any_recipe, self.yield_prob, rng=self.rng)
        if self._blocked >= 2 and self.rng.random() < 0.5:      # two cooks stepping into one cell block forever
            a = STAY
        self.trace.append({"t": self._t, "option": self._option.name,
                           "recipe": {int(r): float(p) for r, p in p_recipe.items()},
                           "type": {k: float(p) for k, p in self.belief.marginal(
                               lambda h: (h.role, h.staging, h.rho)).items()}})
        self._prev_state, self._prev_view, self._last_a = state, view, int(a)
        self._t += 1
        return int(a)

    # ------------------------------------------------------------ planning
    def options(self, p_recipe: Dist) -> List[Option]:
        out = [self.wait, self.plates]
        if len(self.recipes) == 1:
            r = self.recipes[0]
            return out + [Option("cook", tuple(COOP), r), Option("fetch", tuple(FETCHER), r)]
        best = p_recipe.argmax()
        for r, p in p_recipe.items():
            if p >= self.min_recipe_prob or r == best:
                code = recipe_code(ingredients_of(r))
                out += [Option(f"cook[{code}]", tuple(COOP), r, True), Option(f"serve[{code}]", tuple(PLATER), r, True),
                        Option(f"fetch[{code}]", tuple(FETCHER), r)]
        return out

    def _plan(self, state, p_recipe: Dist):
        horizon = min(self.horizon, self.max_steps - self._t)
        options = self.options(p_recipe)
        # Worlds to imagine: for every recipe that is not negligible, a few hypotheses drawn from
        # the belief given that recipe.  All options are imagined in the same worlds with the same
        # random numbers, so their returns differ by what the option does, not by luck.
        worlds: List[Tuple[float, Hypothesis, Tuple[int, ...]]] = []
        for i, (r, p) in enumerate(p_recipe.items()):
            if p < 0.02:
                continue
            given = Dist.from_weights({h: q for h, q in self.belief.items() if h.recipe == r})
            for j in range(self.n_rollouts):
                seed = (self.seed, self._t, i, j)
                worlds.append((p / self.n_rollouts, given.draw(np.random.default_rng(seed + (0,))), seed))
        z = sum(w for w, _, _ in worlds)
        to_the_end = self._t + horizon >= self.max_steps
        value = {o: sum(w * simulate(rollout_return, self, state, o, h, horizon, to_the_end,
                                     rng=np.random.default_rng(seed)) for w, h, seed in worlds) / z
                 for o in options} if horizon > 0 else {o: 0.0 for o in options}
        best = infer(choose, options, value, self.alpha).argmax()
        current = self._option if self._option in value else None
        if current is None or value[best] > value[current] + self.margin:
            self._option = best
        self._t_plan = self._t
        self.plans.append({"t": self._t, "chosen": self._option.name,
                           "value": {o.name: round(v, 2) for o, v in value.items()}, "worlds": len(worlds)})
        if self.verbose:
            print(f"  t={self._t:3d} {self._option.name:12s} "
                  + " ".join(f"{o.name}={v:.1f}" for o, v in value.items()), flush=True)
