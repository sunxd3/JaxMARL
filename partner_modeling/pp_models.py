"""The partner and the belief about it, written as probabilistic programs (``ppl.py``).

* ``partner_action`` is one decision step of the typed-intent partner as a generative program.
  Enumerating it gives the exact action distribution, the same numbers as the hand-derived
  ``TypedIntentModel.dist``; simulating it draws an action.  The same program is therefore used
  for inference (what does this action say about the partner?) and for planning (what will the
  partner do next?).
* ``belief_step`` is one step of social reasoning: pick a hypothesis about the partner, run the
  partner program under it, and keep the hypothesis only if the simulated partner does what the
  real one was seen to do.

A hypothesis is everything hidden about the partner: the recipe it knows, its type (role,
staging, pause rate) and its memory (its last intent).  The deterministic parts of the partner
(what is valid in a state, where its controller would walk) come from ``TypedIntentModel``; all
of the partner's randomness is in the program.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, FrozenSet, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from partner_modeling.env_compat import Actions, DynamicObject
from partner_modeling.partners.typed_intent import (
    INTENT_IDLE, INTENT_STAGE, INTENT_YIELD, NUM_ACTIONS, NUM_INTENTS, PartnerView, TypedIntentModel,
)
from partner_modeling.ppl import Dist, condition, flip, infer, sample

STAY = int(Actions.stay)
ANY_ACTION = Dist.uniform(range(NUM_ACTIONS))


class Hypothesis(NamedTuple):
    recipe: int       # the recipe the partner believes is in force (env encoding)
    role: int
    staging: int
    rho: float        # pause probability
    mem: int          # the partner's memory: its last sticky intent, or -1


def encode_recipe(recipe: Sequence[int]) -> int:
    return int(sum(int(DynamicObject.BASE_INGREDIENT) << (2 * int(i)) for i in recipe))


def ingredients_of(v: int) -> List[int]:
    """Ingredient indices encoded in a dynamic-object value (two bits per ingredient type)."""
    out, v, idx = [], int(v) >> 2, 0
    while v > 0:
        out += [idx] * (v & 3)
        v >>= 2
        idx += 1
    return out


def with_recipe(view: PartnerView, recipe: int) -> PartnerView:
    """The same kitchen as the partner sees it if the recipe is ``recipe``."""
    return view if int(view.recipe) == recipe else dataclasses.replace(view, recipe=recipe)


# ---------------------------------------------------------------------------
# The partner
# ---------------------------------------------------------------------------

def partner_action(pm: TypedIntentModel, view: PartnerView, h: Hypothesis) -> int:
    """One step of the typed-intent partner: noise, then an intent, then the controller's move."""
    cfg = pm.cfg
    kind = sample(Dist(("act", "pause", "slip"), (1.0 - h.rho - cfg.eps, h.rho, cfg.eps)))
    if kind == "pause":
        return STAY
    if kind == "slip":
        return sample(ANY_ACTION)

    f = pm.features(with_recipe(view, h.recipe))
    if f.valid[INTENT_YIELD] and flip(cfg.yield_prob):          # the other cook is close: maybe give way
        return int(f.ctrl[INTENT_YIELD])

    theta = (h.role, h.staging, h.rho)
    available = f.valid.copy()
    available[INTENT_YIELD] = False
    available &= ~pm.excluded_mask(theta)                       # exclusive roles never do the off-role job
    graded = pm.graded_mask(theta)
    if graded.any() and flip(pm.excl_q()):                      # graded roles skip it only some of the time
        available &= ~graded
    if not available.any():
        available[INTENT_IDLE] = True

    weight = pm.weight_vector(theta)
    if f.forced_stage:
        weight[INTENT_STAGE] = cfg.w_forced
    score = weight * f.value - cfg.dist_weight * f.dist + pm.comp_coef(theta) * f.comp
    if 0 <= h.mem < NUM_INTENTS and available[h.mem]:
        score[h.mem] += cfg.inertia                             # it tends to keep doing what it was doing
    options = np.flatnonzero(available)
    logits = cfg.beta * score[options]
    intent = sample(Dist(tuple(int(i) for i in options), np.exp(logits - logits.max())))
    return int(f.ctrl[intent])


class PartnerModel:
    """A partner program plus the deterministic pieces around it, with a cache of its enumerated
    action distributions.  ``program(pm, view, hypothesis) -> action`` may be any program with
    that signature, for example one written by a language model."""

    def __init__(self, pm: TypedIntentModel, program: Callable = partner_action, use_memory: bool = True):
        self.pm, self.program, self.use_memory = pm, program, use_memory
        self._cache: Dict[Tuple[bytes, Hypothesis], Dist] = {}

    def action_dist(self, view: PartnerView, h: Hypothesis) -> Dist:
        """Exact distribution of the partner's next action under hypothesis ``h``."""
        key = (view.key(), h)
        d = self._cache.get(key)
        if d is None:
            if len(self._cache) > 200_000:
                self._cache.clear()
            d = self._cache[key] = infer(self.program, self.pm, view, h)
        return d

    def next_memory(self, view: PartnerView, h: Hypothesis, action: int) -> int:
        """The partner's memory after ``action`` (-1 throughout for programs that keep none)."""
        if not self.use_memory:
            return -1
        return int(self.pm.update_memory(with_recipe(view, h.recipe), h.mem, int(action)))


# ---------------------------------------------------------------------------
# The belief about the partner
# ---------------------------------------------------------------------------

class Delivery(NamedTuple):
    """A soup was handed in during the step: its contents (as a recipe encoding) and the verdict."""
    contents: int
    correct: bool
    who: int = -1       # which cook handed it in (0 is the focal agent), when known


def belief_step(model: PartnerModel, belief: Dist, view: PartnerView, seen: FrozenSet[int],
                delivery: Optional[Delivery], recipes: Dist) -> Hypothesis:
    """One step of reasoning about the partner.

    ``seen`` holds the partner actions that would have produced what was observed (a blocked
    move, a pause and an interaction with no effect can look the same).  A correct delivery
    reveals the recipe that was in force and starts a new one, drawn from ``recipes``; a wrong
    delivery shows that the recipe is not what was delivered.
    """
    h = sample(belief)
    action = sample(model.action_dist(view, h))
    condition(action in seen)
    h = h._replace(mem=model.next_memory(view, h, action))
    if delivery is not None:
        condition((h.recipe == delivery.contents) == delivery.correct)
        if delivery.correct:
            h = h._replace(recipe=sample(recipes))
    return h


def prior(recipes: Sequence[int], thetas: Sequence[Tuple[int, int, float]]) -> Dist:
    """Uniform over recipes and partner types, with empty memory."""
    return Dist.uniform(Hypothesis(r, int(role), int(stag), float(rho), -1)
                        for r in recipes for (role, stag, rho) in thetas)


def update(model: PartnerModel, belief: Dist, view: PartnerView, seen, delivery: Optional[Delivery],
           recipes: Dist) -> Dist:
    """Posterior over hypotheses after one observed step."""
    return infer(belief_step, model, belief, view, frozenset(int(a) for a in seen), delivery, recipes)


def know_recipe(belief: Dist, recipe: int) -> Dist:
    """The belief of an agent that is shown the current recipe."""
    return belief.marginal(lambda h: h._replace(recipe=recipe))
