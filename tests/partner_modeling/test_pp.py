"""The probabilistic-program layer, partner program and belief program.

    .venv/bin/python -m pytest tests/partner_modeling/test_pp.py -q

The partner written as a program must give the same action probabilities as the hand-derived
``TypedIntentModel.dist``, and the belief program must give the same posteriors as the exact
observers (``IdealObserver`` over the type, ``recipe_filter`` over the recipe).
"""

import jax
import numpy as np
import pytest

from partner_modeling.observer import IdealObserver, recipe_filter
from partner_modeling.partners.typed_intent import THETA_FAMILIES, TypedIntentConfig, TypedIntentModel, _theta_tuple
from partner_modeling.pp_models import (
    Delivery, Hypothesis, PartnerModel, encode_recipe, ingredients_of, prior, update,
)
from partner_modeling.ppl import Dist, condition, flip, infer, observe, sample, simulate
from partner_modeling.rollouts import env_layout, make_study_env, run_episode, scripted_strategies
from partner_modeling.teammates import HeuristicTeammateSpec, make_teammate

CFG = {"make_way_station_only": True}


def _episode(layout, family, theta, focal, seed, steps, cfg=CFG):
    env = make_study_env(layout, steps)
    lay = env_layout(env)
    partner = make_teammate(HeuristicTeammateSpec(family=family, theta=theta, extra=cfg), env, "agent_1")
    log = run_episode(env, partner, scripted_strategies(lay, ctrl_cfg=CFG)[focal], jax.random.PRNGKey(seed),
                      max_steps=steps)
    return lay, log


def test_enumeration_conditioning_and_nesting():
    def two_coins():
        a, b = flip(0.3), flip(0.6)
        condition(a or b)
        return a

    assert infer(two_coins).prob(True) == pytest.approx(0.3 / (1 - 0.7 * 0.4), abs=1e-12)

    def which_coin():       # a program that observes the output of another, enumerated, program
        bias = sample(Dist.uniform([0.2, 0.8]))
        observe(infer(lambda: flip(bias)), True)
        return bias

    d = infer(which_coin)
    assert d.prob(0.8) == pytest.approx(0.8, abs=1e-12)
    assert np.exp(d.log_evidence) == pytest.approx(0.5, abs=1e-12)      # P(heads) under the prior


def test_importance_sampling_and_simulation_agree_with_enumeration():
    def die_given_big():
        x = sample(Dist.uniform(range(1, 7)))
        observe(Dist((True, False), (x / 6.0, 1 - x / 6.0)), True)
        return x

    exact = infer(die_given_big)
    approx = infer(die_given_big, method="importance", n=20000, rng=np.random.default_rng(0))
    assert max(abs(exact.prob(v) - approx.prob(v)) for v in range(1, 7)) < 0.02
    draws = [simulate(lambda: sample(Dist((0, 1), (0.25, 0.75))), rng=np.random.default_rng(i)) for i in range(2000)]
    assert abs(np.mean(draws) - 0.75) < 0.03


@pytest.mark.parametrize("family, extra", [("typed_intent", {}), ("typed_intent_x", {}),
                                           ("typed_intent_g", {"excl_q": 0.5})])
def test_partner_program_matches_hand_derived_dist(family, extra):
    cfg = {**CFG, **extra}
    thetas = THETA_FAMILIES[family]()
    lay, log = _episode("coord_ring", family, thetas[2].preset_name(), "coop", 1, 60, cfg)
    pm = TypedIntentModel(lay, agent_id=1, config=TypedIntentConfig(**cfg))
    model = PartnerModel(pm)
    for t in range(0, log.T, 4):
        view, mem = log.views[t], int(log.mem_before[t])
        for th in thetas:
            exact, _ = pm.dist(th, view, mem)
            d = model.action_dist(view, Hypothesis(int(view.recipe), *_theta_tuple(th), mem))
            assert max(abs(d.prob(a) - exact[a]) for a in range(6)) < 1e-12


def test_belief_over_type_matches_ideal_observer():
    thetas = THETA_FAMILIES["typed_intent"]()
    lay, log = _episode("coord_ring", "typed_intent", thetas[9].preset_name(), "fetcher", 3, 60)
    pm = TypedIntentModel(lay, agent_id=1, config=TypedIntentConfig(**CFG))
    model = PartnerModel(pm)
    exact = IdealObserver(pm, thetas).run(log.views, log.partner_actions).post
    recipe = int(log.views[0].recipe)
    belief = prior([recipe], [_theta_tuple(t) for t in thetas])
    for t in range(log.T):
        belief = update(model, belief, log.views[t], [int(log.partner_actions[t])], None, Dist.delta(recipe))
        mine = belief.marginal(lambda h: (h.role, h.staging, h.rho))
        assert max(abs(mine.prob(_theta_tuple(th)) - exact[t, i]) for i, th in enumerate(thetas)) < 1e-10


def test_belief_over_recipe_matches_recipe_filter():
    thetas = THETA_FAMILIES["typed_intent"]()
    theta = next(t for t in thetas if t.preset_name() == "flex_pot_direct_lo")
    lay, log = _episode("cramped_room_v2", "typed_intent", "flex_pot_direct_lo", "coop", 0, 120)
    assert log.soups >= 1                                           # the recipe is redrawn at least once
    pm = TypedIntentModel(lay, agent_id=1, config=TypedIntentConfig(**CFG))
    model = PartnerModel(pm)
    recipes = np.asarray(lay.possible_recipes).tolist()
    enc = [encode_recipe(r) for r in recipes]
    exact, _, _ = recipe_filter(pm, theta, log.views, log.partner_actions, log.rewards, recipes)
    belief = prior(enc, [_theta_tuple(theta)])
    for t in range(log.T):
        mine = belief.marginal(lambda h: h.recipe)
        assert max(abs(mine.prob(e) - exact[t, i]) for i, e in enumerate(enc)) < 1e-10
        delivery = None
        if log.rewards[t] != 0 and t + 1 < log.T:
            for ag in (0, 1):
                inv = int(log.views[t].inv[ag])
                if inv & 1 and inv & 2 and int(log.views[t + 1].inv[ag]) == 0:
                    delivery = Delivery(encode_recipe(ingredients_of(inv)), bool(log.rewards[t] > 0))
        belief = update(model, belief, log.views[t], [int(log.partner_actions[t])], delivery, Dist.uniform(enc))

