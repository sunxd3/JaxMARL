"""Graded exclusivity.

    .venv/bin/python -m pytest tests/partner_modeling/test_graded_exclusivity.py -q

Family ``typed_intent_g`` = {runner_g, plater_g} x staging x rho (8 types, same order as
``typed_intent_x``).  At every step the type's off-role intents (runner_g: GET_PLATE; plater_g:
FETCH_PILE / FETCH_STAGED, i.e. exactly what runner_x / plater_x exclude) are masked with
probability q = ``TypedIntentConfig.excl_q``.  The mask is folded into ``dist`` as the mixture
q * pi(. | masked) + (1 - q) * pi(. | unmasked), so ``dist`` stays exact and Markov in (s, m).

1. dist is a normalised distribution with the eps floor, for q in {0, 0.5, 0.9, 1}; dist_all = dist.
2. dist is exactly the q-mixture of the soft type and the exclusive type; q = 0 is array-equal to
   the soft runner / plater and q = 1 array-equal to runner_x / plater_x (probs, intent probs, valid).
3. The sampler matches dist (chi^2 per state, KS on the p-values) at q = 0.5 and 0.9.
4. The ideal observer's memory replay and intent filter stay consistent at q = 0.5.
5. Bit-identity in real episodes: stored lock / redesign episodes (``data/gradedx_golden.json``,
   recorded with the original study code, 300 steps, both layouts,
   L = 0 and 300) are reproduced action-for-action by the graded family at q = 1
   (from typed_intent_x) and q = 0 (from typed_intent types 0-7).
6. The relay partition (companion arm) respects its role masks.
7. Existing families are untouched by excl_q.
"""

import json
import zlib
from pathlib import Path

import jax
import numpy as np
import pytest
from scipy import stats

from partner_modeling.partners.typed_intent import (
    INTENT_FETCH_PILE, INTENT_FETCH_STAGED, INTENT_GET_PLATE, NUM_ACTIONS, THETA_FAMILIES,
    TypedIntentConfig, TypedIntentModel, TypedIntentTheta,
)
from partner_modeling.checks import chi2_against_dist
from partner_modeling.observer import IdealObserver
from partner_modeling.rollouts import (RandomFocal, env_layout, locked_strategy_family, make_study_env, partner_events,
                               run_episode, strategy_family)
from partner_modeling.teammates import HeuristicTeammateSpec, make_teammate, make_theta_set, sample_theta

QS = (0.0, 0.5, 0.9, 1.0)
CFG = {"make_way_station_only": True}
GOLD = json.loads((Path(__file__).parent / "data" / "gradedx_golden.json").read_text())


def _key(name, ti, seed):   # as analysis.sweep._run_one
    return jax.random.fold_in(jax.random.PRNGKey(seed), ti * 1000 + zlib.crc32(name.encode()) % 997)


def _partner(env, family, ti, **extra):
    th = THETA_FAMILIES[family]()[ti]
    return make_teammate(HeuristicTeammateSpec(family=family, theta=th.preset_name(), extra={**CFG, **extra}),
                         env, "agent_1")


@pytest.fixture(scope="module")
def env():
    return make_study_env("coord_ring", 120, full_obs=True)


@pytest.fixture(scope="module")
def models(env):
    lay = env_layout(env)
    return {q: TypedIntentModel(lay, agent_id=1, config=TypedIntentConfig(excl_q=q, **CFG)) for q in QS}


@pytest.fixture(scope="module")
def logs_g(env):
    """Real episodes with graded partners at q = 0.5 / 0.9 and several focals (covers states where
    the off-role intent is the only one valid)."""
    lay = env_layout(env)
    fam = strategy_family(lay, ctrl_cfg=CFG)
    lock = locked_strategy_family(lay, 300, ctrl_cfg=CFG)
    out = []
    for ti, q, focal in [(0, 0.5, lock["g_bal_pile_nostage@F"]), (2, 0.9, fam["fetcher"]),
                         (4, 0.5, lock["g_bal_pile_nostage@P"]), (7, 0.9, fam["plater"]),
                         (1, 0.9, RandomFocal()), (5, 0.5, fam["g_bal_stg_stage"])]:
        out.append((ti, q, run_episode(env, _partner(env, "typed_intent_g", ti, excl_q=q), focal,
                                       _key(focal.name, ti, 3), max_steps=120)))
    return out


def _states(logs_g, stride=3):
    for ti, q, log in logs_g:
        for t in range(0, log.T, stride):
            yield ti, q, log.views[t], int(log.mem_before[t])


# --------------------------------------------------------------------------- grid / registry

def test_grid_g_and_registry():
    gg, gx, g12 = TypedIntentTheta.grid_g(), TypedIntentTheta.grid_x(), TypedIntentTheta.grid()
    assert len(gg) == 8 and len({t.key() for t in gg}) == 8
    assert {int(t.role) for t in gg} == {5, 6}
    assert not ({t.key() for t in gg} & ({t.key() for t in gx} | {t.key() for t in g12}))
    # same (staging, rho) order as grid_x and as the first 8 soft types (episode keys use the index)
    for a, b, c in zip(gg, gx, g12[:8]):
        assert a.key()[1:] == b.key()[1:] == c.key()[1:]
        assert int(a.role) - 5 == int(b.role) - 3 == int(c.role)
    for t in gg:
        assert TypedIntentTheta.from_preset(t.preset_name()).key() == t.key()
    specs = make_theta_set("typed_intent_g", "study", 8)
    assert [sample_theta(s).key() for s in specs] == [t.key() for t in gg]
    assert len(make_theta_set("typed_intent", "study", 12)) == 12
    assert len(make_theta_set("typed_intent_x", "study", 8)) == 8


def test_q_required_and_range(env):
    lay = env_layout(env)
    _, state = env.reset(jax.random.PRNGKey(0))
    from partner_modeling.partners.typed_intent import PartnerView
    v = PartnerView.from_env_state(state)
    th = TypedIntentTheta.grid_g()[0]
    with pytest.raises(ValueError):
        TypedIntentModel(lay, 1, TypedIntentConfig()).dist(th, v, -1)
    with pytest.raises(ValueError):
        TypedIntentModel(lay, 1, TypedIntentConfig(excl_q=1.5)).dist(th, v, -1)


# --------------------------------------------------------------------------- 1. exact dist

def test_dist_normalised_and_dist_all(models, logs_g):
    gg = TypedIntentTheta.grid_g()
    n = 0
    for _, _, v, m in _states(logs_g):
        for q, model in models.items():
            A, P, _ = model.dist_all(gg, v, m)
            for i, th in enumerate(gg):
                p, info = model.dist(th, v, m)
                assert abs(p.sum() - 1) < 1e-12 and p.min() >= model.cfg.eps / NUM_ACTIONS - 1e-12
                ip = info["intent_probs"]
                assert abs(ip.sum() - 1) < 1e-12 and np.all(ip[~info["valid"]] == 0)
                np.testing.assert_allclose(A[i], p, rtol=0, atol=1e-14)
                np.testing.assert_allclose(P[i], ip, rtol=0, atol=1e-14)
                n += 1
    assert n > 1000


# --------------------------------------------------------------------------- 2. mixture / endpoints

def test_mixture_and_endpoint_identity(models, logs_g):
    gg, gx, g12 = TypedIntentTheta.grid_g(), TypedIntentTheta.grid_x(), TypedIntentTheta.grid()
    base = models[0.0]   # excl_q only affects graded types (test_excl_q_does_not_touch_other_families)
    n_differ = n_only_offrole = 0
    for _, _, v, m in _states(logs_g, 2):
        for i, th in enumerate(gg):
            ps, ips = base.dist(g12[i], v, m)[0], base.dist(g12[i], v, m)[1]
            px, ipx = base.dist(gx[i], v, m)
            # q = 0: the soft runner / plater exactly; q = 1: runner_x / plater_x exactly
            p0, i0 = models[0.0].dist(th, v, m)
            p1, i1 = models[1.0].dist(th, v, m)
            assert np.array_equal(p0, ps) and np.array_equal(i0["intent_probs"], ips["intent_probs"])
            assert np.array_equal(i0["valid"], ips["valid"])
            assert np.array_equal(p1, px) and np.array_equal(i1["intent_probs"], ipx["intent_probs"])
            assert np.array_equal(i1["valid"], ipx["valid"])
            for q in (0.5, 0.9):
                pq, iq = models[q].dist(th, v, m)
                np.testing.assert_allclose(pq, q * px + (1 - q) * ps, rtol=0, atol=1e-14)
                np.testing.assert_allclose(iq["intent_probs"], q * ipx["intent_probs"] + (1 - q) * ips["intent_probs"],
                                           rtol=0, atol=1e-14)
            if not np.allclose(px, ps):
                n_differ += 1
                off = INTENT_GET_PLATE if i < 4 else INTENT_FETCH_PILE
                n_only_offrole += bool(ips["intent_probs"][off] > 0.5)
    # the manipulation is not vacuous: the components differ, incl. states where the off-role intent dominates
    assert n_differ > 50 and n_only_offrole > 10, (n_differ, n_only_offrole)
    # every graded type masks exactly the matching exclusive role's set at q = 1
    for i, th in enumerate(gg):
        np.testing.assert_array_equal(models[1.0].graded_mask(th), models[1.0].excluded_mask(gx[i]))
    assert models[1.0].graded_mask(gg[0])[INTENT_GET_PLATE]
    assert models[1.0].graded_mask(gg[4])[[INTENT_FETCH_PILE, INTENT_FETCH_STAGED]].all()


def test_excl_q_does_not_touch_other_families(models, logs_g):
    others = TypedIntentTheta.grid() + TypedIntentTheta.grid_x() + [TypedIntentTheta.default()]
    ref = TypedIntentModel(env_layout(make_study_env("coord_ring", 10, full_obs=True)), 1, TypedIntentConfig(**CFG))
    for _, _, v, m in _states(logs_g, 9):
        A0, P0, _ = ref.dist_all(others, v, m)
        for q in (0.5, 1.0):
            A, P, _ = models[q].dist_all(others, v, m)
            assert np.array_equal(A, A0) and np.array_equal(P, P0)
            for th in others[::5]:
                assert np.array_equal(models[q].dist(th, v, m)[0], ref.dist(th, v, m)[0])


# --------------------------------------------------------------------------- 3. sampler vs dist

@pytest.mark.parametrize("q", [0.5, 0.9])
def test_sampler_matches_dist_chi2(models, logs_g, q):
    rng = np.random.default_rng(int(q * 10))
    gg = TypedIntentTheta.grid_g()
    model = models[q]
    pvals = []
    key = jax.random.PRNGKey(int(q * 100))
    for ti, _, log in logs_g:
        f_states = [t for t in range(log.T)
                    if not np.allclose(models[0.0].dist(gg[ti], log.views[t], int(log.mem_before[t]))[0],
                                       models[1.0].dist(gg[ti], log.views[t], int(log.mem_before[t]))[0])]
        pick = list(rng.choice(log.T, 3, replace=False)) + list(rng.choice(f_states, min(3, len(f_states)),
                                                                           replace=False))
        for t in pick:
            key, k = jax.random.split(key)
            pvals.append(chi2_against_dist(model, gg[ti], log.views[t], int(log.mem_before[t]), 2000, k)["p"])
    pvals = np.array(pvals)
    assert len(pvals) >= 30
    assert pvals.min() > 1e-4 / len(pvals), pvals.min()
    assert stats.kstest(pvals, "uniform").pvalue > 1e-3


def test_registry_partner_samples_from_graded_dist(env, models):
    """The registry partner (pure_callback path) uses the configured q: its dist_fn equals the model's."""
    _, state = env.reset(jax.random.PRNGKey(5))
    from partner_modeling.partners.typed_intent import PartnerView
    for q in (0.5, 0.9):
        tm = _partner(env, "typed_intent_g", 2, excl_q=q)
        p, _ = tm.dist_fn(tm.init(jax.random.PRNGKey(0)), state)
        np.testing.assert_array_equal(p, models[q].dist(TypedIntentTheta.grid_g()[2],
                                                        PartnerView.from_env_state(state), -1)[0])


# --------------------------------------------------------------------------- 4. observer

def test_observer_consistent_at_intermediate_q(models, logs_g):
    gg = TypedIntentTheta.grid_g()
    for ti, q, log in logs_g:
        obs = IdealObserver(models[q], gg)
        np.testing.assert_array_equal(obs.replay_memory(log.views, log.partner_actions), log.mem_before)
        res = obs.run(log.views, log.partner_actions)
        np.testing.assert_allclose(res.intent_post.sum(-1), res.post, atol=1e-12)
        assert np.all(np.isfinite(res.log_marginal))


# --------------------------------------------------------------------------- 5. bit-identity on stored episodes

@pytest.mark.parametrize("layout", ["coord_ring", "counter_circuit"])
def test_endpoints_reproduce_stored_episodes(layout):
    eps = [e for e in GOLD["episodes"] if e["layout"] == layout]
    assert len(eps) >= 8 and {e["q"] for e in eps} == {0.0, 1.0} and {e["lock"] for e in eps} == {0, 300}
    env = make_study_env(layout, GOLD["max_steps"], full_obs=True)
    lay = env_layout(env)
    libs = {0: strategy_family(lay, ctrl_cfg=CFG), 300: locked_strategy_family(lay, 300, ctrl_cfg=CFG)}
    for e in eps:
        foc = libs[e["lock"]][e["focal"]]
        log = run_episode(env, _partner(env, "typed_intent_g", e["theta"], excl_q=e["q"]), foc,
                          _key(e["focal"], e["theta"], e["seed"]), max_steps=GOLD["max_steps"])
        assert log.partner_actions.tolist() == e["partner_actions"], (e["source"], e["focal"], e["theta"])
        assert log.focal_actions.tolist() == e["focal_actions"], (e["source"], e["focal"], e["theta"])
        assert log.soups == e["soups"]


# --------------------------------------------------------------------------- 6. relay partition

def test_relay_partition_invariants(env):
    lay = env_layout(env)
    lib = locked_strategy_family(lay, 40, ctrl_cfg=CFG, partition="relay")
    assert len(lib) == 30 and all(k.endswith(("@A", "@B")) for k in lib)
    for name in ("g_bal_pile_nostage@A", "g_bal_pile_nostage@B", "fetcher@A", "plater@B"):
        foc = lib[name]
        for ti in (1, 6):
            log = run_episode(env, _partner(env, "typed_intent_g", ti, excl_q=0.0), foc, _key(name, ti, 2),
                              max_steps=120)
            ts = [t for t, _ in foc.role_log]
            assert ts == list(range(0, 120, 40))
            roles = dict(foc.role_log)
            for t, ev in partner_events(log, 0):
                r = roles[max(b for b in ts if b <= t)]
                if r == "A":   # feeder: pile -> counter only
                    assert ev not in ("o->-@pot", "-->p@plate_pile", "-->p@counter", "-->o@counter", "p->d@pot"), (name, ti, t, ev)
                else:          # finisher: never takes from the pile
                    assert ev != "-->o@pile", (name, ti, t, ev)


def test_fp_partition_unchanged(env):
    """partition='fp' (default) builds the same library as before (names, weights, masks)."""
    lay = env_layout(env)
    a = locked_strategy_family(lay, 300, ctrl_cfg=CFG)
    b = locked_strategy_family(lay, 300, ctrl_cfg=CFG, partition="fp")
    assert list(a) == list(b) and all(k.endswith(("@F", "@P")) for k in a)
    for k in a:
        assert np.array_equal(a[k].w, b[k].w) and a[k].partition == "fp"
