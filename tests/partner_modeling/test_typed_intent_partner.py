"""Unit tests for the typed-intent partner and the ideal observer.

Run:  .venv/bin/python -m pytest tests/partner_modeling -q
"""

import itertools

import jax
import numpy as np
import pytest
from scipy import stats

from partner_modeling.partners.typed_intent import (
    NUM_ACTIONS, NUM_INTENTS, PartnerView, TypedIntentConfig, TypedIntentModel, TypedIntentTheta,
)
from partner_modeling.checks import chi2_against_dist, state_flags
from partner_modeling.observer import IdealObserver, switching_kernel
from partner_modeling.rollouts import RandomFocal, StaticFocal, env_layout, make_study_env, run_episode, scripted_strategies
from partner_modeling.teammates import HeuristicTeammateSpec, make_teammate, make_theta_set, sample_theta


@pytest.fixture(scope="module")
def env():
    return make_study_env()


@pytest.fixture(scope="module")
def layout(env):
    return env_layout(env)


@pytest.fixture(scope="module")
def model(layout):
    return TypedIntentModel(layout, agent_id=1, config=TypedIntentConfig())


@pytest.fixture(scope="module")
def logs(env, layout):
    """A handful of real two-agent episodes (registry partner, several focals)."""
    out = []
    focals = [StaticFocal(), RandomFocal(), scripted_strategies(layout)["coop"]]
    key = jax.random.PRNGKey(7)
    for ti, focal in [(0, focals[0]), (7, focals[1]), (9, focals[2]), (4, focals[2])]:
        th = TypedIntentTheta.grid()[ti]
        p = make_teammate(HeuristicTeammateSpec(family="typed_intent", theta=th.preset_name()), env, "agent_1")
        key, k = jax.random.split(key)
        out.append((ti, run_episode(env, p, focal, k, max_steps=60)))
    return out


# --------------------------------------------------------------------------- theta grid

def test_theta_grid_and_registry_presets():
    grid = TypedIntentTheta.grid()
    assert len(grid) == 12 and len({t.key() for t in grid}) == 12
    for t in grid:
        assert TypedIntentTheta.from_preset(t.preset_name()).key() == t.key()
    specs = make_theta_set("typed_intent", "study", 12)
    assert [sample_theta(s).key() for s in specs] == [t.key() for t in grid]
    with pytest.raises(ValueError):
        make_theta_set("typed_intent", "train", 3)


# --------------------------------------------------------------------------- dist

def test_dist_is_normalised_with_floor(model, logs):
    eps = model.cfg.eps
    for ti, log in logs:
        for v, m in zip(log.views, log.mem_before):
            for th in TypedIntentTheta.grid():
                p, info = model.dist(th, v, int(m))
                assert p.shape == (NUM_ACTIONS,)
                assert abs(p.sum() - 1) < 1e-12
                assert p.min() >= eps / NUM_ACTIONS - 1e-12
                ip = info["intent_probs"]
                assert abs(ip.sum() - 1) < 1e-12 and np.all(ip[~info["valid"]] == 0)


def test_dist_all_matches_dist(model, logs):
    thetas = TypedIntentTheta.grid()
    _, log = logs[2]
    for v, m in list(zip(log.views, log.mem_before))[::5]:
        A, P, _ = model.dist_all(thetas, v, int(m))
        for i, th in enumerate(thetas):
            np.testing.assert_allclose(A[i], model.dist(th, v, int(m))[0], rtol=0, atol=1e-14)


def test_registry_dist_fn_matches_model(env, model):
    th = TypedIntentTheta.grid()[5]
    tm = make_teammate(HeuristicTeammateSpec(family="typed_intent", theta=th.preset_name()), env, "agent_1")
    _, state = env.reset(jax.random.PRNGKey(3))
    carry = tm.init(jax.random.PRNGKey(0))
    p, _ = tm.dist_fn(carry, state)
    np.testing.assert_allclose(p, model.dist(th, PartnerView.from_env_state(state), -1)[0])


# --------------------------------------------------------------------------- 7.1 sampler vs likelihood

def test_sampler_matches_dist_chi2(model, logs):
    """chi^2 per state (2000 draws each), incl. states next to / blocked by the focal agent."""
    rng = np.random.default_rng(0)
    thetas = TypedIntentTheta.grid()
    pvals, near_seen = [], 0
    key = jax.random.PRNGKey(11)
    for ti, log in logs:
        idx = rng.choice(len(log.views), 6, replace=False)
        near = [t for t in range(len(log.views)) if state_flags(model, log.views[t])["near"]]
        idx = list(idx) + list(rng.choice(near, min(3, len(near)), replace=False)) if near else list(idx)
        for t in idx:
            key, k = jax.random.split(key)
            r = chi2_against_dist(model, thetas[ti], log.views[t], int(log.mem_before[t]), 2000, k)
            pvals.append(r["p"])
            near_seen += state_flags(model, log.views[t])["near"]
    pvals = np.array(pvals)
    assert near_seen > 0
    assert pvals.min() > 1e-4 / len(pvals), pvals.min()
    assert stats.kstest(pvals, "uniform").pvalue > 1e-3


# --------------------------------------------------------------------------- 7.2 replay invariance

def test_replay_reproduces_partner_memory(model, logs):
    obs = IdealObserver(model, TypedIntentTheta.grid())
    for ti, log in logs:
        replay = obs.replay_memory(log.views, log.partner_actions)
        np.testing.assert_array_equal(replay, log.mem_before)
        # memory after the last action as well
        m_last = model.update_memory(log.views[-1], int(replay[-1]), int(log.partner_actions[-1]))
        assert m_last == log.mem_after[-1]


def test_memory_update_is_theta_and_key_independent(model, logs):
    _, log = logs[1]
    for t, v in enumerate(log.views):
        m = int(log.mem_before[t])
        outs = set()
        for th in TypedIntentTheta.grid():
            for s in range(3):
                a, m2, _ = model.sample(th, v, m, np.asarray(jax.random.PRNGKey(s)))
                outs.add((a, m2) if a == log.partner_actions[t] else None)
        outs.discard(None)
        assert len({m2 for _, m2 in outs}) <= 1   # same action -> same memory, for every theta/key


# --------------------------------------------------------------------------- 7.6 episode independence

def test_typed_intent_episodes_differ_with_same_start(env):
    tm = make_teammate(HeuristicTeammateSpec(family="typed_intent", theta="flex_pot_direct_lo"), env, "agent_1")
    seqs = []
    for s in range(4):
        log = run_episode(env, tm, StaticFocal(), jax.random.PRNGKey(100 + s),
                          fixed_reset_key=jax.random.PRNGKey(5), max_steps=40)
        seqs.append(log.partner_actions.tobytes())
    assert len(set(seqs)) == 4


# --------------------------------------------------------------------------- filter exactness

def test_filter_matches_brute_force(model, logs):
    thetas = [TypedIntentTheta.grid()[i] for i in (0, 5, 9)]
    _, log = logs[3]
    views, acts = log.views[10:14], log.partner_actions[10:14]
    mem0 = int(log.mem_before[10])
    for hazard in (0.0, 0.2):
        K = switching_kernel(3, hazard)
        obs = IdealObserver(model, thetas, kernel=K if hazard > 0 else None)
        res = obs.run(views, acts, mem0=mem0)
        # brute force over all theta paths, same starting memory
        mem = [mem0]
        for v, a in zip(views, acts):
            mem.append(model.update_memory(v, mem[-1], int(a)))
        lik = np.array([[model.dist(th, views[t], mem[t])[0][acts[t]] for th in thetas] for t in range(4)])
        post = np.zeros(3)
        for path in itertools.product(range(3), repeat=4):
            p = 1.0 / 3
            for t in range(4):
                if t > 0:
                    p *= K[path[t - 1], path[t]]
                p *= lik[t, path[t]]
            post[path[-1]] += p
        np.testing.assert_allclose(res.post[-1], post / post.sum(), rtol=1e-10)


def test_joint_intent_posterior_marginalises_to_theta_posterior(model, logs):
    obs = IdealObserver(model, TypedIntentTheta.grid())
    _, log = logs[2]
    res = obs.run(log.views, log.partner_actions)
    np.testing.assert_allclose(res.intent_post.sum(-1), res.post, atol=1e-12)
    assert np.all(res.info >= -1e-12)


# --------------------------------------------------------------------------- forward compatibility

def test_wall_added_mid_episode_reroutes_and_changes_dist(env, layout, model):
    """A wall on the shortest route must change the controller and dist (path tables keyed on the
    current static grid, not on the layout captured at init)."""
    from partner_modeling.partners.typed_intent import INTENT_FETCH_PILE
    from partner_modeling.env_compat import StaticObject
    _, state = env.reset(jax.random.PRNGKey(0))
    v = PartnerView.from_env_state(state)
    # partner at (1,1) empty-handed, focal at (3,3); onion pile access is (3,1)
    v.pos = np.array([[3, 3], [1, 1]], dtype=np.int32)
    v.dir = np.array([0, 0], dtype=np.int32)
    v.inv = np.array([0, 0], dtype=np.int32)
    w = PartnerView(grid=v.grid.copy(), pos=v.pos, dir=v.dir, inv=v.inv, recipe=v.recipe, time=v.time)
    w.grid[2, 1, 0] = StaticObject.WALL                                          # wall on (2,1)
    # static-path controller: reroutes the long way round
    sp = TypedIntentModel(layout, agent_id=1, config=TypedIntentConfig(controller="shortest"))
    f0, f1 = sp.features(v), sp.features(w)
    assert f0.ctrl[INTENT_FETCH_PILE] == 1 and f0.dist[INTENT_FETCH_PILE] == 2   # down via (2,1)
    assert f1.ctrl[INTENT_FETCH_PILE] == 0 and f1.dist[INTENT_FETCH_PILE] == 6   # right, 6 steps
    # aware controller: the only remaining route runs through the focal agent -> it must not walk
    # into the wall, and the (blocked) path cost must change
    a0, a1 = model.features(v), model.features(w)
    assert a0.ctrl[INTENT_FETCH_PILE] == 1 and a1.ctrl[INTENT_FETCH_PILE] != 1
    assert a1.dist[INTENT_FETCH_PILE] > a0.dist[INTENT_FETCH_PILE]
    th = TypedIntentTheta.grid()[0]
    assert not np.allclose(model.dist(th, v, -1)[0], model.dist(th, w, -1)[0])
    assert not np.allclose(sp.dist(th, v, -1)[0], sp.dist(th, w, -1)[0])


def test_switch_mask_kernel_only_at_marked_steps(model, logs):
    thetas = [TypedIntentTheta.grid()[i] for i in (0, 9)]
    _, log = logs[3]
    K = switching_kernel(2, 0.3)
    none = IdealObserver(model, thetas).run(log.views, log.partner_actions)
    never = IdealObserver(model, thetas, kernel=K).run(log.views, log.partner_actions,
                                                       switch_mask=np.zeros(log.T, bool))
    np.testing.assert_allclose(none.post, never.post, atol=1e-12)


# --------------------------------------------------------------------------- redesign family (typed_intent_x)

@pytest.fixture(scope="module")
def logs_x(env, layout):
    """Real episodes with the 8-type exclusive-role family (registry partner, several focals)."""
    out = []
    fam = scripted_strategies(layout)
    key = jax.random.PRNGKey(17)
    for ti, focal in [(0, fam["plater"]), (3, fam["coop"]), (4, fam["fetcher"]), (7, RandomFocal()),
                      (2, StaticFocal())]:
        th = TypedIntentTheta.grid_x()[ti]
        p = make_teammate(HeuristicTeammateSpec(family="typed_intent_x", theta=th.preset_name()), env, "agent_1")
        key, k = jax.random.split(key)
        out.append((ti, run_episode(env, p, focal, k, max_steps=80)))
    return out


def test_theta_grid_x_and_registry():
    gx = TypedIntentTheta.grid_x()
    assert len(gx) == 8 and len({t.key() for t in gx}) == 8
    assert {int(t.role) for t in gx} == {3, 4}
    assert not ({t.key() for t in gx} & {t.key() for t in TypedIntentTheta.grid()})
    for t in gx:
        assert TypedIntentTheta.from_preset(t.preset_name()).key() == t.key()
    specs = make_theta_set("typed_intent_x", "study", 8)
    assert [sample_theta(s).key() for s in specs] == [t.key() for t in gx]
    assert len(make_theta_set("typed_intent", "study", 12)) == 12      # original family unchanged
    with pytest.raises(ValueError):   # the original family does not accept the new presets
        make_teammate(HeuristicTeammateSpec(family="typed_intent", theta=gx[0].preset_name()), make_study_env(),
                      "agent_1")


def test_exclusive_roles_never_take_excluded_intents(model, logs_x):
    from partner_modeling.partners.typed_intent import (INTENT_FETCH_PILE, INTENT_FETCH_STAGED,
                                                         INTENT_GET_PLATE, INTENT_IDLE)
    eps = model.cfg.eps
    n_plate_valid = n_fetch_valid = n_idle_fallback = 0
    for _, log in logs_x:
        for v, m in zip(log.views, log.mem_before):
            f = model.features(v)
            for th in TypedIntentTheta.grid_x():
                p, info = model.dist(th, v, int(m))
                ip = info["intent_probs"]
                assert abs(p.sum() - 1) < 1e-12 and p.min() >= eps / NUM_ACTIONS - 1e-12
                assert abs(ip.sum() - 1) < 1e-12 and np.all(ip[~info["valid"]] == 0)
                if int(th.role) == 3:
                    assert ip[INTENT_GET_PLATE] == 0
                    n_plate_valid += bool(f.valid[INTENT_GET_PLATE])
                else:
                    assert ip[INTENT_FETCH_PILE] == 0 and ip[INTENT_FETCH_STAGED] == 0
                    n_fetch_valid += bool(f.valid[INTENT_FETCH_PILE])
                n_idle_fallback += bool(ip[INTENT_IDLE] > 0 and not f.valid[INTENT_IDLE])
    assert n_plate_valid > 0 and n_fetch_valid > 0 and n_idle_fallback > 0


def test_grid_x_dist_all_and_replay(model, logs_x):
    gx = TypedIntentTheta.grid_x()
    obs = IdealObserver(model, gx)
    for ti, log in logs_x:
        for v, m in list(zip(log.views, log.mem_before))[::7]:
            A, P, _ = model.dist_all(gx, v, int(m))
            for i, th in enumerate(gx):
                np.testing.assert_allclose(A[i], model.dist(th, v, int(m))[0], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(obs.replay_memory(log.views, log.partner_actions), log.mem_before)
        res = obs.run(log.views, log.partner_actions)
        np.testing.assert_allclose(res.intent_post.sum(-1), res.post, atol=1e-12)


def test_grid_x_sampler_matches_dist_chi2(model, logs_x):
    rng = np.random.default_rng(1)
    gx = TypedIntentTheta.grid_x()
    pvals = []
    key = jax.random.PRNGKey(23)
    for ti, log in logs_x:
        for t in rng.choice(len(log.views), 6, replace=False):
            key, k = jax.random.split(key)
            pvals.append(chi2_against_dist(model, gx[ti], log.views[t], int(log.mem_before[t]), 2000, k)["p"])
    pvals = np.array(pvals)
    assert pvals.min() > 1e-4 / len(pvals), pvals.min()
    assert stats.kstest(pvals, "uniform").pvalue > 1e-3
