"""Validation checks for the typed-intent partners.

Run everything (writes JSON under ``--out``):

    python -m partner_modeling.checks --out outputs/partner_modeling/checks

Individual helpers are imported by ``tests/test_typed_intent_partner.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import jax
import numpy as np
from scipy import stats

from partner_modeling.partners.typed_intent import (
    NUM_ACTIONS, PartnerView, TypedIntentConfig, TypedIntentModel, TypedIntentTheta, unwrap_env_state,
)


# ---------------------------------------------------------------------------
# Sampler / likelihood agreement
# ---------------------------------------------------------------------------

def sample_many(model: TypedIntentModel, theta, view: PartnerView, mem: int, n: int, key) -> np.ndarray:
    """n independent draws from ``model.sample`` (the only sampler) with split keys."""
    keys = np.asarray(jax.random.split(key, n))
    counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    for k in keys:
        a, _, _ = model.sample(theta, view, mem, k)
        counts[a] += 1
    return counts


def chi2_against_dist(model, theta, view, mem, n, key):
    probs, _ = model.dist(theta, view, mem)
    counts = sample_many(model, theta, view, mem, n, key)
    exp = probs * n
    chi2 = float(np.sum((counts - exp) ** 2 / exp))
    p = float(stats.chi2.sf(chi2, NUM_ACTIONS - 1))
    return {"chi2": chi2, "p": p, "counts": counts.tolist(), "probs": probs.tolist()}


def collect_states(env, partner, focal, key, n_episodes: int, max_steps: int = 100):
    """(view, mem) pairs visited on-policy, with flags for adjacency / blocking."""
    from partner_modeling.rollouts import run_episode
    out = []
    for e in range(n_episodes):
        key, k = jax.random.split(key)
        log = run_episode(env, partner, focal, k, max_steps=max_steps)
        for t, v in enumerate(log.views):
            out.append((v, int(log.mem_before[t])))
    return out


def state_flags(model: TypedIntentModel, view: PartnerView) -> Dict[str, bool]:
    f = model.features(view)
    me, ot = model.agent_id, 1 - model.agent_id
    d = int(np.abs(view.pos[me] - view.pos[ot]).sum())
    return {"adjacent": d == 1, "near": d <= 2, "decision": bool(f.decision_point),
            "blocking": bool(model.station_access_np[tuple(view.pos[ot])]) and d <= 2}


# ---------------------------------------------------------------------------
# Partial observability under the study settings (agent_view_size=2)
# ---------------------------------------------------------------------------

def view_coverage(views: Sequence[PartnerView], focal_id: int = 0, radius: int = 2) -> Dict[str, float]:
    """How much of what the partner policy reads is inside the focal agent's egocentric window."""
    from partner_modeling.env_compat import StaticObject
    part_vis, pots_vis, all_vis = [], [], []
    for v in views:
        fy, fx = v.pos[focal_id]
        H, W = v.grid.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        win = (np.abs(yy - fy) <= radius) & (np.abs(xx - fx) <= radius)
        py, px = v.pos[1 - focal_id]
        part_vis.append(bool(win[py, px]))
        pots = v.grid[:, :, 0] == StaticObject.POT
        pots_vis.append(bool(np.all(win[pots])))
        all_vis.append(bool(np.all(win)))
    return {"partner_visible": float(np.mean(part_vis)), "all_pots_visible": float(np.mean(pots_vis)),
            "whole_grid_visible": float(np.mean(all_vis))}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/partner_modeling/checks")
    ap.add_argument("--n-samples", type=int, default=10_000)
    ap.add_argument("--states-per-focal", type=int, default=24)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from partner_modeling.rollouts import (RandomFocal, StaticFocal, env_layout,
                                   make_study_env, scripted_strategies)
    from partner_modeling.teammates import HeuristicTeammateSpec, make_teammate

    env = make_study_env()
    lay = env_layout(env)
    focals = {"static": StaticFocal(), "random": RandomFocal(),
              "coop": scripted_strategies(lay)["coop"]}
    thetas = TypedIntentTheta.grid()
    model = TypedIntentModel(lay, agent_id=1, config=TypedIntentConfig())
    report: Dict = {}
    t0 = time.time()

    # 1. state-level chi^2: states visited on-policy vs each focal; oversample adjacent/blocking
    rng = np.random.default_rng(0)
    chi = []
    key = jax.random.PRNGKey(0)
    for fname, focal in focals.items():
        for ti in (1, 6, 8, 11):   # RDhi, PSlo, FDlo, FShi: spans all three factors
            partner = make_teammate(HeuristicTeammateSpec(family="typed_intent", theta=thetas[ti].preset_name()),
                                    env, "agent_1")
            key, k = jax.random.split(key)
            states = collect_states(env, partner, focal, k, n_episodes=2)
            flags = [state_flags(model, v) for v, _ in states]
            near = [i for i, f in enumerate(flags) if f["near"]]
            far = [i for i, f in enumerate(flags) if not f["near"]]
            m = args.states_per_focal // 4
            pick = list(rng.choice(near, min(m, len(near)), replace=False)) + \
                list(rng.choice(far, min(m, len(far)), replace=False))
            for i in pick:
                v, mem = states[i]
                key, k = jax.random.split(key)
                r = chi2_against_dist(model, thetas[ti], v, mem, args.n_samples, k)
                r.update({"focal": fname, "theta": ti, **flags[i]})
                chi.append(r)
    ps = np.array([c["p"] for c in chi])
    report["chi2"] = {
        "n_tests": len(chi), "n_samples_per_state": args.n_samples,
        "min_p": float(ps.min()), "bonferroni_reject_0.01": int(np.sum(ps < 0.01 / len(ps))),
        "frac_p_below_0.05": float(np.mean(ps < 0.05)),
        "ks_uniform_p": float(stats.kstest(ps, "uniform").pvalue),
        "fisher_combined_p": float(stats.combine_pvalues(ps, method="fisher").pvalue),
        "by_stratum": {s: {"n": int(np.sum([c[s] for c in chi])),
                           "frac_p<0.05": float(np.mean([c["p"] < 0.05 for c in chi if c[s]]) if any(c[s] for c in chi) else np.nan)}
                       for s in ("adjacent", "near", "blocking", "decision")},
    }
    print("chi2:", report["chi2"], f"{time.time() - t0:.0f}s", flush=True)

    # 2. episodes are independent: same reset state, different rng -> different partner play
    sr = []
    # typed intent under the same protocol
    key, k = jax.random.split(key)
    from partner_modeling.rollouts import run_episode
    tpart = make_teammate(HeuristicTeammateSpec(family="typed_intent", theta="flex_pot_direct_lo"), env, "agent_1")
    seqs = []
    for e in range(5):
        key, k = jax.random.split(key)
        seqs.append(run_episode(env, tpart, focals["static"], k, fixed_reset_key=jax.random.PRNGKey(12345)).partner_actions)
    L = min(len(s) for s in seqs)
    sr.append({"family": "typed_intent", "preset": "flex_pot_direct_lo", "reseed": "n/a (per-step rng)",
               "fixed_reset": True, "episodes": 5,
               "distinct_action_sequences": len({s[:L].tobytes() for s in seqs})})
    report["seed_repeat"] = sr
    print("seed repeat:", sr, flush=True)

    # 3. partial observability under the study settings (make_study_env)
    cov = {}
    for fname in ("random", "coop"):
        partner = make_teammate(HeuristicTeammateSpec(family="typed_intent", theta="flex_pot_direct_lo"), env, "agent_1")
        views = []
        for e in range(5):
            key, k = jax.random.split(key)
            views += run_episode(env, partner, focals[fname], k).views
        cov[fname] = view_coverage(views)
    report["observability"] = {"agent_view_size": int(env.agent_view_size),
                               "random_agent_positions": bool(env.random_agent_positions),
                               "coverage": cov}
    print("observability:", report["observability"], flush=True)

    with open(out / "checks.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    with open(out / "chi2_tests.json", "w") as f:
        json.dump(chi, f, default=float)
    print(f"wrote {out / 'checks.json'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
