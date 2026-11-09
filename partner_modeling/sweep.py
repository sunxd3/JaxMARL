"""Run typed-intent partners against several focal policies in real coord_ring episodes.

For every (focal, theta, seed) the worker runs one 100-step episode through the
teammate registry (partner = agent_1), replays the partner's
memory, runs the ideal observer over the 12-type grid and stores compact results.

Usage:
    python -m partner_modeling.sweep --out outputs/partner_modeling/sweep --episodes 30 \
        --focals static random coop fetcher plater shuttle --controller aware
"""

from __future__ import annotations

import argparse
import json
import zlib
import os
import pickle
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

_G = {}


def _init_worker(cfg_kwargs, focals, layout, max_steps, full_obs=False, compact=False,
                 theta_family="typed_intent", lock_L=0, partition="fp"):
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    import jax  # noqa: F401
    from partner_modeling.partners.typed_intent import THETA_FAMILIES, TypedIntentConfig, TypedIntentModel
    from partner_modeling.observer import IdealObserver
    from partner_modeling.rollouts import (RandomFocal, StaticFocal, env_layout,
                                   make_study_env, scripted_strategies, strategy_family)
    from partner_modeling.teammates import HeuristicTeammateSpec, make_teammate

    env = make_study_env(layout, max_steps, full_obs=full_obs)
    lay = env_layout(env)
    thetas = THETA_FAMILIES[theta_family]()
    partners = [make_teammate(HeuristicTeammateSpec(family=theta_family, theta=t.preset_name(),
                                                    extra=cfg_kwargs), env, "agent_1")
                for t in thetas]
    from partner_modeling.rollouts import FOCAL_CTRL_KEYS
    fcfg = {k: v for k, v in cfg_kwargs.items() if k in FOCAL_CTRL_KEYS}
    strat = scripted_strategies(lay, controller="aware", ctrl_cfg=fcfg)
    all_focals = {
        "static": StaticFocal(),
        "random": RandomFocal(),
        **strat,
        **strategy_family(lay, ctrl_cfg=fcfg),
    }
    if lock_L > 0:
        from partner_modeling.rollouts import locked_strategy_family
        all_focals.update(locked_strategy_family(lay, lock_L, ctrl_cfg=fcfg, partition=partition))
    model = TypedIntentModel(lay, agent_id=1, config=TypedIntentConfig(**cfg_kwargs))
    _G.update(env=env, partners=partners, focals={k: all_focals[k] for k in focals},
              model=model, observer=IdealObserver(model, thetas), thetas=thetas, max_steps=max_steps,
              compact=compact)


def _run_one(task):
    fh_dir = os.environ.get("SWEEP_FH_DIR")   # diagnostics only: dump tracebacks if a task runs > 60 s
    if fh_dir:
        import faulthandler
        if "fh_file" not in _G:
            _G["fh_file"] = open(os.path.join(fh_dir, f"worker_{os.getpid()}.log"), "a")
        _G["fh_file"].write(f"{time.strftime('%H:%M:%S')} start {task}\n")
        _G["fh_file"].flush()
        faulthandler.dump_traceback_later(60, repeat=True, file=_G["fh_file"])
    try:
        return _run_one_inner(task)
    finally:
        if fh_dir:
            faulthandler.cancel_dump_traceback_later()


def _run_one_inner(task):
    import jax
    from partner_modeling.observer import identification_time
    from partner_modeling.rollouts import partner_events, run_episode, stuck_stats

    focal_name, ti, seed = task
    env, partner, focal = _G["env"], _G["partners"][ti], _G["focals"][focal_name]
    key = jax.random.fold_in(jax.random.PRNGKey(seed), ti * 1000 + zlib.crc32(focal_name.encode()) % 997)
    t0 = time.time()
    log = run_episode(env, partner, focal, key, max_steps=_G["max_steps"])
    t_ep = time.time() - t0
    obs = _G["observer"]
    res = obs.run(log.views, log.partner_actions)
    replay = obs.replay_memory(log.views, log.partner_actions)
    ident = identification_time(res.post, ti)
    pos = np.array([v.pos for v in log.views])          # (T, 2, 2)
    if _G.get("compact"):
        # layout sweep: only what the cross-BR / identification / behaviour tables need
        return {
            "focal": focal_name, "theta": ti, "seed": seed, "T": log.T, "t_episode": t_ep,
            "soups": log.soups, "wrong": log.wrong_soups,
            "rewards": log.rewards.astype(np.int8 if np.all(np.abs(log.rewards) < 128) else np.float32),
            "partner_actions": log.partner_actions.astype(np.int8),
            "focal_actions": log.focal_actions.astype(np.int8),
            "pos": pos.astype(np.int8), "inv": np.array([v.inv for v in log.views]).astype(np.int16),
            "post": res.post.astype(np.float16),
            "mem_ok": bool(np.array_equal(replay, log.mem_before)),
            "ident_first": ident["first"], "ident_stable": ident["stable"],
            "events": partner_events(log, 1), "focal_events": partner_events(log, 0),
        }
    inv = np.array([v.inv for v in log.views])          # (T, 2)
    first_cook = next((t for t, v in enumerate(log.views)
                       if np.any((v.grid[:, :, 0] == 5) & (v.grid[:, :, 2] > 0))), log.T)
    return {
        "focal": focal_name, "theta": ti, "seed": seed,
        "soups": log.soups, "T": log.T, "t_episode": t_ep,
        "partner_actions": log.partner_actions.astype(np.int8),
        "focal_actions": log.focal_actions.astype(np.int8),
        "pos": pos.astype(np.int8), "inv": inv.astype(np.int16),
        "mem_logged": log.mem_before.astype(np.int8), "mem_replayed": replay.astype(np.int8),
        "mem_after_logged": log.mem_after.astype(np.int8),
        "post": res.post.astype(np.float32), "info": res.info.astype(np.float32),
        "pi": res.pi.astype(np.float32), "pibar": res.pibar.astype(np.float32),
        "intent_post": res.intent_post.sum(1).astype(np.float32),          # marginal over theta
        "intent_post_true": (res.intent_post[:, ti] / np.maximum(res.intent_post[:, ti].sum(-1, keepdims=True), 1e-300)).astype(np.float32),
        "intent_prior_true": res.intent_prior[:, ti].astype(np.float32),
        "H_theta": res.entropy_theta().astype(np.float32),
        "H_intent": res.entropy_intent_given_theta().astype(np.float32),
        "decision": res.decision, "adjacent": res.adjacent,
        "log_marginal": res.log_marginal.astype(np.float32),
        "ident_first": ident["first"], "ident_stable": ident["stable"],
        "events": partner_events(log, 1), "focal_events": partner_events(log, 0),
        "stuck": stuck_stats(log, 1), "first_cook": first_cook,
        "rewards": log.rewards.astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--focals", nargs="+",
                    default=["static", "random", "coop", "fetcher", "plater", "shuttle"])
    ap.add_argument("--thetas", nargs="*", type=int, default=None, help="type indices (default: all)")
    ap.add_argument("--theta-family", default="typed_intent", choices=["typed_intent", "typed_intent_x", "typed_intent_g"],
                    help="partner family: the original 12 types, the 8-type exclusive-role redesign, or the "
                         "8-type graded-exclusivity family (set q with --cfg '{\"excl_q\": q}')")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--layout", default="coord_ring")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--cfg", default="{}", help="JSON dict of TypedIntentConfig overrides")
    ap.add_argument("--full-obs", action="store_true", help="agent_view_size=None (see rollouts.make_study_env)")
    ap.add_argument("--compact", action="store_true", help="store a compact per-episode record")
    ap.add_argument("--family", action="store_true", help="focals = the 15-strategy library (x {F,P} with --lock)")
    ap.add_argument("--lock", type=int, default=0, help="role-commitment lock epoch L (0 = no lock)")
    ap.add_argument("--partition", default="fp", choices=["fp", "relay"],
                    help="lock role partition: fp = {F, P} (lock round), relay = {A, B} (gradedx companion)")
    args = ap.parse_args()
    cfg = json.loads(args.cfg)
    if args.family:
        args.focals = ["g_%s_%s_%s" % (p, s, g) for p in ("ff", "bal", "pf") for s in ("pile", "stg")
                       for g in ("nostage", "stage")] + ["fetcher", "plater", "shuttle"]
        if args.lock > 0:
            roles = ("F", "P") if args.partition == "fp" else ("A", "B")
            args.focals = [f"{f}@{r}" for f in args.focals for r in roles]
    if args.thetas is None:
        from partner_modeling.partners.typed_intent import THETA_FAMILIES
        args.thetas = list(range(len(THETA_FAMILIES[args.theta_family]())))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(f, ti, args.seed0 + s) for f in args.focals for ti in args.thetas for s in range(args.episodes)]
    t0 = time.time()
    ctx = get_context("spawn")
    results = []
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(cfg, args.focals, args.layout, args.max_steps, args.full_obs, args.compact,
                            args.theta_family, args.lock, args.partition)) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, tasks, chunksize=2)):
            results.append(r)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(tasks)} done, {time.time() - t0:.0f}s", flush=True)
    meta = {"cfg": cfg, "focals": args.focals, "episodes": args.episodes, "layout": args.layout,
            "max_steps": args.max_steps, "seed0": args.seed0, "wall_s": time.time() - t0,
            "full_obs": args.full_obs, "compact": args.compact,
            "theta_family": args.theta_family, "thetas": args.thetas, "lock": args.lock,
            "partition": args.partition}
    with open(out / "results.pkl", "wb") as f:
        pickle.dump({"meta": meta, "results": results}, f)
    print(f"saved {len(results)} episodes to {out / 'results.pkl'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
