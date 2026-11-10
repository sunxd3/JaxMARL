"""Run a ``claude -p`` focal agent against one typed-intent partner and compare it with the
scripted focals and the exact observers.

    # station agent, recipe hidden from it, on a multi-recipe layout
    python -m partner_modeling.llm_run --layout cramped_room_v2 --hide-recipe --theta flex_pot_direct_lo \
        --max-steps 200 --out outputs/partner_modeling/llm/hidden_s0

    # task agent on a classic layout
    python -m partner_modeling.llm_run --agent tasks --family typed_intent_x --theta plater_x_pot_direct_lo \
        --out outputs/partner_modeling/llm/plater_x_s0

Writes ``decisions.jsonl`` (one line per LLM decision, with the full prompt), ``trace.txt`` and
``summary.json`` under ``--out``.  The scripted baselines run on the same seed, so they start
from the same state and face the same partner; they always know the recipe.  Every LLM decision
is a paid ``claude -p`` call.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np


def pot_adds(log):
    """Per agent: ingredients added to a pot, and how many of those adds made a pot that still
    fitted the recipe in force stop fitting it."""
    from partner_modeling.env_compat import DynamicObject, StaticObject
    from partner_modeling.llm_focal import ingredients_of
    from partner_modeling.rollouts import _fwd

    adds, broke = [0, 0], [0, 0]
    for t in range(log.T - 1):
        v0, v1 = log.views[t], log.views[t + 1]
        for ag in (0, 1):
            inv0, inv1 = int(v0.inv[ag]), int(v1.inv[ag])
            if inv1 != 0 or inv0 == 0 or inv0 & (DynamicObject.PLATE | DynamicObject.COOKED):
                continue
            fy, fx = _fwd(v1, ag)
            if not (0 <= fy < v0.grid.shape[0] and 0 <= fx < v0.grid.shape[1]) or v0.grid[fy, fx, 0] != StaticObject.POT:
                continue
            before, after = ingredients_of(v0.grid[fy, fx, 1]), ingredients_of(v1.grid[fy, fx, 1])
            if len(after) != len(before) + 1:
                continue
            recipe = Counter(ingredients_of(v0.recipe))
            adds[ag] += 1
            broke[ag] += int(not (Counter(before) - recipe) and bool(Counter(after) - recipe))
    return adds, broke


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent", default="stations", choices=["stations", "tasks"],
                    help="stations: pick any station to interact with; tasks: pick among the valid intents")
    ap.add_argument("--layout", default="coord_ring")
    ap.add_argument("--family", default="typed_intent",
                    choices=["typed_intent", "typed_intent_x", "typed_intent_g"])
    ap.add_argument("--theta", default="runner_pot_direct_lo", help="partner preset name of --family")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--hide-recipe", action="store_true",
                    help="stations agent, multi-recipe layouts: do not tell the model the current recipe")
    ap.add_argument("--ask", nargs="*", default=None, choices=["recipe", "partner_role", "partner_staging"],
                    help="stations agent: beliefs to request (default: recipe if hidden, else role and staging)")
    ap.add_argument("--give-beliefs", action="store_true",
                    help="stations agent: put the exact recipe posterior in the prompt")
    ap.add_argument("--model", default=None, help="passed to `claude --model` (default: the CLI's default model)")
    ap.add_argument("--effort", default="low", help="passed to `claude --effort`")
    ap.add_argument("--max-hold", type=int, default=8, help="ask again after this many steps at the latest")
    ap.add_argument("--baselines", nargs="*", default=["coop", "fetcher", "plater"])
    ap.add_argument("--cfg", default='{"make_way_station_only": true}',
                    help="JSON dict of TypedIntentConfig overrides (graded family: add excl_q)")
    args = ap.parse_args()

    import jax
    from partner_modeling.llm_focal import ROLE_LABEL, ClaudeFocal, ClaudeStationFocal, recipe_code
    from partner_modeling.observer import IdealObserver, recipe_filter
    from partner_modeling.partners.typed_intent import (
        STAGING_NAMES, THETA_FAMILIES, TypedIntentConfig, TypedIntentModel)
    from partner_modeling.rollouts import (
        FOCAL_CTRL_KEYS, env_layout, make_study_env, render_trace, run_episode, scripted_strategies)
    from partner_modeling.teammates import HeuristicTeammateSpec, make_teammate

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "decisions.jsonl").unlink(missing_ok=True)
    cfg = json.loads(args.cfg)
    fcfg = {k: v for k, v in cfg.items() if k in FOCAL_CTRL_KEYS}
    env = make_study_env(args.layout, args.max_steps)
    lay = env_layout(env)
    thetas = THETA_FAMILIES[args.family]()
    ti = [t.preset_name() for t in thetas].index(args.theta)
    partner = make_teammate(HeuristicTeammateSpec(family=args.family, theta=args.theta, extra=cfg), env, "agent_1")
    key = jax.random.PRNGKey(args.seed)

    kw = dict(family=args.family, model=args.model, effort=args.effort, max_hold=args.max_hold,
              max_steps=args.max_steps, ctrl_cfg=fcfg, log_path=str(out / "decisions.jsonl"), verbose=True)
    tracker = None
    if args.give_beliefs:
        from partner_modeling.pp_focal import PPFocal
        tracker = PPFocal(env, family=args.family, cfg=cfg, hide_recipe=True, horizon=0, replan=10 ** 6,
                          seed=args.seed)
    focal = (ClaudeStationFocal(lay, hide_recipe=args.hide_recipe, ask=args.ask, belief_tracker=tracker, **kw)
             if args.agent == "stations" else ClaudeFocal(lay, **kw))
    print(f"{focal.name} vs {args.family}[{args.theta}] on {args.layout}, seed {args.seed}"
          + (", recipe hidden" if getattr(focal, "hide_recipe", False) else ""), flush=True)
    t0 = time.time()
    log = run_episode(env, partner, focal, key, max_steps=args.max_steps)
    wall = time.time() - t0
    recs = focal.records
    (out / "trace.txt").write_text(render_trace(log) + "\n")
    partner_model = TypedIntentModel(lay, agent_id=1, config=TypedIntentConfig(**cfg))

    def fmt(x):
        return "  -  " if x is None else f"{x:.2f}"

    summary = {}
    # Beliefs are compared with the observer's belief *before* a_t (prior_pred[t]): that is what
    # an observer knows at the moment the LLM decides at step t.
    if "partner_role" in focal.beliefs or "partner_staging" in focal.beliefs:
        res = IdealObserver(partner_model, thetas).run(log.views, log.partner_actions)
        role_of = np.array([ROLE_LABEL[int(t.role)] for t in thetas])
        stag_of = np.array([STAGING_NAMES[int(t.staging)] for t in thetas])
        true_role, true_stag = role_of[ti], stag_of[ti]
        rows = []
        for r in recs:
            t, b = r["t"], r["beliefs"]
            rows.append({
                "t": t, "choice": r["choice"], "fallback": r["fallback"],
                "llm_role": b["partner_role"][true_role] if b.get("partner_role") else None,
                "obs_role": float(res.prior_pred[t, role_of == true_role].sum()),
                "llm_staging": b["partner_staging"][true_stag] if b.get("partner_staging") else None,
                "obs_staging": float(res.prior_pred[t, stag_of == true_stag].sum()),
            })
        summary.update(true_role=str(true_role), true_staging=str(true_stag), type_beliefs=rows)
        print(f"\npartner {args.theta}: true role = {true_role}, true staging = {true_stag}")
        print("P(true role) and P(true staging): LLM report vs ideal observer, at each LLM decision")
        print("   t  choice          role llm / obs    staging llm / obs")
        for r in rows:
            print(f" {r['t']:3d}  {r['choice']:14s}  {fmt(r['llm_role'])} / {fmt(r['obs_role'])}"
                  f"       {fmt(r['llm_staging'])} / {fmt(r['obs_staging'])}" + ("   [fallback]" if r["fallback"] else ""))
    if "recipe" in focal.beliefs:
        recipes = np.asarray(lay.possible_recipes).tolist()
        codes = [recipe_code(r) for r in recipes]
        prior_pred, _, true_idx = recipe_filter(partner_model, thetas[ti], log.views, log.partner_actions,
                                                log.rewards, recipes)
        rows = []
        for r in recs:
            t, b = r["t"], r["beliefs"].get("recipe")
            true = codes[true_idx[t]]
            rows.append({"t": t, "choice": r["choice"], "fallback": r["fallback"], "true": true,
                         "llm": b[true] if b else None, "llm_top": max(b, key=b.get) if b else None,
                         "obs": float(prior_pred[t, true_idx[t]]),
                         "obs_top": codes[int(np.argmax(prior_pred[t]))]})
        ok = [r for r in rows if r["llm"] is not None]
        summary.update(recipe_beliefs=rows,
                       mean_p_true_recipe={"llm": float(np.mean([r["llm"] for r in ok])) if ok else None,
                                           "observer": float(np.mean([r["obs"] for r in ok])) if ok else None})
        print("\nP(true recipe): LLM report vs exact recipe observer (partner type known), at each LLM decision")
        print("   t  choice          true   llm (top)     obs (top)")
        for r in rows:
            print(f" {r['t']:3d}  {r['choice']:14s}  {r['true']}    {fmt(r['llm'])} ({r['llm_top']})    "
                  f"{fmt(r['obs'])} ({r['obs_top']})" + ("   [fallback]" if r["fallback"] else ""))
        if ok:
            m = summary["mean_p_true_recipe"]
            print(f"mean P(true recipe) over decisions: llm {m['llm']:.2f}, observer {m['observer']:.2f}")

    def score(lg):
        adds, broke = pot_adds(lg)
        return {"reward": float(np.sum(lg.rewards)), "soups": lg.soups, "wrong": lg.wrong_soups,
                "focal_adds": adds[0], "focal_adds_that_broke_a_pot": broke[0],
                "partner_adds": adds[1], "partner_adds_that_broke_a_pot": broke[1]}

    strat = scripted_strategies(lay, ctrl_cfg=fcfg)
    base = {name: score(run_episode(env, partner, strat[name], key, max_steps=args.max_steps))
            for name in args.baselines}
    mine = score(log)
    lat = [r["latency_s"] for r in recs]
    summary = {
        "agent": focal.name, "layout": args.layout, "family": args.family, "theta": args.theta, "seed": args.seed,
        "max_steps": args.max_steps, "hide_recipe": bool(getattr(focal, "hide_recipe", False)),
        "given_beliefs": bool(args.give_beliefs),
        "model": next((r["model"] for r in recs if r["model"]), args.model), "effort": args.effort,
        "claude": mine, "baselines": base, "decisions": len(recs),
        "fallbacks": int(sum(r["fallback"] for r in recs)), "cost_usd": float(sum(r["cost_usd"] for r in recs)),
        "mean_latency_s": float(np.mean(lat)) if lat else 0.0, "wall_s": wall,
        "choices": dict(Counter(r["choice"] for r in recs)), **summary,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    print("\n            reward  soups  wrong  own pot adds (broke a pot)")
    for name, s in [("claude", mine)] + list(base.items()):
        print(f"  {name:8s}  {s['reward']:+6.0f}  {s['soups']:5d}  {s['wrong']:5d}  "
              f"{s['focal_adds']:3d} ({s['focal_adds_that_broke_a_pot']})")
    print(f"{len(recs)} decisions, {summary['fallbacks']} fallbacks, ${summary['cost_usd']:.2f}, "
          f"{summary['mean_latency_s']:.1f}s per call, {wall:.0f}s wall, model {summary['model']}")


if __name__ == "__main__":
    main()
