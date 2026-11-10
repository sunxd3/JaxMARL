"""Run the probabilistic-program focal agent (``pp_focal.PPFocal``) against one typed-intent
partner, and compare it with the scripted focals and the exact observers.

    python -m partner_modeling.pp_run --layout asymm_advantages_recipes_center --theta runner_pot_direct_lo \
        --seeds 0 1 2 --max-steps 200 --out outputs/partner_modeling/pp/asymm_hidden

``--show-recipe`` gives the agent the current recipe (it then only has the partner's type to
infer).  The scripted baselines run on the same seeds and always know the recipe.  Writes one
``s<seed>.json`` per seed (scores, per-step beliefs, plans) and ``summary.json``.  No paid calls.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", default="asymm_advantages_recipes_center")
    ap.add_argument("--family", default="typed_intent", choices=["typed_intent", "typed_intent_x", "typed_intent_g"])
    ap.add_argument("--theta", default="runner_pot_direct_lo", help="partner preset name of --family")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--show-recipe", action="store_true", help="tell the agent the current recipe")
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--rollouts", type=int, default=2, help="rollouts per (option, recipe hypothesis)")
    ap.add_argument("--replan", type=int, default=4)
    ap.add_argument("--baselines", nargs="*", default=["coop", "plater"])
    ap.add_argument("--cfg", default='{"make_way_station_only": true}', help="JSON dict of TypedIntentConfig overrides")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import jax
    from partner_modeling.llm_focal import recipe_code, ingredients_of
    from partner_modeling.llm_run import pot_adds
    from partner_modeling.observer import IdealObserver, recipe_filter
    from partner_modeling.partners.typed_intent import THETA_FAMILIES, TypedIntentConfig, TypedIntentModel, _theta_tuple
    from partner_modeling.pp_focal import PPFocal
    from partner_modeling.pp_models import encode_recipe
    from partner_modeling.rollouts import FOCAL_CTRL_KEYS, env_layout, make_study_env, run_episode, scripted_strategies
    from partner_modeling.teammates import HeuristicTeammateSpec, make_teammate

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.cfg)
    fcfg = {k: v for k, v in cfg.items() if k in FOCAL_CTRL_KEYS}
    env = make_study_env(args.layout, args.max_steps)
    lay = env_layout(env)
    thetas = THETA_FAMILIES[args.family]()
    ti = [t.preset_name() for t in thetas].index(args.theta)
    partner = make_teammate(HeuristicTeammateSpec(family=args.family, theta=args.theta, extra=cfg), env, "agent_1")
    pm = TypedIntentModel(lay, agent_id=1, config=TypedIntentConfig(**cfg))
    recipes = np.asarray(lay.possible_recipes).tolist()
    enc = [encode_recipe(r) for r in recipes]
    strat = scripted_strategies(lay, ctrl_cfg=fcfg)

    def score(lg):
        adds, broke = pot_adds(lg)
        return {"reward": float(np.sum(lg.rewards)), "soups": lg.soups, "wrong": lg.wrong_soups,
                "adds": adds[0], "adds_that_broke_a_pot": broke[0]}

    rows = []
    for seed in args.seeds:
        key = jax.random.PRNGKey(seed)
        focal = PPFocal(env, family=args.family, cfg=cfg, hide_recipe=not args.show_recipe,
                        horizon=args.horizon, n_rollouts=args.rollouts, replan=args.replan, seed=seed,
                        verbose=args.verbose)
        t0 = time.time()
        log = run_episode(env, partner, focal, key, max_steps=args.max_steps)
        wall = time.time() - t0
        # exact references: the recipe observer knows the partner's type and sees its actions;
        # the type observer knows the recipe.  Both give the belief before the action at step t.
        prior_pred, _, true_idx = recipe_filter(pm, thetas[ti], log.views, log.partner_actions, log.rewards, recipes)
        res = IdealObserver(pm, thetas).run(log.views, log.partner_actions)
        tt = _theta_tuple(thetas[ti])
        p_recipe = np.array([focal.trace[t]["recipe"].get(enc[true_idx[t]], 0.0) for t in range(log.T)])
        p_type = np.array([focal.trace[t]["type"].get(tt, 0.0) for t in range(log.T)])
        row = {"seed": seed, "pp": score(log), "wall_s": wall, "plans": len(focal.plans),
               "unexplained_steps": focal.impossible,
               "baselines": {n: score(run_episode(env, partner, strat[n], key, max_steps=args.max_steps))
                             for n in args.baselines},
               "p_true_recipe": {"pp": float(p_recipe.mean()),
                                 "observer": float(prior_pred[np.arange(log.T), true_idx].mean())},
               "p_true_type_final": {"pp": float(p_type[-1]), "observer": float(res.prior_pred[-1, ti])},
               "options": {o: sum(s["option"] == o for s in focal.trace) for o in {s["option"] for s in focal.trace}}}
        row["wrong_soups_by"] = {"pp": sum(d["who"] == 0 and not d["correct"] for d in focal.deliveries),
                                 "partner": sum(d["who"] == 1 and not d["correct"] for d in focal.deliveries)}
        rows.append(row)
        (out / f"s{seed}.json").write_text(json.dumps({
            **row, "true_recipe": [recipe_code(recipes[i]) for i in true_idx],
            "belief_recipe": [{recipe_code(ingredients_of(r)): round(p, 4) for r, p in s["recipe"].items()}
                              for s in focal.trace],
            "p_true_recipe_trace": {"pp": p_recipe.round(4).tolist(),
                                    "observer": prior_pred[np.arange(log.T), true_idx].round(4).tolist()},
            "option_trace": [s["option"] for s in focal.trace], "plan_log": focal.plans,
            "deliveries": focal.deliveries}, indent=1))
        b = "  ".join(f"{n} {s['reward']:+.0f} ({s['soups']}/{s['wrong']})" for n, s in row["baselines"].items())
        print(f"seed {seed}: pp {row['pp']['reward']:+.0f} ({row['pp']['soups']} correct/{row['pp']['wrong']} wrong, "
              f"{row['pp']['adds']} adds, {row['pp']['adds_that_broke_a_pot']} broke)  |  {b}  |  "
              f"P(true recipe) pp {row['p_true_recipe']['pp']:.2f} obs {row['p_true_recipe']['observer']:.2f}  |  "
              f"{row['plans']} plans, {wall:.0f}s", flush=True)

    def mean(f):
        return float(np.mean([f(r) for r in rows]))

    summary = {"layout": args.layout, "family": args.family, "theta": args.theta, "seeds": args.seeds,
               "max_steps": args.max_steps, "show_recipe": args.show_recipe,
               "horizon": args.horizon, "rollouts": args.rollouts, "replan": args.replan,
               "mean_reward": {"pp": mean(lambda r: r["pp"]["reward"]),
                               **{n: mean(lambda r, n=n: r["baselines"][n]["reward"]) for n in args.baselines}},
               "mean_wrong": {"pp": mean(lambda r: r["pp"]["wrong"]),
                              **{n: mean(lambda r, n=n: r["baselines"][n]["wrong"]) for n in args.baselines}},
               "mean_p_true_recipe": {"pp": mean(lambda r: r["p_true_recipe"]["pp"]),
                                      "observer": mean(lambda r: r["p_true_recipe"]["observer"])},
               "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("mean reward: " + "  ".join(f"{k} {v:+.1f}" for k, v in summary["mean_reward"].items())
          + f"   mean P(true recipe): pp {summary['mean_p_true_recipe']['pp']:.2f} "
            f"observer {summary['mean_p_true_recipe']['observer']:.2f}")


if __name__ == "__main__":
    main()
