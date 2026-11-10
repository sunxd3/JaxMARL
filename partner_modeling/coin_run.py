"""Play the Coin Game in JaxMARL's environment: red is the focal agent, blue a typed partner.

    python -m partner_modeling.coin_run scripted --games 20 --out outputs/partner_modeling/coin/scripted
    python -m partner_modeling.coin_run pp --games 10 --out outputs/partner_modeling/coin/pp
    python -m partner_modeling.coin_run llm --games 1 --model opus --out outputs/partner_modeling/coin/llm

Every episode is played in ``CoinGame``; the plain-Python model is stepped alongside and checked
against it after every move.  Red's score is what red plays for.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from partner_modeling.coin_agents import ClaudeCoinAgent, CoinAgent, PPCoinAgent, ScriptedRed
from partner_modeling.coin_model import TYPES, CoinEnv, partner_move, step
from partner_modeling.ppl import simulate


def play(game: CoinEnv, red: CoinAgent, kind: str, seed: int) -> Dict[str, Any]:
    import jax
    state, s = game.reset(jax.random.PRNGKey(seed))
    red.reset(seed, game.steps)
    rng = np.random.default_rng([seed, 1])
    keys = jax.random.split(jax.random.PRNGKey(seed + 10_000), game.steps)
    for t in range(game.steps):
        a_red = int(red.act(s))
        a_blue = int(simulate(partner_move, s, kind, rng=rng))
        state, rew = game.step(keys[t], state, a_red, a_blue)
        pos, coins = game.read(state)
        s2, r = step(s, a_red, a_blue, coins)
        assert r == rew and (s2.pos, s2.coins) == (pos, coins), "model and environment disagree"
        red.saw(s, a_red, a_blue, s2)
        s = s2
    return {"partner": kind, "seed": seed, "steps": game.steps, "red": s.score[0], "blue": s.score[1],
            "red_took_blue": s.steals[0], "blue_took_red": s.steals[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scripted", "pp", "llm"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--types", nargs="*", default=["cooperator", "greedy", "tit_for_tat"], choices=list(TYPES),
                    help="the partner strategies in play: partners are drawn from these and the agents know the set")
    ap.add_argument("--partners", nargs="*", default=None, choices=list(TYPES), help="default: every --types")
    ap.add_argument("--model", default=None, help="llm: passed to `claude --model`")
    ap.add_argument("--effort", default="low", help="llm: passed to `claude --effort`")
    ap.add_argument("--give-beliefs", action="store_true", help="llm: show the model the exact posterior")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    game = CoinEnv(args.steps)
    args.partners = args.partners or args.types
    seeds = range(args.seed0, args.seed0 + args.games)
    rows = []

    if args.mode == "scripted":
        print(f"red's mean score over {args.games} episodes of {args.steps} steps (rows: red's policy)")
        print(" " * 10 + "".join(f"{k:>13s}" for k in args.partners))
        for policy in ("own", "greedy", "oracle"):
            line = f"{policy:10s}"
            for kind in args.partners:
                rs = [play(game, ScriptedRed(policy, kind), kind, sd) for sd in seeds]
                rows += [{**r, "agent": policy} for r in rs]
                line += f"{np.mean([r['red'] for r in rs]):13.1f}"
            print(line, flush=True)
        (out / "scripted.json").write_text(json.dumps(rows, indent=1))
        return

    for kind in args.partners:
        for sd in seeds:
            t0 = time.time()
            if args.mode == "pp":
                agent = PPCoinAgent(types=args.types)
                r = play(game, agent, kind, sd)
                p = [st["belief"].get(kind, 0.0) for st in agent.trace]
                r.update(agent="pp", types=args.types, p_true_mean=float(np.mean(p)),
                         p_true_final=float(p[-1]), unexplained=agent.unexplained,
                         options={o: sum(st["option"] == o for st in agent.trace) for o in PPCoinAgent.OPTIONS})
                extra = f"P(true type) mean {r['p_true_mean']:.2f} final {r['p_true_final']:.2f}  {r['options']}"
            else:
                agent = ClaudeCoinAgent(model=args.model, effort=args.effort, give_beliefs=args.give_beliefs,
                                        types=args.types)
                r = play(game, agent, kind, sd)
                recs = [x for x in agent.records if x["belief"]]
                r.update(agent="claude", model=next((x["model"] for x in agent.records if x["model"]), args.model),
                         effort=args.effort, given_beliefs=args.give_beliefs, types=args.types,
                         decisions=len(agent.records),
                         fallbacks=int(sum(x["fallback"] for x in agent.records)),
                         cost_usd=float(sum(x["cost_usd"] for x in agent.records)),
                         p_true_mean={"llm": float(np.mean([x["belief"][kind] for x in recs])) if recs else None,
                                      "exact": float(np.mean([x["exact"][kind] for x in recs])) if recs else None},
                         choices={c: sum(x["choice"] == c for x in agent.records) for c in ("red_coin", "blue_coin", "stay")})
                with open(out / f"decisions_{kind}_s{sd}.jsonl", "w") as fh:
                    for x in agent.records:
                        fh.write(json.dumps(x) + "\n")
                extra = (f"P(true type) llm {r['p_true_mean']['llm']:.2f} exact {r['p_true_mean']['exact']:.2f}  "
                         f"{r['choices']}  {r['decisions']} calls ${r['cost_usd']:.2f}")
            r["wall_s"] = time.time() - t0
            rows.append(r)
            print(f"{r['agent']} vs {kind:12s} seed {sd}: red {r['red']:+3d} blue {r['blue']:+3d} "
                  f"(red took {r['red_took_blue']} blue coins, blue took {r['blue_took_red']} red)  {extra}  "
                  f"{r['wall_s']:.0f}s", flush=True)
    (out / f"{args.mode}_{args.seed0}.json").write_text(json.dumps(rows, indent=1))
    for kind in args.partners:
        rs = [r for r in rows if r["partner"] == kind]
        print(f"{args.mode} vs {kind:12s}: red's mean score {np.mean([r['red'] for r in rs]):+.1f}")


if __name__ == "__main__":
    main()
