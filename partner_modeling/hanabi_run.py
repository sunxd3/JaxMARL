"""Play the small Hanabi in JaxMARL's environment with scripted, probabilistic-program or LLM players.

    # how much does sharing a convention matter?  every convention against every convention
    python -m partner_modeling.hanabi_run cross --games 30 --out outputs/partner_modeling/hanabi/cross

    # the probabilistic-program player against each partner convention
    python -m partner_modeling.hanabi_run pp --games 10 --out outputs/partner_modeling/hanabi/pp

Every game is played in ``HanabiEnv``; the plain-Python model is stepped alongside and checked
against it after every move.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from partner_modeling.hanabi_agents import ConventionAgent, HanabiAgent, PPHanabiAgent
from partner_modeling.hanabi_model import MiniHanabi, Rules, legal_moves, model_public, step, touched_slots
from partner_modeling.hanabi_partner import CONVENTIONS, Convention, observe


def play_game(game: MiniHanabi, agents: Sequence[HanabiAgent], seed: int, focal: Optional[int] = None,
              partner_conv: Optional[Convention] = None) -> Dict[str, Any]:
    """One game.  With ``focal`` set, also records that player's beliefs against the truth at
    each of its turns (it must expose ``trace`` entries with 'convention' and 'slots')."""
    import jax
    rules = game.rules
    state, s = game.reset(jax.random.PRNGKey(seed))
    for i, ag in enumerate(agents):
        ag.reset(rules, i, seed)
    key = jax.random.PRNGKey(0)
    moves, beliefs = [], []
    while not s.done:
        p = s.cur
        before = [observe(s, 0), observe(s, 1)]
        a = int(agents[p].act(before[p]))
        assert a in legal_moves(rules, s), (a, legal_moves(rules, s))
        if p == focal and getattr(agents[p], "trace", None):          # probabilistic-program player
            tr = agents[p].trace[-1]
            beliefs.append({"move_no": len(moves),
                            "p_signal": sum(q for c, q in tr["convention"].items() if c.signal == partner_conv.signal),
                            "p_convention": tr["convention"].get(partner_conv, 0.0),
                            "p_card": [float(d.get(s.hands[p][i], 0.0)) for i, d in enumerate(tr["slots"])],
                            "p_playable": [float(sum(q for c, q in d.items() if s.playable(c))) for d in tr["slots"]],
                            "playable": [bool(s.playable(c)) for c in s.hands[p]]})
        elif p == focal and getattr(agents[p], "records", None):      # LLM player, with exact beliefs alongside
            rec = agents[p].records[-1]
            beliefs.append({"move_no": len(moves), "choice": rec["choice"], "fallback": rec["fallback"],
                            "p_signal": rec["signal"][partner_conv.signal] if rec["signal"] else None,
                            "exact_p_signal": rec["exact_signal"][partner_conv.signal],
                            "p_playable": rec["slot_playable"], "exact_p_playable": rec["exact_slot_playable"],
                            "playable": [bool(s.playable(c)) for c in s.hands[p]],
                            "has_card": [c is not None for c in s.hands[p]]})
        kind, k = rules.kind(a)
        revealed = s.hands[p][k] if kind in ("discard", "play") else None
        hit = tuple(touched_slots(rules, s, a)) if kind in ("color", "rank") else None
        state, _, _ = game.step(key, state, a)
        s2, _ = step(rules, s, a)
        assert game.public(state) == model_public(s2), "model and environment disagree"
        after = [observe(s2, 0), observe(s2, 1)]
        for i, ag in enumerate(agents):
            ag.saw(before[i], p, a, revealed, hit, after[i])
        moves.append((p, a))
        s = s2
    return {"seed": seed, "score": s.score, "cards_played": sum(s.fireworks), "lives": s.lives,
            "moves": len(moves), "beliefs": beliefs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cross", "pp", "llm"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--partners", nargs="*", type=int, default=None, help="indices into CONVENTIONS (default: all)")
    ap.add_argument("--worlds", type=int, default=40, help="pp: imagined games per decision")
    ap.add_argument("--exclude-true-signal", action="store_true",
                    help="pp: the player's hypotheses leave out what a hint really means to this partner "
                         "(exact inference in a wrong model class)")
    ap.add_argument("--model", default=None, help="llm: passed to `claude --model`")
    ap.add_argument("--effort", default="low", help="llm: passed to `claude --effort`")
    ap.add_argument("--give-beliefs", action="store_true", help="llm: show the model the exact posterior")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    game = MiniHanabi(Rules())
    seeds = range(args.seed0, args.seed0 + args.games)
    partners = [CONVENTIONS[i] for i in args.partners] if args.partners is not None else list(CONVENTIONS)
    name = lambda c: f"{c.signal}/{c.chop}"

    if args.mode == "cross":
        table = {}
        for a in CONVENTIONS:
            for b in partners:
                rs = [play_game(game, [ConventionAgent(a), ConventionAgent(b)] if sd % 2 == 0
                                else [ConventionAgent(b), ConventionAgent(a)], sd) for sd in seeds]
                table[f"{name(a)} with {name(b)}"] = {"score": float(np.mean([r["score"] for r in rs])),
                                                      "cards_played": float(np.mean([r["cards_played"] for r in rs])),
                                                      "lost_all_lives": float(np.mean([r["lives"] == 0 for r in rs]))}
        (out / "cross.json").write_text(json.dumps(table, indent=1))
        print("mean score (rows: focal convention, columns: partner convention)")
        print(" " * 16 + "".join(f"{name(b):>16s}" for b in partners))
        for a in CONVENTIONS:
            print(f"{name(a):16s}" + "".join(f"{table[f'{name(a)} with {name(b)}']['score']:16.1f}" for b in partners))
        return

    if args.mode == "llm":
        from partner_modeling.hanabi_llm import ClaudeHanabiAgent
        rows = []
        for b in partners:
            for sd in seeds:
                seat = sd % 2
                agents = [None, None]
                agents[seat] = ClaudeHanabiAgent(model=args.model, effort=args.effort, give_beliefs=args.give_beliefs)
                agents[1 - seat] = ConventionAgent(b)
                t0 = time.time()
                r = play_game(game, agents, sd, focal=seat, partner_conv=b)
                bl, recs = r.pop("beliefs"), agents[seat].records

                def brier(key):
                    v = [(q - float(t)) ** 2 for x in bl if x[key] is not None
                         for q, t, h in zip(x[key], x["playable"], x["has_card"]) if h and q is not None]
                    return float(np.mean(v)) if v else None

                ok = [x for x in bl if x["p_signal"] is not None]
                r.update(partner=name(b), seat=seat, wall_s=time.time() - t0, turns=len(recs),
                         fallbacks=int(sum(x["fallback"] for x in recs)),
                         cost_usd=float(sum(x["cost_usd"] for x in recs)),
                         model=next((x["model"] for x in recs if x["model"]), args.model),
                         effort=args.effort, given_beliefs=args.give_beliefs,
                         p_signal={"llm": float(np.mean([x["p_signal"] for x in ok])) if ok else None,
                                   "exact": float(np.mean([x["exact_p_signal"] for x in ok])) if ok else None},
                         brier_playable={"llm": brier("p_playable"), "exact": brier("exact_p_playable")},
                         p_signal_by_turn={"llm": [round(x["p_signal"], 2) if x["p_signal"] is not None else None for x in bl],
                                           "exact": [round(x["exact_p_signal"], 2) for x in bl]},
                         choices=[x["choice"] for x in bl])
                rows.append(r)
                with open(out / f"decisions_{name(b).replace('/', '_')}_s{sd}.jsonl", "w") as fh:
                    for x in recs:
                        fh.write(json.dumps(x) + "\n")
                print(f"claude vs {name(b):16s} seed {sd}: score {r['score']:2d} (cards {r['cards_played']}, lives "
                      f"{r['lives']})  P(true signal) llm {r['p_signal']['llm']:.2f} exact {r['p_signal']['exact']:.2f}  "
                      f"playable Brier llm {r['brier_playable']['llm']:.3f} exact {r['brier_playable']['exact']:.3f}  "
                      f"{r['turns']} turns ${r['cost_usd']:.2f} {r['wall_s']:.0f}s", flush=True)
        (out / f"llm_{args.seed0}.json").write_text(json.dumps(rows, indent=1))
        return

    rows = []
    for b in partners:
        for sd in seeds:
            seat = sd % 2
            agents: List[HanabiAgent] = [None, None]
            assumed = [c for c in CONVENTIONS if not (args.exclude_true_signal and c.signal == b.signal)]
            agents[seat] = PPHanabiAgent(conventions=assumed, worlds=args.worlds)
            agents[1 - seat] = ConventionAgent(b)
            t0 = time.time()
            r = play_game(game, agents, sd, focal=seat, partner_conv=b)
            bl = r.pop("beliefs")
            r.update(partner=name(b), seat=seat, wall_s=time.time() - t0, unexplained=agents[seat].unexplained,
                     assumed=sorted({c.signal for c in assumed}),
                     p_signal_by_turn=[round(x["p_signal"], 3) for x in bl],
                     p_card=float(np.mean([np.mean(x["p_card"]) for x in bl])) if bl else None,
                     brier_playable=float(np.mean([np.mean([(q - float(t)) ** 2 for q, t in
                                                            zip(x["p_playable"], x["playable"])]) for x in bl])) if bl else None)
            rows.append(r)
            print(f"pp vs {name(b):16s} seed {sd}: score {r['score']:2d} (cards {r['cards_played']}, lives {r['lives']}) "
                  f"P(signal) {' '.join(f'{x:.2f}' for x in r['p_signal_by_turn'][:6])}  {r['wall_s']:.0f}s", flush=True)
    (out / f"pp_{args.seed0}.json").write_text(json.dumps(rows, indent=1))
    for b in partners:
        rs = [r for r in rows if r["partner"] == name(b)]
        print(f"pp vs {name(b):16s}: mean score {np.mean([r['score'] for r in rs]):.1f}, "
              f"cards {np.mean([r['cards_played'] for r in rs]):.1f}, lost all lives {np.mean([r['lives'] == 0 for r in rs]):.2f}")


if __name__ == "__main__":
    main()
