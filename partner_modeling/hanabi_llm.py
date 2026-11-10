"""A Hanabi player that asks ``claude -p`` for every move.

Each call is stateless: the prompt holds the table, what the hints say about each card, the
recent moves, and the short notes the model wrote on its previous turn.  The model is told the
partner's possible conventions in words.  Besides its move it reports its belief about the
partner's hint convention and, for each of its own cards, the probability that the card is
playable now.

A ``PPHanabiAgent`` rides along as a silent observer of the same seat, so every reported belief
can be compared with the exact posterior at that moment.  The model never sees it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from partner_modeling.hanabi_agents import HanabiAgent, PPHanabiAgent
from partner_modeling.hanabi_model import COLOR_NAMES, Card, HState, Rules, legal_moves
from partner_modeling.hanabi_partner import EPS
from partner_modeling.llm_focal import ClaudeCLI, _normalise

SIGNALS = ("newest", "oldest", "literal")

SYSTEM = """\
You are playing a small two-player Hanabi with a scripted partner (PAR). You share one score.

The game: {colors} colours ({color_names}), ranks 1 to {ranks}; each colour has {copies} copies of \
ranks 1..{ranks} respectively. Each player holds {hand} cards and sees the partner's cards but not \
their own. Fireworks are built per colour from rank 1 upwards. On your turn you do one thing:
- play a card: if it is the next rank of its colour it is added to the fireworks (+1 point), \
otherwise a life is lost and the card is discarded;
- discard a card: wins back one hint token (not allowed when all {max_info} tokens are available);
- give a hint (costs one token): name one colour or one rank, and every card in the partner's \
hand that matches is pointed out.
After a play or discard the remaining cards shift to lower slots and a new card enters at the \
last slot, so slot 0 is always your oldest card. The game ends when the deck runs out and each \
player has had one more turn, when all fireworks are complete, or when the last of {max_lives} \
lives is lost. Losing the last life forfeits the whole score.

PAR is scripted and its rule is fixed for the game. On its turn it does the first of these that \
applies: (1) play a card it believes is playable; (2) if a hint token is available, give a hint \
about a playable card of yours; (3) discard a card it has no hint about; (4) otherwise give some \
hint. With probability {eps} it makes a random legal move instead. What PAR believes, and which \
hints it gives, depend on a convention you are not told:
- signal "newest": a hint means "play the newest card this hint touched". PAR plays the newest \
card a hint of yours touched, and when it hints you, the newest touched card is playable.
- signal "oldest": the same with the oldest touched card.
- signal "literal": a hint means only what it says. PAR plays a card only once the hints it has \
received leave no identity that is not playable, and it hints to pin your playable cards down.
- its discard: the oldest, or the newest, of its cards that no hint has touched.
PAR does not adapt to you.

Reply with:
- reasoning: at most 60 words.
- partner_signal: your probability for each signal (newest / oldest / literal), summing to 1.
- slot_playable: for each of your slots, your probability that the card there is playable right now.
- choice: one of the offered move names.
- notes: at most 40 words. This is your only memory; it is shown to you on your next turn."""


def _code(card: Optional[Card]) -> str:
    return "--" if card is None else f"{COLOR_NAMES[card[0]][0].upper()}{card[1] + 1}"


def _possible(rules: Rules, mask: int) -> str:
    ids = [(c, r) for c in range(rules.colors) for r in range(rules.ranks) if mask & rules.bit((c, r))]
    return " ".join(_code(c) for c in ids)


class ClaudeHanabiAgent(HanabiAgent):
    name = "claude"

    def __init__(self, model: Optional[str] = None, effort: Optional[str] = "low", timeout: float = 180.0,
                 give_beliefs: bool = False, verbose: bool = False):
        """``give_beliefs``: put the exact posterior (partner signal, and for each own card the
        probability that it is playable) in the prompt, so that only the decision is the model's."""
        self.cli = ClaudeCLI(model=model, effort=effort, timeout=timeout)
        self.shadow = PPHanabiAgent(worlds=1)       # exact beliefs; never consulted for moves
        self.give_beliefs, self.verbose = give_beliefs, verbose

    def reset(self, rules: Rules, me: int, seed: int):
        super().reset(rules, me, seed)
        self.shadow.reset(rules, me, seed)
        self.system = SYSTEM.format(colors=rules.colors, color_names=", ".join(COLOR_NAMES[:rules.colors]),
                                    ranks=rules.ranks, copies="/".join(str(n) for n in rules.copies),
                                    hand=rules.hand, max_info=rules.max_info, max_lives=rules.max_lives, eps=EPS)
        self.history: List[str] = []
        self.notes = ""
        self.records: List[Dict[str, Any]] = []

    def _name(self, action: int) -> str:
        kind, k = self.rules.kind(action)
        return {"discard": f"discard_{k}", "play": f"play_{k}", "color": f"hint_{COLOR_NAMES[k]}",
                "rank": f"hint_rank_{k + 1}"}[kind]

    def saw(self, before: HState, actor: int, action: int, revealed: Optional[Card],
            hit: Optional[Tuple[int, ...]], after: HState):
        self.shadow.saw(before, actor, action, revealed, hit, after)
        who = "YOU" if actor == self.me else "PAR"
        kind, k = self.rules.kind(action)
        if kind == "play":
            ok = sum(after.fireworks) > sum(before.fireworks)
            text = f"played slot {k}: {_code(revealed)}, " + ("added to the fireworks" if ok else "NOT playable, a life lost")
        elif kind == "discard":
            text = f"discarded slot {k}: {_code(revealed)}"
        else:
            what = COLOR_NAMES[k] if kind == "color" else f"rank {k + 1}"
            whose = "PAR's" if actor == self.me else "your"
            text = f"hinted {what}: touched {whose} slot" + ("s " if len(hit) > 1 else " ") + ", ".join(map(str, hit))
        self.history.append(f"{who} {text}")

    def _prompt(self, obs: HState, names: List[str]) -> str:
        r, me, par = self.rules, self.me, 1 - self.me
        fire = ", ".join(f"{COLOR_NAMES[c]} {obs.fireworks[c]}" for c in range(r.colors))
        out = [f"Fireworks (highest rank played per colour): {fire}. Score {sum(obs.fireworks)}.",
               f"Hint tokens {obs.info} of {r.max_info}. Lives {obs.lives}. Cards left in the deck: {len(obs.deck)}.",
               "Discarded so far: " + (" ".join(_code(c) for c in sorted(obs.discards)) or "nothing") + ".", "",
               "Cards are written as colour initial + rank (R1 = red 1).",
               "PAR's hand (you see it; in brackets, what the hints leave possible from PAR's side):"]

        def flags(p, i):
            f = [n for n, on in (("touched by a hint", obs.touched[p][i]),
                                 ("was the newest card a hint touched", obs.newest[p][i]),
                                 ("was the oldest card a hint touched", obs.oldest[p][i])) if on]
            return "; ".join(f) if f else "no hint has touched it"

        for i, c in enumerate(obs.hands[par]):
            if c is not None:
                out.append(f"  slot {i}: {_code(c)}  [{_possible(r, obs.masks[par][i])}]  {flags(par, i)}")
        out.append("Your hand (hidden from you; what the hints leave possible):")
        for i, c in enumerate(obs.hands[me]):
            if c is not None:
                out.append(f"  slot {i}: [{_possible(r, obs.masks[me][i])}]  {flags(me, i)}")
        out += ["", "Moves so far (most recent last; slots are as they were at the time):"]
        out += [f"  {h}" for h in self.history[-14:]] or ["  none"]
        out += ["", f'Your notes from your last turn: "{self.notes}"' if self.notes else "You have no notes yet.",
                "", "Moves you can choose now: " + ", ".join(names)]
        return "\n".join(out)

    def act(self, obs: HState) -> int:
        r = self.rules
        legal = legal_moves(r, obs)
        names = [self._name(a) for a in legal]
        slots = [f"slot_{i}" for i in range(r.hand)]
        schema = {"type": "object", "additionalProperties": False,
                  "required": ["reasoning", "partner_signal", "slot_playable", "choice", "notes"],
                  "properties": {
                      "reasoning": {"type": "string"},
                      "partner_signal": {"type": "object", "additionalProperties": False, "required": list(SIGNALS),
                                         "properties": {k: {"type": "number"} for k in SIGNALS}},
                      "slot_playable": {"type": "object", "additionalProperties": False, "required": slots,
                                        "properties": {k: {"type": "number"} for k in slots}},
                      "choice": {"type": "string", "enum": names},
                      "notes": {"type": "string"}}}
        if self.shadow.belief is None:
            self.shadow._start(obs)
        conv = self.shadow.belief.marginal(lambda v: v[1].signal)
        exact_play = [float(sum(p for (hand, _), p in self.shadow.belief.items()
                                if hand[i] is not None and obs.fireworks[hand[i][0]] == hand[i][1]))
                      for i in range(r.hand)]
        prompt = self._prompt(obs, names)
        if self.give_beliefs:
            prompt += ("\n\nAn exact inference engine, given everything PAR has done, computes:\n"
                       "  PAR's signal: " + ", ".join(f"{k} {conv.prob(k):.2f}" for k in SIGNALS) + "\n"
                       "  probability that your card is playable now: "
                       + ", ".join(f"slot {i} {p:.2f}" for i, p in enumerate(exact_play)
                                   if obs.hands[self.me][i] is not None)
                       + "\nThese are correct; you may rely on them.")
        reply, meta = self.cli.ask(self.system, prompt, schema)
        choice = reply.get("choice") if isinstance(reply, dict) else None
        fallback = choice not in names
        if fallback:        # the call failed: discard the oldest card if allowed, else play it
            action, reply = (r.discard(0) if r.discard(0) in legal else legal[0]), {}
        else:
            action = legal[names.index(choice)]
            self.notes = str(reply.get("notes", ""))[:400]
        sp = reply.get("slot_playable") or {}
        rec = {"choice": self._name(action), "fallback": fallback,
               "signal": _normalise(reply.get("partner_signal"), SIGNALS),
               "exact_signal": {k: conv.prob(k) for k in SIGNALS},
               "slot_playable": [min(max(float(sp.get(k, 0.0)), 0.0), 1.0) if sp else None for k in slots],
               "exact_slot_playable": exact_play, "reasoning": reply.get("reasoning", ""),
               "notes": reply.get("notes", ""), "prompt": prompt, **meta}
        self.records.append(rec)
        if self.verbose:
            print(f"    {rec['choice']:14s} signal {rec['signal']}  {meta['latency_s']:.1f}s"
                  + ("  [fallback]" if fallback else ""), flush=True)
        return int(action)
