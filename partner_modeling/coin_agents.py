"""Coin Game players (the focal agent is red).

* ``ScriptedRed``: a fixed policy -- go for red coins only (``"own"``), go for the nearest coin
  (``"greedy"``), or the best fixed policy for a known partner type (``"oracle"``).
* ``PPCoinAgent``: exact belief over the partner's type from its moves; expected-score planning
  by imagining the rest of the episode with the partner program playing the partner.  One of
  its options is to *probe*: take one blue coin to see how the partner reacts, then do whatever
  is best against the type that reveals.  Whether probing is worth it against the risk of a
  grudge is settled by the rollouts.
* ``ClaudeCoinAgent``: asks ``claude -p`` which coin to go for.

All of them see the whole board: nothing is hidden in this game except the partner's type.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from partner_modeling.coin_model import (
    ANY_CELL, BLUE, EPS, RED, STAY, TYPES, CState, approach, distance, partner_move, step,
)
from partner_modeling.ppl import Dist, factor, infer, observe, sample, simulate

# Best fixed red policy against each partner type, for a red that plays for its own score.
BEST_RESPONSE = {"cooperator": "greedy", "greedy": "greedy", "tit_for_tat": "own", "grim": "own"}


def red_move(s: CState, policy: str) -> int:
    """Red's move under a policy, as a program (ties between equally good moves are random).
    ``"own"``: go for the red coin and step around the blue one; ``"steal"``: go for the blue
    coin; ``"greedy"``: go for whichever is nearer."""
    own, other = s.coins[RED], s.coins[BLUE]
    if policy == "own":
        moves = approach(s.pos[RED], own, other)
    elif policy == "steal":
        moves = approach(s.pos[RED], other)
    else:
        moves = approach(s.pos[RED], own if distance(s.pos[RED], own) <= distance(s.pos[RED], other) else other)
    return moves[0] if len(moves) == 1 else sample(Dist.uniform(moves))


class CoinAgent:
    name = "agent"

    def reset(self, seed: int, steps: int):
        self.rng, self.steps = np.random.default_rng([seed, 7]), steps

    def act(self, s: CState) -> int:
        raise NotImplementedError

    def saw(self, before: CState, a_red: int, a_blue: int, after: CState):
        pass


class ScriptedRed(CoinAgent):
    def __init__(self, policy: str, partner: Optional[str] = None):
        self.policy = BEST_RESPONSE[partner] if policy == "oracle" else policy
        self.name = policy

    def act(self, s: CState) -> int:
        return simulate(red_move, s, self.policy, rng=self.rng)


# ---------------------------------------------------------------------------
# Probabilistic programs
# ---------------------------------------------------------------------------

def belief_step(program: Callable, belief: Dist, before: CState, a_blue: int) -> str:
    """Which partner type would have made that move?"""
    kind = sample(belief)
    observe(infer(program, before, kind), a_blue)
    return kind


def episode_return(program: Callable, s: CState, option: str, kind: str, steps: int) -> int:
    """Red's score from here to the end of the episode if it follows ``option`` against a partner
    of type ``kind``.  ``"probe"`` takes one blue coin and then plays the best response to
    ``kind``, which by then the partner's reaction will have shown."""
    total, taken = 0, s.steals[RED]
    while s.t < steps:
        policy = option
        if option == "probe":
            policy = "steal" if s.steals[RED] == taken else BEST_RESPONSE[kind]
        s, (r, _) = step(s, red_move(s, policy), program(s, kind), (sample(ANY_CELL), sample(ANY_CELL)))
        total += r
    return total


def choose(options: Sequence[str], value: Dict[str, float], alpha: float) -> str:
    o = sample(Dist.uniform(options))
    factor(alpha * value[o])
    return o


class PPCoinAgent(CoinAgent):
    """``program(s, kind)``: the partner program used for belief and for imagination."""

    name = "pp"
    OPTIONS = ("own", "greedy", "probe")

    def __init__(self, program: Callable = partner_move, types: Sequence[str] = TYPES, rollouts: int = 24,
                 alpha: float = 20.0, margin: float = 0.3):
        """``types``: the partner strategies the agent considers possible (its prior is uniform)."""
        self.program, self.types = program, tuple(types)
        self.rollouts, self.alpha, self.margin = rollouts, alpha, margin

    def reset(self, seed: int, steps: int):
        super().reset(seed, steps)
        self.belief = Dist.uniform(self.types)
        self.option, self.probe_from = "own", None
        self.unexplained = 0
        self.trace: List[Dict[str, Any]] = []

    def saw(self, before: CState, a_red: int, a_blue: int, after: CState):
        try:
            self.belief = infer(belief_step, self.program, self.belief, before, a_blue)
        except ValueError:
            self.unexplained += 1

    def act(self, s: CState) -> int:
        if self.option == "probe" and s.steals[RED] > self.probe_from:
            self.option = "own"                              # the probe is done: decide afresh
        seeds = self.rng.integers(0, 2 ** 31, size=self.rollouts)
        value = {}
        for o in self.OPTIONS:            # the same imagined partners and coin positions for every option
            value[o] = sum(p * np.mean([simulate(episode_return, self.program, s, o, kind, self.steps,
                                                 rng=np.random.default_rng([int(sd), i])) for sd in seeds])
                           for i, (kind, p) in enumerate(self.belief.items()) if p > 0.01)
        best = infer(choose, self.OPTIONS, value, self.alpha).argmax()
        if value[best] > value[self.option] + self.margin:
            self.option = best
            self.probe_from = s.steals[RED] if best == "probe" else None
        policy = "steal" if self.option == "probe" else self.option
        self.trace.append({"t": s.t, "option": self.option, "belief": dict(self.belief.items()),
                           "value": {o: round(float(v), 2) for o, v in value.items()}})
        return simulate(red_move, s, policy, rng=self.rng)


# ---------------------------------------------------------------------------
# LLM direct
# ---------------------------------------------------------------------------

SYSTEM = """\
You play the Coin Game as RED against a scripted partner, BLUE. You play for your own score.

The board is a 3x3 grid that wraps around (moving off one edge brings you back on the opposite \
side), so everything is at most 2 steps away. There is always one red coin and one blue coin. \
Both players move one step at a time, at the same moment. Moving onto a coin takes it: the taker \
scores +1, and if the coin is the other player's colour, the other player also loses 2. A taken \
coin reappears on a random cell. The episode lasts {steps} steps.

BLUE follows one of these fixed strategies, each equally likely; you are not told which:
{strategies}
A coin counts as stolen when a player takes the other colour's coin without taking its own in the \
same move; when the two coins share a cell, taking them is held against no one (the 2 points are \
still lost). With probability {eps} BLUE makes a random move instead.

You do not choose individual moves. You choose what to go for: the red coin, the blue coin, or \
stay where you are; a controller takes the shortest way, and when you go for the red coin it \
steps around the blue one. You are asked again after a coin has been taken (at most every 2 \
steps) and at least every 4 steps.

Reply with:
- reasoning: at most 50 words.
- partner_type: your probability for each strategy ({names}), summing to 1.
- choice: red_coin, blue_coin or stay.
- notes: at most 40 words. This is your only memory; it is shown to you at your next decision."""

CHOICES = {"red_coin": "own", "blue_coin": "steal", "stay": None}
STRATEGY_TEXT = {
    "cooperator": "only ever goes for blue coins, stepping around red ones.",
    "greedy": "goes for the nearest coin, whatever its colour.",
    "tit_for_tat": "like the cooperator, but takes one red coin for each blue coin you have stolen.",
    "grim": "like the cooperator until you steal a blue coin once, and greedy for the rest of the episode.",
}


class ClaudeCoinAgent(CoinAgent):
    name = "claude"

    def __init__(self, model: Optional[str] = None, effort: Optional[str] = "low", give_beliefs: bool = False,
                 types: Sequence[str] = TYPES, timeout: float = 180.0):
        from partner_modeling.llm_focal import ClaudeCLI
        self.cli = ClaudeCLI(model=model, effort=effort, timeout=timeout)
        self.give_beliefs, self.types = give_beliefs, tuple(types)

    def reset(self, seed: int, steps: int):
        super().reset(seed, steps)
        self.system = SYSTEM.format(steps=steps, eps=EPS, names=" / ".join(self.types),
                                    strategies="\n".join(f"- {k}: {STRATEGY_TEXT[k]}" for k in self.types))
        self.exact = Dist.uniform(self.types)   # the exact posterior, kept alongside for comparison
        self.events: List[str] = []
        self.notes, self.policy, self.t_asked, self.changed = "", "own", -10, True
        self.records: List[Dict[str, Any]] = []

    def saw(self, before: CState, a_red: int, a_blue: int, after: CState):
        self.exact = infer(belief_step, partner_move, self.exact, before, a_blue)
        for who, mover in (("YOU", RED), ("BLUE", BLUE)):
            for colour, coin in (("red", RED), ("blue", BLUE)):
                if after.pos[mover] == before.coins[coin]:
                    cost = ""
                    if mover != coin:
                        cost = " (costing BLUE 2)" if mover == RED else " (costing you 2)"
                        if before.coins[RED] == before.coins[BLUE]:
                            cost += ", along with its own coin on the same cell"
                    self.events.append(f"step {before.t}: {who} took the {colour} coin{cost}")
                    self.changed = True

    def act(self, s: CState) -> int:
        from partner_modeling.llm_focal import _normalise
        since = s.t - self.t_asked
        if since >= 4 or (self.changed and since >= 2):
            grid = [[". " for _ in range(3)] for _ in range(3)]
            marks = (("R", s.pos[RED]), ("B", s.pos[BLUE]), ("r", s.coins[RED]), ("b", s.coins[BLUE]))
            for m, (a, b) in marks:
                grid[a][b] = (grid[a][b].strip(". ") + m).ljust(2)
            prompt = "\n".join([
                f"Step {s.t} of {self.steps}. Your score {s.score[RED]}, BLUE's score {s.score[BLUE]}.",
                "Board (R = you, B = BLUE, r = red coin, b = blue coin; rows and columns wrap around):",
                *("  " + " ".join(row) for row in grid),
                f"You are {distance(s.pos[RED], s.coins[RED])} steps from the red coin and "
                f"{distance(s.pos[RED], s.coins[BLUE])} from the blue coin; BLUE is "
                f"{distance(s.pos[BLUE], s.coins[RED])} from the red coin and "
                f"{distance(s.pos[BLUE], s.coins[BLUE])} from the blue coin.",
                f"So far you have stolen {s.steals[RED]} blue coins and BLUE has stolen {s.steals[BLUE]} red coins.",
                "", "Coins taken (most recent last):", *([f"  {e}" for e in self.events[-12:]] or ["  none yet"]),
                "", f'Your notes from your last decision: "{self.notes}"' if self.notes else "You have no notes yet."])
            if self.give_beliefs:
                prompt += ("\n\nAn exact inference engine, given every move BLUE has made, computes: "
                           + ", ".join(f"{k} {self.exact.prob(k):.2f}" for k in self.types)
                           + ". These are correct; you may rely on them.")
            schema = {"type": "object", "additionalProperties": False,
                      "required": ["reasoning", "partner_type", "choice", "notes"],
                      "properties": {"reasoning": {"type": "string"},
                                     "partner_type": {"type": "object", "additionalProperties": False,
                                                      "required": list(self.types),
                                                      "properties": {k: {"type": "number"} for k in self.types}},
                                     "choice": {"type": "string", "enum": list(CHOICES)},
                                     "notes": {"type": "string"}}}
            reply, meta = self.cli.ask(self.system, prompt, schema)
            ok = isinstance(reply, dict) and reply.get("choice") in CHOICES
            if ok:
                self.policy = CHOICES[reply["choice"]]
                self.notes = str(reply.get("notes", ""))[:400]
            self.records.append({"t": s.t, "choice": reply.get("choice") if ok else None, "fallback": not ok,
                                 "belief": _normalise(reply.get("partner_type"), self.types) if ok else None,
                                 "exact": {k: self.exact.prob(k) for k in self.types},
                                 "reasoning": reply.get("reasoning", "") if ok else "", "prompt": prompt, **meta})
            self.t_asked, self.changed = s.t, False
        return STAY if self.policy is None else simulate(red_move, s, self.policy, rng=self.rng)
