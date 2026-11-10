"""Hanabi players for the convention study.

* ``ConventionAgent`` follows one fixed convention (``hanabi_partner.convention_move``).  It is
  both the scripted partner and the baseline for the other seat.
* ``PPHanabiAgent`` reasons with probabilistic programs.  Its belief is the exact joint
  distribution over its own hidden hand and the partner's convention, updated after every move
  by conditioning the partner program on what the partner did.  It chooses a move by imagining
  the rest of the game in worlds drawn from that belief, with the partner program playing the
  partner, and taking the move with the best expected final score.

Agents see the game only through ``hanabi_partner.observe``: their own cards and the deck are
face down.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from partner_modeling.hanabi_model import Card, HState, Rules, legal_moves, step
from partner_modeling.hanabi_partner import CONVENTIONS, EPS, HIDDEN, Convention, convention_move, observe
from partner_modeling.ppl import Dist, condition, factor, infer, observe as observe_draw, sample, simulate

Hand = Tuple[Optional[Card], ...]


class HanabiAgent:
    name = "agent"

    def reset(self, rules: Rules, me: int, seed: int):
        self.rules, self.me, self.rng = rules, me, np.random.default_rng([seed, me])

    def act(self, obs: HState) -> int:
        raise NotImplementedError

    def saw(self, before: HState, actor: int, action: int, revealed: Optional[Card],
            hit: Optional[Tuple[int, ...]], after: HState):
        """A move was made.  ``before`` / ``after`` are this agent's observations around it;
        ``revealed`` is the card that was played or discarded and ``hit`` the slots a hint
        touched (both are public the moment the move is made)."""


class ConventionAgent(HanabiAgent):
    def __init__(self, conv: Convention, eps: float = EPS):
        self.conv, self.eps = conv, eps
        self.name = f"{conv.signal}/{conv.chop}"

    def act(self, obs: HState) -> int:
        return simulate(convention_move, self.rules, obs, self.me, self.conv, self.eps, rng=self.rng)


# ---------------------------------------------------------------------------
# Belief programs
# ---------------------------------------------------------------------------

def unseen(rules: Rules, obs: HState) -> Counter:
    """Copies of each card that the observer cannot see: they are in its own hand or in the deck."""
    left = Counter(rules.all_cards())
    left.subtract(c for hand in obs.hands for c in hand if c is not None and c != HIDDEN)
    left.subtract(obs.discards)
    left.subtract((c, r) for c in range(rules.colors) for r in range(obs.fireworks[c]))
    return left


def in_deck(pool: Counter, hand: Hand) -> Dist:
    """The next card off the deck, if my hand is ``hand``."""
    left = pool.copy()
    left.subtract(c for c in hand if c is not None)
    return Dist(list(left), [max(n, 0) for n in left.values()])


def deal(rules: Rules, obs: HState) -> Hand:
    """My opening hand: cards drawn one by one from everything I cannot see."""
    left = unseen(rules, obs)
    hand: List[Card] = []
    for _ in range(rules.hand):
        card = sample(Dist(list(left), [max(n, 0) for n in left.values()]))
        left[card] -= 1
        hand.append(card)
    return tuple(hand)


def _put(obs: HState, me: int, hand: Hand) -> HState:
    """The game as the partner sees it, if my hand is ``hand``."""
    partner = 1 - me
    hands = [None, None]
    hands[me] = hand
    hands[partner] = tuple(HIDDEN if c is not None else None for c in obs.hands[partner])
    return obs._replace(hands=tuple(hands))


class PartnerProgram:
    """A partner move program with a cache of its enumerated move distributions."""

    def __init__(self, program=convention_move):
        self.program = program
        self._cache: Dict[Any, Dist] = {}

    def moves(self, rules: Rules, s: HState, who: int, conv: Convention) -> Dist:
        key = (s, who, conv)
        d = self._cache.get(key)
        if d is None:
            if len(self._cache) > 300_000:
                self._cache.clear()
            d = self._cache[key] = infer(self.program, rules, s, who, conv)
        return d


def after_partner_move(model: PartnerProgram, rules: Rules, belief: Dist, before: HState, me: int, action: int,
                       hit: Optional[Tuple[int, ...]], drew: Optional[Card], pool: Counter):
    """Reasoning about a partner move: which of my possible hands, under which convention, would
    have led it to do that?  ``hit`` are my slots a hint touched; ``drew`` is its replacement
    card; ``pool`` is everything I could not see before the move (``unseen``)."""
    hand, conv = sample(belief)
    observe_draw(model.moves(rules, _put(before, me, hand), 1 - me, conv), action)
    if hit is not None:                                 # the hint itself says which of my cards match
        kind, k = rules.kind(action)
        axis = 0 if kind == "color" else 1
        condition(tuple(i for i, c in enumerate(hand) if c is not None and c[axis] == k) == hit)
    if drew is not None:                                # the card it drew was in the deck, so not in my hand
        observe_draw(in_deck(pool, hand), drew)
    return hand, conv


def after_my_move(belief: Dist, slot: int, revealed: Card, draws: bool, pool: Counter):
    """I played or discarded ``slot`` and saw the card; a new, unknown card takes the last slot."""
    hand, conv = sample(belief)
    condition(hand[slot] == revealed)
    new = sample(in_deck(pool, hand)) if draws else None
    return hand[:slot] + hand[slot + 1:] + (new,), conv


def choose(moves: Sequence[int], value: Dict[int, float], alpha: float) -> int:
    """Planning as inference: moves are weighted by exp(alpha * expected score)."""
    m = sample(Dist.uniform(moves))
    factor(alpha * value[m])
    return m


# ---------------------------------------------------------------------------
# The probabilistic-program player
# ---------------------------------------------------------------------------

class PPHanabiAgent(HanabiAgent):
    """Exact belief over (my hand, partner's convention); expected-score planning by rollouts.

    ``conventions``: the conventions it considers.  ``worlds``: imagined games per decision.
    ``program``: the partner program (default: the reference ``convention_move``).
    """

    name = "pp"

    def __init__(self, conventions: Sequence[Convention] = CONVENTIONS, worlds: int = 40, program=convention_move,
                 alpha: float = 20.0, prune: float = 1e-7):
        self.conventions, self.worlds, self.alpha, self.prune = tuple(conventions), worlds, alpha, prune
        self.model = PartnerProgram(program)

    def reset(self, rules: Rules, me: int, seed: int):
        super().reset(rules, me, seed)
        self.belief: Optional[Dist] = None
        self.unexplained = 0
        self.trace: List[Dict[str, Any]] = []

    def _start(self, obs: HState):
        hands = infer(deal, self.rules, obs)
        self.belief = Dist([(h, c) for h in hands.values for c in self.conventions],
                           [p for p in hands.probs for _ in self.conventions])

    def _set(self, belief: Dist):
        keep = belief.probs >= self.prune
        self.belief = Dist([v for v, k in zip(belief.values, keep) if k], belief.probs[keep])

    def saw(self, before: HState, actor: int, action: int, revealed: Optional[Card],
            hit: Optional[Tuple[int, ...]], after: HState):
        if self.belief is None:
            self._start(before)
        rules, me = self.rules, self.me
        kind, k = rules.kind(action)
        pool = unseen(rules, before)
        try:
            if actor == me:
                if kind in ("discard", "play"):
                    self._set(infer(after_my_move, self.belief, k, revealed, len(before.deck) > 0, pool))
            else:
                drew = after.hands[actor][-1] if kind in ("discard", "play") else None
                self._set(infer(after_partner_move, self.model, rules, self.belief, before, me, action, hit, drew,
                                pool))
        except ValueError:      # the model gave what happened probability zero under every hypothesis
            self.unexplained += 1
            hands = self.belief.marginal(lambda v: v[0])
            self.belief = Dist([(h, c) for h in hands.values for c in self.conventions],
                               [p for p in hands.probs for _ in self.conventions])

    # ------------------------------------------------------------ planning
    def _world(self, obs: HState, rng: np.random.Generator) -> Tuple[HState, Convention]:
        """One fully specified game consistent with what I see and believe."""
        hand, conv = self.belief.draw(rng)
        left = unseen(self.rules, obs)
        left.subtract(c for c in hand if c is not None)
        deck = [c for c, n in left.items() for _ in range(max(n, 0))]
        rng.shuffle(deck)
        hands = list(obs.hands)
        hands[self.me] = hand
        return obs._replace(hands=tuple(hands), deck=tuple(tuple(c) for c in deck)), conv

    def _rollout(self, s: HState, conv: Convention, move: int, rng: np.random.Generator) -> int:
        """Final score change if I make ``move`` and both of us then follow ``conv``, as the
        partner program describes it."""
        s, total = step(self.rules, s, move)
        while not s.done:
            p = s.cur
            a = simulate(self.model.program, self.rules, observe(s, p), p, conv, EPS if p != self.me else 0.0,
                         rng=rng)
            s, r = step(self.rules, s, a)
            total += r
        return total

    def act(self, obs: HState) -> int:
        if self.belief is None:
            self._start(obs)
        moves = legal_moves(self.rules, obs)
        seeds = self.rng.integers(0, 2 ** 31, size=self.worlds)
        worlds = [self._world(obs, np.random.default_rng(int(sd))) for sd in seeds]
        # every move is imagined in the same worlds with the same random numbers
        value = {m: float(np.mean([self._rollout(s, conv, m, np.random.default_rng(int(sd)))
                                   for (s, conv), sd in zip(worlds, seeds)])) for m in moves}
        best = infer(choose, moves, value, self.alpha).argmax()
        self.trace.append({"move": int(best),
                           "value": {int(m): round(v, 2) for m, v in value.items()},
                           "convention": {c: float(p) for c, p in self.belief.marginal(lambda v: v[1]).items()},
                           "slots": [dict(self.belief.marginal(lambda v, i=i: v[0][i]).items())
                                     for i in range(self.rules.hand)]})
        return int(best)
