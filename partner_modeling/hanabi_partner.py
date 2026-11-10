"""A scripted Hanabi partner whose hidden type is its *convention*, as a probabilistic program.

The partner never sees its own cards.  What it does with a hint, and which hints it gives,
depends on a convention the other player does not know:

* ``signal`` -- what a hint means.
    ``"newest"``: play the newest card the hint touched.
    ``"oldest"``: play the oldest card the hint touched.
    ``"literal"``: nothing beyond what it says; play a card only once the hints pin it down.
* ``chop`` -- which unhinted card it discards when it has nothing better to do: its oldest or
  its newest.

The same hint therefore tells a "newest" partner and an "oldest" partner to play different
cards, and a "literal" partner to play none.  A player who has not worked out the convention
will misread the partner's hints about its own hidden hand and give hints the partner misreads.

``convention_move`` is the partner's move as a program: enumerating it gives the exact
probability of each move, simulating it plays.  It reads only what the partner can see (the
other player's cards, and public facts about its own), so the same program is also a policy for
the other seat.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, NamedTuple, Tuple

from partner_modeling.hanabi_model import Card, HState, Rules, legal_moves, touched_slots
from partner_modeling.ppl import Dist, flip, sample

EPS = 0.05          # probability of a uniformly random legal move
HIDDEN: Card = (-1, -1)


class Convention(NamedTuple):
    signal: str     # "newest" | "oldest" | "literal"
    chop: str       # "oldest" | "newest"


CONVENTIONS = tuple(Convention(s, c) for s in ("newest", "oldest", "literal") for c in ("oldest", "newest"))


def observe(s: HState, who: int) -> HState:
    """The game as player ``who`` sees it: its own cards and the deck are face down."""
    hands = tuple(tuple(HIDDEN if (p == who and c is not None) else c for c in hand)
                  for p, hand in enumerate(s.hands))
    return s._replace(hands=hands, deck=(HIDDEN,) * len(s.deck))


def _identities(rules: Rules, mask: int) -> List[Card]:
    return [(c, r) for c in range(rules.colors) for r in range(rules.ranks) if mask & rules.bit((c, r))]


def _certain(rules: Rules, s: HState, mask: int) -> bool:
    """The hints leave only identities that can be played right now."""
    ids = _identities(rules, mask)
    return bool(ids) and all(s.fireworks[c] == r for c, r in ids)


def _possible(rules: Rules, s: HState, mask: int) -> bool:
    return any(s.fireworks[c] == r for c, r in _identities(rules, mask))


def play_slots(rules: Rules, s: HState, who: int, conv: Convention) -> Tuple[int, ...]:
    """Cards ``who`` believes it should play: pinned down by hints, or marked by its convention."""
    marks = s.newest[who] if conv.signal == "newest" else s.oldest[who] if conv.signal == "oldest" else None
    return tuple(i for i, c in enumerate(s.hands[who]) if c is not None and (
        _certain(rules, s, s.masks[who][i])
        or (marks is not None and marks[i] and _possible(rules, s, s.masks[who][i]))))


def _hint_mask(rules: Rules, action: int) -> int:
    kind, k = rules.kind(action)
    axis = 0 if kind == "color" else 1
    return sum(rules.bit((c, r)) for c in range(rules.colors) for r in range(rules.ranks) if (c, r)[axis] == k)


@lru_cache(maxsize=400_000)
def analyse(rules: Rules, s: HState, who: int, conv: Convention):
    """Everything deterministic about ``who``'s move: (cards to play, useful hints, card to discard,
    hints to fall back on).  ``s`` is the game as ``who`` sees it."""
    other = 1 - who
    legal = legal_moves(rules, s)
    plays = play_slots(rules, s, who, conv)
    hints = [a for a in legal if rules.kind(a)[0] in ("color", "rank")]
    expected = play_slots(rules, s, other, conv)       # what the other will play anyway, if it shares my convention
    useful, filler = [], []
    for a in hints:
        hit = touched_slots(rules, s, a)
        if conv.signal == "literal":
            allowed = _hint_mask(rules, a)
            after = [(s.masks[other][i] & allowed) if i in hit else (s.masks[other][i] & ~allowed)
                     for i in range(rules.hand)]
            pins = any(s.hands[other][i] is not None and i not in expected and _certain(rules, s, after[i])
                       for i in range(rules.hand))
            narrows = any(s.playable(s.hands[other][i]) and after[i] != s.masks[other][i] for i in hit)
            (useful if pins else filler if narrows else []).append(a)
        else:
            focus = hit[-1] if conv.signal == "newest" else hit[0]
            if s.playable(s.hands[other][focus]):
                (filler if focus in expected else useful).append(a)
    if conv.signal == "literal" and not useful:        # no hint pins a card down yet: narrow one that is playable
        useful, filler = filler, []
    cards = [i for i, c in enumerate(s.hands[who]) if c is not None]
    unhinted = [i for i in cards if not s.touched[who][i]] or cards
    chop = (unhinted[0] if conv.chop == "oldest" else unhinted[-1]) if unhinted else None
    return plays, tuple(useful), chop, tuple(filler or hints), tuple(legal)


def convention_move(rules: Rules, s: HState, who: int, conv: Convention, eps: float = EPS) -> int:
    """The move of a player following ``conv``, given the game as it sees it (``s.cur == who``)."""
    plays, useful, chop, filler, legal = analyse(rules, s, who, conv)
    if flip(eps):
        return sample(Dist.uniform(legal))
    if plays:                                           # 1. play what I believe is playable
        return rules.play(plays[0] if len(plays) == 1 else sample(Dist.uniform(plays)))
    if s.info > 0 and useful:                           # 2. tell the other about a playable card
        return useful[0] if len(useful) == 1 else sample(Dist.uniform(useful))
    if s.info < rules.max_info and chop is not None:    # 3. discard to win a hint token back
        return rules.discard(chop)
    if s.info > 0 and filler:                           # 4. tokens are full: spend one
        return filler[0] if len(filler) == 1 else sample(Dist.uniform(filler))
    return legal[0] if len(legal) == 1 else sample(Dist.uniform(legal))
