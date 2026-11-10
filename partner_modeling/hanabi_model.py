"""A small two-player Hanabi as a plain-Python game model, matched to JaxMARL's ``HanabiEnv``.

The real game is played in JaxMARL's environment (``MiniHanabi``).  Agents reason on ``HState``,
an immutable copy of the game with the same rules (``step``), which is cheap to copy, to hash and
to run forward in imagination.  ``tests/partner_modeling/test_hanabi.py`` steps the two in
lockstep and checks they agree.

Besides the rules, ``HState`` keeps three public facts about every card in hand that
conventions are built from: whether a hint has touched it, and whether it was the newest /
the oldest card touched by some hint.

Cards are ``(color, rank)`` with both counted from 0; a hand slot is ``None`` when the deck has
run out.  New cards enter at the last slot, so slot 0 is the oldest card.  Actions use JaxMARL's
numbering: discard slot i, play slot i, hint color c, hint rank r.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

Card = Tuple[int, int]


class Rules(NamedTuple):
    colors: int = 3
    ranks: int = 5
    hand: int = 3
    copies: Tuple[int, ...] = (3, 2, 2, 2, 1)      # copies of each rank, per color
    max_info: int = 8
    max_lives: int = 3

    # ---- action numbering (two players, so every hint goes to the other player)
    def discard(self, slot: int) -> int:
        return slot

    def play(self, slot: int) -> int:
        return self.hand + slot

    def hint_color(self, color: int) -> int:
        return 2 * self.hand + color

    def hint_rank(self, rank: int) -> int:
        return 2 * self.hand + self.colors + rank

    def kind(self, action: int) -> Tuple[str, int]:
        """('discard' | 'play' | 'color' | 'rank', index)."""
        h, c = self.hand, self.colors
        if action < h:
            return "discard", action
        if action < 2 * h:
            return "play", action - h
        if action < 2 * h + c:
            return "color", action - 2 * h
        return "rank", action - 2 * h - c

    def describe(self, action: int) -> str:
        k, i = self.kind(action)
        return {"discard": f"discard slot {i}", "play": f"play slot {i}",
                "color": f"hint {COLOR_NAMES[i]}", "rank": f"hint rank {i + 1}"}[k]

    @property
    def num_actions(self) -> int:
        return 2 * self.hand + self.colors + self.ranks

    def all_cards(self) -> List[Card]:
        return [(c, r) for c in range(self.colors) for r in range(self.ranks) for _ in range(self.copies[r])]

    def bit(self, card: Card) -> int:
        return 1 << (card[0] * self.ranks + card[1])

    @property
    def full_mask(self) -> int:
        return (1 << (self.colors * self.ranks)) - 1


COLOR_NAMES = ("red", "yellow", "green", "white", "blue")


class HState(NamedTuple):
    hands: Tuple[Tuple[Optional[Card], ...], ...]   # per player, slot 0 = oldest
    deck: Tuple[Card, ...]                          # top first; no player sees it
    fireworks: Tuple[int, ...]                      # cards played so far, per color
    discards: Tuple[Card, ...]
    info: int
    lives: int
    cur: int                                        # player to move
    masks: Tuple[Tuple[int, ...], ...]              # per card: bitmask of identities the hints still allow
    touched: Tuple[Tuple[bool, ...], ...]           # per card: some hint has touched it
    newest: Tuple[Tuple[bool, ...], ...]            # per card: it was the newest card touched by some hint
    oldest: Tuple[Tuple[bool, ...], ...]            # per card: it was the oldest card touched by some hint
    last_round: int                                 # moves made since the deck ran out, plus one
    done: bool
    score: int

    def playable(self, card: Optional[Card]) -> bool:
        return card is not None and self.fireworks[card[0]] == card[1]


def initial_state(rules: Rules, deck: Sequence[Card]) -> HState:
    deck = tuple(deck)
    h = rules.hand
    hands = (deck[:h], deck[h:2 * h])
    flags = ((False,) * h,) * 2
    return HState(hands=hands, deck=deck[2 * h:], fireworks=(0,) * rules.colors, discards=(), info=rules.max_info,
                  lives=rules.max_lives, cur=0, masks=((rules.full_mask,) * h,) * 2, touched=flags, newest=flags,
                  oldest=flags, last_round=0, done=False, score=0)


def legal_moves(rules: Rules, s: HState) -> List[int]:
    me, other = s.cur, 1 - s.cur
    out = []
    if s.info < rules.max_info:
        out += [rules.discard(i) for i, c in enumerate(s.hands[me]) if c is not None]
    out += [rules.play(i) for i, c in enumerate(s.hands[me]) if c is not None]
    if s.info > 0:
        cards = [c for c in s.hands[other] if c is not None]
        out += [rules.hint_color(k) for k in range(rules.colors) if any(c[0] == k for c in cards)]
        out += [rules.hint_rank(k) for k in range(rules.ranks) if any(c[1] == k for c in cards)]
    return out


def touched_slots(rules: Rules, s: HState, action: int) -> List[int]:
    """Slots of the other player's hand that hint ``action`` touches."""
    kind, k = rules.kind(action)
    axis = 0 if kind == "color" else 1
    return [i for i, c in enumerate(s.hands[1 - s.cur]) if c is not None and c[axis] == k]


def _set(t: tuple, i: int, v) -> tuple:
    return t[:i] + (v,) + t[i + 1:]


def _drop_append(t: tuple, i: int, v) -> tuple:
    return t[:i] + t[i + 1:] + (v,)


def step(rules: Rules, s: HState, action: int) -> Tuple[HState, int]:
    """The move ``action`` by the player to move.  Returns the next state and the reward."""
    me, other = s.cur, 1 - s.cur
    kind, k = rules.kind(action)
    before = sum(s.fireworks)
    fireworks, discards, info, lives, deck = s.fireworks, s.discards, s.info, s.lives, s.deck
    hands, masks, touched, newest, oldest = s.hands, s.masks, s.touched, s.newest, s.oldest
    if kind in ("discard", "play"):
        card = s.hands[me][k]
        if kind == "discard":
            info = min(info + 1, rules.max_info)
            discards += (card,)
        elif fireworks[card[0]] == card[1]:
            fireworks = _set(fireworks, card[0], card[1] + 1)
            if card[1] == rules.ranks - 1:
                info = min(info + 1, rules.max_info)
        else:
            lives -= 1
            discards += (card,)
        new = deck[0] if deck else None
        deck = deck[1:]
        hands = _set(hands, me, _drop_append(hands[me], k, new))
        masks = _set(masks, me, _drop_append(masks[me], k, rules.full_mask if new is not None else 0))
        touched = _set(touched, me, _drop_append(touched[me], k, False))
        newest = _set(newest, me, _drop_append(newest[me], k, False))
        oldest = _set(oldest, me, _drop_append(oldest[me], k, False))
    else:
        info -= 1
        axis = 0 if kind == "color" else 1
        allowed = sum(rules.bit((c, r)) for c in range(rules.colors) for r in range(rules.ranks)
                      if (c, r)[axis] == k)
        hit = [i for i, c in enumerate(hands[other]) if c is not None and c[axis] == k]
        m = tuple((mk & allowed) if i in hit else (mk & ~allowed) for i, mk in enumerate(masks[other]))
        masks = _set(masks, other, m)
        touched = _set(touched, other, tuple(t or i in hit for i, t in enumerate(touched[other])))
        newest = _set(newest, other, tuple(t or i == hit[-1] for i, t in enumerate(newest[other])))
        oldest = _set(oldest, other, tuple(t or i == hit[0] for i, t in enumerate(oldest[other])))
    after = sum(fireworks)
    out_of_lives = lives == 0
    last_round = s.last_round + (1 if not deck else 0)
    done = out_of_lives or after == rules.colors * rules.ranks or last_round == 3
    reward = -after if out_of_lives else after - before       # losing the last life forfeits the score
    return HState(hands, deck, fireworks, discards, info, lives, other, masks, touched, newest, oldest,
                  last_round, done, s.score + reward), reward


# ---------------------------------------------------------------------------
# The real game: JaxMARL's environment
# ---------------------------------------------------------------------------

class MiniHanabi:
    """JaxMARL's ``HanabiEnv`` in the configuration of ``rules``, played alongside the model."""

    def __init__(self, rules: Rules = Rules()):
        from jaxmarl.environments.hanabi.hanabi import HanabiEnv
        self.rules = rules
        self.env = HanabiEnv(num_agents=2, num_colors=rules.colors, num_ranks=rules.ranks, hand_size=rules.hand,
                             max_info_tokens=rules.max_info, max_life_tokens=rules.max_lives,
                             num_cards_of_rank=np.array(rules.copies))

    def reset(self, key):
        """(env state, model state) for a freshly shuffled game."""
        _, state = self.env.reset(key)
        hands = self._hands(state)
        deck = tuple(self._card(c) for c in np.asarray(state.deck)[2 * self.rules.hand:])
        return state, initial_state(self.rules, hands[0] + hands[1] + deck)

    def step(self, key, state, action: int):
        """The current player's move in the real environment: (state, reward, done)."""
        _, state, rew, dones, _ = self.env.step_env(key, state, {a: action for a in self.env.agents})
        return state, int(rew["__all__"]), bool(dones["__all__"])

    @staticmethod
    def _card(onehot) -> Optional[Card]:
        idx = np.argwhere(np.asarray(onehot) > 0)
        return (int(idx[0][0]), int(idx[0][1])) if len(idx) else None

    def _hands(self, state):
        return tuple(tuple(self._card(c) for c in hand) for hand in np.asarray(state.player_hands))

    def public(self, state) -> tuple:
        """What the model must reproduce: hands, fireworks, tokens, hint masks, turn, score, end."""
        masks = tuple(tuple(int(sum(1 << i for i in np.flatnonzero(m > 0))) for m in player)
                      for player in np.asarray(state.card_knowledge))
        return (self._hands(state), tuple(int(v) for v in np.asarray(state.fireworks).sum(axis=1)),
                int(np.asarray(state.info_tokens).sum()), int(np.asarray(state.life_tokens).sum()),
                int(np.argmax(np.asarray(state.cur_player_idx))), masks, int(state.score),
                bool(state.terminal) or bool(state.out_of_lives))


def model_public(s: HState) -> tuple:
    """``HState`` in the same form as ``MiniHanabi.public``."""
    return (s.hands, s.fireworks, s.info, s.lives, s.cur, s.masks, s.score, s.done)
