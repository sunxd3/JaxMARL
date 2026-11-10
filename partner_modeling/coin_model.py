"""Coin Game with a partner whose hidden type is how it treats the other player's coins.

JaxMARL's ``CoinGame``: two players, red and blue, on a 3x3 grid that wraps around, and one red
and one blue coin.  Moving onto a coin takes it: +1 for the taker, and -2 for the other player
if the coin was the other player's colour.  A taken coin reappears on a random cell.  The focal
agent is red and plays for its own score.

The partner (blue) is one of four strategies, which is what the focal agent has to work out:

* ``cooperator`` -- only ever goes for blue coins, stepping around red ones;
* ``greedy``     -- goes for the nearest coin, whatever its colour;
* ``tit_for_tat`` -- like the cooperator, but takes one red coin for each blue coin red has stolen;
* ``grim``       -- like the cooperator until red steals a blue coin once, and greedy ever after.

(A coin is *stolen* when a player takes the other colour's coin without taking its own in the
same move; when the two coins share a cell, taking them is held against no one.)

Three of the four behave identically until red takes a blue coin, and how they then react
differs: probing tells them apart, at a price that depends on which one it is.  Unlike the
Overcooked and Hanabi partners, this one *reacts* to the focal agent.

``CState`` is the game as a plain-Python model with the same moves and payoffs as the
environment (checked in ``tests/partner_modeling/test_coin.py``).  ``partner_move`` is the
partner as a probabilistic program.  What its reactive strategies remember (``debt``,
``grudge``) is a function of who took which coin, so it is public.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from partner_modeling.ppl import Dist, flip, sample

Cell = Tuple[int, int]
MOVES: Tuple[Cell, ...] = ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 0))      # right, left, up, down, stay
MOVE_NAMES = ("right", "left", "up", "down", "stay")
STAY = 4
CELLS: Tuple[Cell, ...] = tuple((a, b) for a in range(3) for b in range(3))
ANY_CELL = Dist.uniform(CELLS)
TYPES = ("cooperator", "greedy", "tit_for_tat", "grim")
EPS = 0.1           # probability of a uniformly random move
RED, BLUE = 0, 1


class CState(NamedTuple):
    pos: Tuple[Cell, Cell]          # red, blue
    coins: Tuple[Cell, Cell]        # red coin, blue coin
    t: int
    score: Tuple[int, int]
    debt: int                       # blue coins red has stolen, minus red coins blue has stolen since (at least 0)
    grudge: bool                    # red has stolen a blue coin at some point
    steals: Tuple[int, int]         # coins stolen by red, by blue ("stolen": taken while not also taking one's own)


def moved(cell: Cell, action: int) -> Cell:
    return ((cell[0] + MOVES[action][0]) % 3, (cell[1] + MOVES[action][1]) % 3)


def distance(a: Cell, b: Cell) -> int:
    """Steps between two cells on the wrapping grid."""
    return sum(min((x - y) % 3, (y - x) % 3) for x, y in zip(a, b))


def approach(cell: Cell, target: Cell, avoid: Optional[Cell] = None) -> List[int]:
    """The moves that get ``cell`` closest to ``target`` without stepping onto ``avoid``
    (unless ``avoid`` is the target itself).  'stay' is one of the moves."""
    ok = [a for a in range(5) if avoid is None or avoid == target or moved(cell, a) != avoid]
    best = min(distance(moved(cell, a), target) for a in ok)
    return [a for a in ok if distance(moved(cell, a), target) == best]


def step(s: CState, a_red: int, a_blue: int, new_coins: Tuple[Cell, Cell]) -> Tuple[CState, Tuple[int, int]]:
    """Both players move at once.  ``new_coins`` are where the red and the blue coin reappear
    if they are taken.  Returns the next state and (red reward, blue reward)."""
    pos = (moved(s.pos[RED], a_red), moved(s.pos[BLUE], a_blue))
    rr, rb = pos[RED] == s.coins[RED], pos[RED] == s.coins[BLUE]        # red takes red / blue coin
    br, bb = pos[BLUE] == s.coins[RED], pos[BLUE] == s.coins[BLUE]      # blue takes red / blue coin
    r_red = int(rr) + int(rb) - 2 * int(br)
    r_blue = int(br) + int(bb) - 2 * int(rb)
    coins = (new_coins[RED] if (rr or br) else s.coins[RED], new_coins[BLUE] if (rb or bb) else s.coins[BLUE])
    # Taking the other's coin is held against a player only if it was not sitting on its own coin.
    red_stole, blue_stole = rb and not rr, br and not bb
    return CState(pos, coins, s.t + 1, (s.score[RED] + r_red, s.score[BLUE] + r_blue),
                  max(0, s.debt + int(red_stole) - int(blue_stole)), s.grudge or red_stole,
                  (s.steals[RED] + int(red_stole), s.steals[BLUE] + int(blue_stole))), (r_red, r_blue)


def partner_target(s: CState, kind: str) -> Tuple[Cell, bool]:
    """The coin a partner of type ``kind`` is after, and whether it is being friendly about it
    (a friendly partner steps around the red coin)."""
    own, other = s.coins[BLUE], s.coins[RED]
    nearest = own if distance(s.pos[BLUE], own) <= distance(s.pos[BLUE], other) else other
    if kind == "cooperator":
        return own, True
    if kind == "greedy":
        return nearest, False
    if kind == "tit_for_tat":
        return (other, False) if s.debt > 0 else (own, True)
    if kind == "grim":
        return (nearest, False) if s.grudge else (own, True)
    raise ValueError(kind)


def partner_move(s: CState, kind: str, eps: float = EPS) -> int:
    """The partner's move as a program: a random move now and then, otherwise a step toward its coin."""
    if flip(eps):
        return sample(Dist.uniform(range(5)))
    target, friendly = partner_target(s, kind)
    moves = approach(s.pos[BLUE], target, s.coins[RED] if friendly else None)
    return moves[0] if len(moves) == 1 else sample(Dist.uniform(moves))


# ---------------------------------------------------------------------------
# The real game: JaxMARL's environment
# ---------------------------------------------------------------------------

class CoinEnv:
    """JaxMARL's ``CoinGame`` as one episode of ``steps`` moves, played alongside the model."""

    def __init__(self, steps: int = 60):
        from jaxmarl.environments.coin_game.coin_game import CoinGame
        # one extra inner step: the environment zeroes the rewards of the step that ends an episode
        self.steps = steps
        self.env = CoinGame(num_inner_steps=steps + 1, num_outer_steps=1)

    @staticmethod
    def _cell(x) -> Cell:
        x = np.asarray(x)
        return (int(x[0]), int(x[1]))

    def read(self, state) -> Tuple[Tuple[Cell, Cell], Tuple[Cell, Cell]]:
        return ((self._cell(state.red_pos), self._cell(state.blue_pos)),
                (self._cell(state.red_coin_pos), self._cell(state.blue_coin_pos)))

    def reset(self, key):
        _, state = self.env.reset(key)
        pos, coins = self.read(state)
        return state, CState(pos, coins, 0, (0, 0), 0, False, (0, 0))

    def step(self, key, state, a_red: int, a_blue: int):
        """(state, (red reward, blue reward)) in the real environment."""
        _, state, rew, _, _ = self.env.step(key, state, {self.env.agents[0]: a_red, self.env.agents[1]: a_blue})
        return state, (int(rew[self.env.agents[0]]), int(rew[self.env.agents[1]]))
