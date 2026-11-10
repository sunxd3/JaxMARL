"""Coin Game: the model against JaxMARL's environment, and the typed partner.

    .venv/bin/python -m pytest tests/partner_modeling/test_coin.py -q
"""

import jax
import numpy as np
import pytest

from partner_modeling.coin_model import BLUE, RED, STAY, TYPES, CoinEnv, CState, moved, partner_move, step
from partner_modeling.ppl import infer


def test_model_matches_jaxmarl_environment():
    game = CoinEnv(steps=30)
    rng = np.random.default_rng(0)
    for g in range(8):
        state, s = game.reset(jax.random.PRNGKey(g))
        for t in range(game.steps):
            a, b = int(rng.integers(5)), int(rng.integers(5))
            state, rew = game.step(jax.random.PRNGKey(100 * g + t), state, a, b)
            pos, coins = game.read(state)
            s, r = step(s, a, b, coins)
            assert r == rew and (s.pos, s.coins) == (pos, coins)


def _state(red, blue, red_coin, blue_coin, debt=0, grudge=False):
    return CState((red, blue), (red_coin, blue_coin), 0, (0, 0), debt, grudge, (0, 0))


def test_partner_types_differ_where_they_should():
    # blue at (0,0); the red coin is one step right, the blue coin two steps away past it
    s = _state((2, 2), (0, 0), (0, 1), (1, 1))
    dist = {k: infer(partner_move, s, k) for k in TYPES}
    for d in dist.values():
        assert set(d.values) == set(range(5)) and d.probs.sum() == pytest.approx(1.0)
    takes_red = lambda d: sum(p for a, p in d.items() if moved(s.pos[BLUE], a) == s.coins[RED])
    assert takes_red(dist["greedy"]) > 0.9
    for k in ("cooperator", "tit_for_tat", "grim"):        # friendly play steps around the red coin
        assert takes_red(dist[k]) < 0.05
    # after red has stolen: tit-for-tat goes for the red coin, grim turns greedy, the cooperator does not change
    after = s._replace(debt=1, grudge=True)
    assert takes_red(infer(partner_move, after, "tit_for_tat")) > 0.9
    assert takes_red(infer(partner_move, after, "grim")) > 0.9
    assert takes_red(infer(partner_move, after, "cooperator")) < 0.05


def test_only_deliberate_takes_are_held_against_a_player():
    s = _state((0, 0), (2, 2), (1, 1), (0, 1))
    s2, r = step(s, 0, STAY, ((2, 0), (2, 0)))               # red steps right onto the blue coin
    assert r == (1, -2) and s2.debt == 1 and s2.grudge and s2.steals == (1, 0)
    both = _state((0, 0), (2, 2), (0, 1), (0, 1))            # both coins on one cell
    s3, r = step(both, 0, STAY, ((2, 0), (2, 1)))
    assert r == (2, -2) and s3.debt == 0 and not s3.grudge and s3.steals == (0, 0)
