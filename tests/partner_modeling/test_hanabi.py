"""The small Hanabi: the game model against JaxMARL's environment, and the convention partner program.

    .venv/bin/python -m pytest tests/partner_modeling/test_hanabi.py -q
"""

import jax
import numpy as np
import pytest

from partner_modeling.hanabi_model import MiniHanabi, Rules, initial_state, legal_moves, model_public, step
from partner_modeling.hanabi_partner import CONVENTIONS, convention_move, observe
from partner_modeling.ppl import infer, simulate

RULES = Rules()


@pytest.fixture(scope="module")
def game():
    return MiniHanabi(RULES)


@pytest.mark.parametrize("careful", [False, True])
def test_model_matches_jaxmarl_environment(game, careful):
    """Hands, fireworks, tokens, hint masks, turn, score, end of game and legal moves agree after
    every move.  Reckless play covers losing all lives; careful play reaches the final round."""
    rng = np.random.default_rng(int(careful))
    ended_with_score = 0
    for g in range(6):
        state, s = game.reset(jax.random.PRNGKey(g))
        assert game.public(state) == model_public(s)
        while not s.done:
            legal = legal_moves(RULES, s)
            env_legal = np.flatnonzero(np.asarray(game.env.get_legal_moves(state)[game.env.agents[s.cur]])[:-1])
            assert sorted(legal) == sorted(int(a) for a in env_legal)
            good = [RULES.play(i) for i, c in enumerate(s.hands[s.cur]) if s.playable(c)]
            safe = [a for a in legal if RULES.kind(a)[0] != "play"]
            if good and rng.random() < 0.7:
                a = int(rng.choice(good))
            else:
                a = int(rng.choice(safe if careful and safe else legal))
            state, reward, done = game.step(jax.random.PRNGKey(0), state, a)
            s, r = step(RULES, s, a)
            assert (r, s.done) == (reward, done)
            assert game.public(state) == model_public(s)
        ended_with_score += int(s.score > 0)
    assert ended_with_score > 0 if careful else True


def _random_deck(seed):
    deck = RULES.all_cards()
    np.random.default_rng(seed).shuffle(deck)
    return [tuple(c) for c in deck]


def test_convention_program_gives_every_legal_move_some_probability():
    rng = np.random.default_rng(0)
    for g, conv in enumerate(CONVENTIONS):
        s = initial_state(RULES, _random_deck(g))
        while not s.done:
            obs = observe(s, s.cur)
            d = infer(convention_move, RULES, obs, s.cur, conv)
            assert set(d.values) == set(legal_moves(RULES, s))
            assert d.probs.sum() == pytest.approx(1.0)
            s, _ = step(RULES, s, simulate(convention_move, RULES, obs, s.cur, conv, rng=rng))


def test_conventions_read_the_same_hint_differently():
    """A hint that touches two cards marks the newest for a 'newest' player and the oldest for
    an 'oldest' one; a 'literal' player plays neither, because the hint does not pin a card down."""
    hands = [(1, 0), (2, 0), (2, 1), (0, 0), (1, 4), (0, 2)]      # player 1 holds R1 Y5 R3
    rest = RULES.all_cards()
    for c in hands:
        rest.remove(c)
    s = initial_state(RULES, hands + rest)
    s, _ = step(RULES, s, RULES.hint_color(0))                    # "red" touches its slots 0 and 2
    obs = observe(s, 1)
    best = {c.signal: infer(convention_move, RULES, obs, 1, c).argmax() for c in CONVENTIONS}
    assert best["newest"] == RULES.play(2) and best["oldest"] == RULES.play(0)
    assert RULES.kind(best["literal"])[0] != "play"
