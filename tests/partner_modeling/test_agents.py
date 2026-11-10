"""The agents used to test the settings: their beliefs stay consistent with what happened.

    .venv/bin/python -m pytest tests/partner_modeling/test_agents.py -q

The probabilistic-program agents hold an exact belief about what is hidden.  In each game the
belief must never rule out the truth, must stay normalised, and must find the partner's type
where the game makes that possible.
"""

import jax
import pytest

from partner_modeling.coin_agents import PPCoinAgent
from partner_modeling.coin_model import CoinEnv
from partner_modeling.coin_run import play
from partner_modeling.hanabi_agents import ConventionAgent, PPHanabiAgent
from partner_modeling.hanabi_model import MiniHanabi, Rules, step, touched_slots
from partner_modeling.hanabi_partner import CONVENTIONS, observe
from partner_modeling.pp_focal import PPFocal
from partner_modeling.rollouts import make_study_env, run_episode
from partner_modeling.teammates import HeuristicTeammateSpec, make_teammate

CFG = {"make_way_station_only": True}
RULES = Rules()


@pytest.fixture(scope="module")
def game():
    return MiniHanabi(RULES)


def test_pp_focal_sees_the_partner_action_and_never_rules_out_the_true_recipe():
    env = make_study_env("cramped_room_v2", 40)
    partner = make_teammate(HeuristicTeammateSpec(family="typed_intent", theta="flex_pot_direct_lo", extra=CFG),
                            env, "agent_1")
    focal = PPFocal(env, cfg=CFG, hide_recipe=True, horizon=8, n_rollouts=1, replan=8, seed=0)
    seen_sets = []
    original = focal._seen_actions

    def record(prev_state, my_a, state):
        seen_sets.append(original(prev_state, my_a, state))
        return seen_sets[-1]

    focal._seen_actions = record
    log = run_episode(env, partner, focal, jax.random.PRNGKey(0), max_steps=40)
    assert len(focal.trace) == log.T and focal.impossible == 0
    for t in range(log.T - 1):                       # what it saw is consistent with what the partner did
        assert int(log.partner_actions[t]) in seen_sets[t]
    for t, step_ in enumerate(focal.trace):
        assert sum(step_["recipe"].values()) == pytest.approx(1.0)
        assert step_["recipe"].get(int(log.views[t].recipe), 0.0) > 0.0


def test_belief_keeps_the_truth_and_finds_the_convention(game):
    """The exact belief never rules out the true hand and convention, and it identifies the
    partner's hint convention within a game."""
    for seed, conv in [(0, CONVENTIONS[0]), (1, CONVENTIONS[2])]:
        seat = seed % 2
        agents = [None, None]
        agents[seat], agents[1 - seat] = PPHanabiAgent(worlds=4), ConventionAgent(conv)
        state, s = game.reset(jax.random.PRNGKey(seed))
        for i, ag in enumerate(agents):
            ag.reset(RULES, i, seed)
        while not s.done:
            p = s.cur
            before = [observe(s, 0), observe(s, 1)]
            a = int(agents[p].act(before[p]))
            kind, k = RULES.kind(a)
            revealed = s.hands[p][k] if kind in ("discard", "play") else None
            hit = tuple(touched_slots(RULES, s, a)) if kind in ("color", "rank") else None
            s, _ = step(RULES, s, a)
            after = [observe(s, 0), observe(s, 1)]
            for i, ag in enumerate(agents):
                ag.saw(before[i], p, a, revealed, hit, after[i])
            pp = agents[seat]
            assert pp.unexplained == 0
            assert pp.belief.prob((s.hands[seat], conv)) > 0
        assert pp.belief.marginal(lambda v: v[1].signal).prob(conv.signal) > 0.9


def test_belief_identifies_a_greedy_partner_and_stays_normalised():
    game = CoinEnv(steps=40)
    agent = PPCoinAgent(types=("cooperator", "greedy", "tit_for_tat"), rollouts=4)
    play(game, agent, "greedy", seed=0)
    assert agent.unexplained == 0
    assert all(sum(st["belief"].values()) == pytest.approx(1.0) for st in agent.trace)
    assert agent.trace[-1]["belief"].get("greedy", 0.0) > 0.9
