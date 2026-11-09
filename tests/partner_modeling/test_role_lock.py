"""Role-commitment lock.

    .venv/bin/python -m pytest tests/partner_modeling/test_role_lock.py -q

1. L = 0 is bit-identical to the pre-lock focal: replays four redesign-round episodes
   (recorded with the original study code) action-for-action.
2. Lock invariants at L = 300 / 25: a P-locked focal never picks up an ingredient from a pile or
   counter, an F-locked focal never picks up a plate; commitments happen exactly at t = 0, L, 2L, ...
"""

import json
import zlib
from pathlib import Path

import jax
import numpy as np
import pytest

from partner_modeling.partners.typed_intent import THETA_FAMILIES
from partner_modeling.rollouts import (env_layout, locked_strategy_family, make_study_env, partner_events,
                               run_episode, strategy_family)
from partner_modeling.teammates import HeuristicTeammateSpec, make_teammate

CFG = {"make_way_station_only": True}
GOLD = json.loads((Path(__file__).parent / "data" / "lock_l0_golden.json").read_text())


@pytest.fixture(scope="module")
def env():
    return make_study_env("coord_ring", 150, full_obs=True)


def _partner(env, family, ti):
    th = THETA_FAMILIES[family]()[ti]
    return make_teammate(HeuristicTeammateSpec(family=family, theta=th.preset_name(), extra=CFG), env, "agent_1")


def _key(name, ti, seed):   # as analysis.sweep._run_one
    return jax.random.fold_in(jax.random.PRNGKey(seed), ti * 1000 + zlib.crc32(name.encode()) % 997)


def test_l0_bit_identical_to_recorded_redesign_episodes(env):
    lay = env_layout(env)
    fam = strategy_family(lay, ctrl_cfg=CFG)
    for ep in GOLD["episodes"]:
        foc = fam[ep["focal"]]
        assert foc.lock_L == 0
        log = run_episode(env, _partner(env, "typed_intent_x", ep["theta"]), foc,
                          _key(ep["focal"], ep["theta"], ep["seed"]), max_steps=150)
        assert log.focal_actions.tolist() == ep["focal_actions"]
        assert log.partner_actions.tolist() == ep["partner_actions"]


@pytest.mark.parametrize("L", [300, 25])
def test_lock_invariants(env, L):
    lay = env_layout(env)
    lib = locked_strategy_family(lay, L, ctrl_cfg=CFG)
    assert len(lib) == 30
    for name in ("g_bal_pile_nostage@F", "g_bal_pile_nostage@P", "g_pf_stg_stage@P", "fetcher@P"):
        foc = lib[name]
        for ti in (0, 6):
            log = run_episode(env, _partner(env, "typed_intent", ti), foc, _key(name, ti, 1), max_steps=150)
            ts = [t for t, _ in foc.role_log]
            assert ts == list(range(0, 150, L))
            roles = dict(foc.role_log)
            for t, e in partner_events(log, 0):
                r = roles[max(b for b in ts if b <= t)]
                if r == "P":
                    assert e not in ("-->o@pile", "-->o@counter"), (name, ti, t, e)
                else:
                    assert e != "-->p@plate_pile" and e != "-->p@counter", (name, ti, t, e)
