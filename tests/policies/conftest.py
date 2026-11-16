"""The list every policy family and every env used with policies is added to.

Each case is ``(name, make the env, make a population for every seat)``.  The tests in
``test_conformance.py`` run on all of them.
"""

import pytest

from jaxmarl import make
from jaxmarl.policies import uniform_population
from jaxmarl.policies.overcooked_v2 import soup


def _overcooked():
    return make("overcooked_v2", layout="cramped_room", max_steps=40)


def _coin_game():
    return make("coin_game", num_inner_steps=8, num_outer_steps=3)


def _uniform(env):
    return {seat: uniform_population(env, seat) for seat in env.agents}


def _soup(env):
    population = soup.make_population(env)
    return {seat: population for seat in env.agents}


CASES = [
    ("overcooked_v2/soup", _overcooked, _soup),
    ("overcooked_v2/uniform", _overcooked, _uniform),
    ("coin_game/uniform", _coin_game, _uniform),
]


@pytest.fixture(params=CASES, ids=[name for name, _, _ in CASES])
def case(request):
    _, make_env, make_seats = request.param
    env = make_env()
    return env, make_seats(env)
