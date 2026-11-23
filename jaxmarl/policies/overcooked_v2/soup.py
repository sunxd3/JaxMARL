"""The soup family: scripted Overcooked-V2 players that run the whole soup loop.

What a player does depends on what it holds:

* a dish:        take it to the goal;
* a plate:       take the soup out of a pot that is cooking or cooked, else put it down
                 if the pot still needs ingredients, else keep clear;
* an ingredient: put it in a pot that has room, else keep clear;
* nothing:       choose between fetching an ingredient, fetching a plate, and keeping clear.

The type sets this choice and how often the player idles.  The choice is a softmax over the
options that are currently useful, scored by a type weight minus a penalty on path length:

    p(option) is proportional to exp(weight[option] - distance_weight * distance[option])

A *runner* only fetches ingredients, a *plater* only fetches plates, a *flex* player takes
whichever is nearer, and a *lazy* player is a flex player who mostly stands still.  Once a move
shows which of the two errands it went for, the player keeps to it until its hands are full or
the errand stops being useful; this is its only memory.

A plate fetched before the soup is cooking is put down on a counter while the pot still needs
ingredients that nobody is bringing, and fetched from there once there is soup.

On top of that, every type stands still with probability ``rho`` and moves uniformly at random
with probability ``eps``.

Scope: two players, full view, and layouts where any ingredient pile is a useful one (the
classic single-recipe layouts).  The player reads only its own observation.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct

from jaxmarl.environments.overcooked_v2 import OvercookedV2
from jaxmarl.policies.base import Policy, Population, Seen
from jaxmarl.policies.overcooked_v2.common import (
    NUM_ACTIONS,
    STAY,
    decode,
    go_interact,
    holds_dish,
    holds_ingredient,
    holds_nothing,
    holds_plate,
    pot_contents,
    step_aside,
)

FETCH_INGREDIENT, FETCH_PLATE, KEEP_CLEAR = 0, 1, 2
POT_SIZE = 3


@struct.dataclass
class SoupParams:
    """A type of soup player."""

    w_ingredient: jax.Array  # weight on fetching an ingredient when free
    w_plate: jax.Array  # weight on fetching a plate when free
    rho: jax.Array  # probability of standing still
    eps: jax.Array  # probability of a uniformly random move


class SoupMem(NamedTuple):
    errand: jax.Array  # int32: FETCH_INGREDIENT or FETCH_PLATE once committed, else -1
    offered: (
        jax.Array
    )  # int32[2]: the two errands' moves at the last open choice, else -1


class SoupPolicy(Policy):
    def __init__(self, env, distance_weight: float = 0.5):
        if not isinstance(env, OvercookedV2):
            raise TypeError("SoupPolicy is for OvercookedV2")
        if env.agent_view_size is not None:
            raise ValueError("SoupPolicy needs full view (agent_view_size=None)")
        if env.num_agents != 2:
            raise ValueError("SoupPolicy is written for two players")
        self.num_ingredients = env.layout.num_ingredients
        self.distance_weight = distance_weight

    def init(self, params):
        return SoupMem(
            errand=jnp.array(-1, dtype=jnp.int32),
            offered=jnp.full((2,), -1, dtype=jnp.int32),
        )

    def step(self, params: SoupParams, mem: SoupMem, seen: Seen):
        v = decode(seen.view, self.num_ingredients)
        count, cooking, cooked = pot_contents(v)
        has_room = v.pot & ~cooking & ~cooked & (count < POT_SIZE)
        has_soup = cooking | cooked
        free_hands = holds_nothing(v.held)

        # the moves for each thing a player can be doing
        to_goal, _, _ = go_interact(v, v.goal)
        to_soup, _, _ = go_interact(v, has_soup, usable=cooked)
        to_pot, _, _ = go_interact(v, has_room)
        to_ingredient, d_ingredient, can_ingredient = go_interact(
            v, v.ingredient_pile.any(-1)
        )
        # a plate put down on a counter is fetched like one from the pile
        plate_on_counter = (
            v.wall
            & (v.items[..., 0] > 0)
            & (v.items[..., 1] == 0)
            & (v.items[..., 2:].sum(-1) == 0)
        )
        to_plate, d_plate, can_plate = go_interact(v, v.plate_pile | plate_on_counter)
        aside = step_aside(v)

        # which errands are useful for a player with free hands.  A plate is fetched early,
        # while the pot fills, only if none is already waiting on a counter.
        on_its_way = holds_ingredient(v.other_held).astype(count.dtype)
        want_ingredient = (
            jnp.any(has_room & (count + on_its_way < POT_SIZE)) & can_ingredient
        )
        want_plate = (
            (has_soup.any() | (jnp.any(v.pot & (count > 0)) & ~plate_on_counter.any()))
            & ~holds_plate(v.other_held)
            & can_plate
        )
        useful = jnp.stack([want_ingredient, want_plate, jnp.array(True)])

        # memory: commit to an errand once a move has shown which one it was
        went_for = jnp.where(
            (mem.offered[0] != mem.offered[1]) & (seen.last_action == mem.offered[0]),
            FETCH_INGREDIENT,
            jnp.where(
                (mem.offered[0] != mem.offered[1])
                & (seen.last_action == mem.offered[1]),
                FETCH_PLATE,
                mem.errand,
            ),
        )
        went_for = jnp.where(mem.offered[0] >= 0, went_for, mem.errand)
        errand = jnp.where(
            free_hands & (went_for >= 0) & useful[went_for], went_for, -1
        )

        # the choice when hands are free
        scores = jnp.stack(
            [
                params.w_ingredient - self.distance_weight * d_ingredient,
                params.w_plate - self.distance_weight * d_plate,
                0.0,
            ]
        )
        open_choice = jax.nn.softmax(jnp.where(useful, scores, -jnp.inf))
        choice = jnp.where(errand >= 0, jax.nn.one_hot(errand, 3), open_choice)
        free_moves = jnp.stack([to_ingredient, to_plate, aside])
        free_probs = (choice[:, None] * jax.nn.one_hot(free_moves, NUM_ACTIONS)).sum(0)

        # the move when hands are full.  A plate with no soup to wait for is put down on an
        # empty counter while the pot needs ingredients that the other player is not bringing;
        # otherwise a player who fetched a plate early would hold it forever.
        empty_counter = v.wall & (v.items.sum(-1) == 0)
        to_counter, _, can_counter = go_interact(v, empty_counter)
        plate_blocks = (
            ~has_soup.any()
            & jnp.any(has_room & (count + on_its_way < POT_SIZE))
            & can_counter
        )
        full_move = jnp.where(
            holds_dish(v.held),
            to_goal,
            jnp.where(
                holds_plate(v.held),
                jnp.where(
                    has_soup.any(),
                    to_soup,
                    jnp.where(plate_blocks, to_counter, aside),
                ),
                jnp.where(holds_ingredient(v.held) & has_room.any(), to_pot, aside),
            ),
        )
        intended = jnp.where(
            free_hands, free_probs, jax.nn.one_hot(full_move, NUM_ACTIONS)
        )

        probs = (
            (1.0 - params.rho - params.eps) * intended
            + params.rho * jax.nn.one_hot(STAY, NUM_ACTIONS)
            + params.eps / NUM_ACTIONS
        )

        is_open = free_hands & (errand < 0) & want_ingredient & want_plate
        offered = jnp.where(is_open, jnp.stack([to_ingredient, to_plate]), -1).astype(
            jnp.int32
        )
        new_mem = SoupMem(errand=errand.astype(jnp.int32), offered=offered)
        return new_mem, jnp.log(jnp.maximum(probs, 1e-30))


SOUP_TYPES = {
    #          w_ingredient, w_plate, rho
    "runner": (8.0, -8.0, 0.05),  # only fetches ingredients
    "plater": (-8.0, 8.0, 0.05),  # only fetches plates
    "flex": (2.5, 2.5, 0.05),  # whichever is nearer
    "lazy": (2.5, 2.5, 0.7),  # flex, but stands still most of the time
}


def make_population(env, eps: float = 0.03) -> Population:
    """The soup types with a uniform prior.

    The types are chosen so that the partner's type changes what the other seat should do:
    two runners or two platers deliver nothing, while a runner and a plater work well
    together, and a lazy partner leaves most of the work to you.
    """
    names = tuple(SOUP_TYPES)
    table = jnp.array([SOUP_TYPES[name] for name in names])
    k = len(names)
    return Population(
        policy=SoupPolicy(env),
        params=SoupParams(
            w_ingredient=table[:, 0],
            w_plate=table[:, 1],
            rho=table[:, 2],
            eps=jnp.full((k,), eps),
        ),
        probs=jnp.full((k,), 1.0 / k),
        names=names,
    )
