"""Shared pieces for scripted Overcooked-V2 policies.

``decode``        turn a seat's observation grid back into positions, items and pots.
``go_interact``   the move that brings the seat to an object of some kind and uses it.
``step_aside``    the move that gets the seat out of the way while it has nothing to do.

Everything here reads only the seat's default observation at full view
(``agent_view_size=None``), which is an absolute grid of layers; see
``OvercookedV2.get_obs_default`` for the layer order.  Cells are ``(y, x)``.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v2.common import OvercookedActionsEnum

RIGHT = int(OvercookedActionsEnum.right)
DOWN = int(OvercookedActionsEnum.down)
LEFT = int(OvercookedActionsEnum.left)
UP = int(OvercookedActionsEnum.up)
STAY = int(OvercookedActionsEnum.stay)
INTERACT = int(OvercookedActionsEnum.interact)
NUM_ACTIONS = len(OvercookedActionsEnum)
MOVE_NAMES = tuple(a.name for a in OvercookedActionsEnum)

# (dy, dx) of the four move actions, indexed by action
_DELTAS = ((0, 1), (1, 0), (0, -1), (-1, 0))
MOVE_DELTAS = jnp.array(_DELTAS, dtype=jnp.int32)
# the move action that faces each Direction (UP, DOWN, RIGHT, LEFT)
DIRECTION_TO_ACTION = jnp.array([UP, DOWN, RIGHT, LEFT], dtype=jnp.int32)

FAR = 10_000  # larger than any path


class View(NamedTuple):
    """One seat's observation, decoded.  Grids are ``[H, W]``; ``n`` is the number of
    ingredients.  Item vectors are ``[plate, cooked, count of ingredient 0, ...]``."""

    pos: jax.Array  # [2] (y, x)
    dir: jax.Array  # Direction
    held: jax.Array  # [2 + n]
    other_pos: jax.Array  # [2]
    other_dir: jax.Array
    other_held: jax.Array  # [2 + n]
    wall: jax.Array  # bool; walls double as counters
    goal: jax.Array  # bool
    pot: jax.Array  # bool
    plate_pile: jax.Array  # bool
    ingredient_pile: jax.Array  # bool [H, W, n]
    walkable: jax.Array  # bool; floor cells, including those the players stand on
    items: jax.Array  # int [H, W, 2 + n]; what lies on each cell or is in each pot
    recipe: jax.Array  # int [H, W, 2 + n]; shown only at recipe indicators
    pot_timer: jax.Array  # int; steps of cooking left


def _decode_player(layers):
    """Position, facing and inventory from one player's ``[H, W, 7 + n]`` layers."""
    width = layers.shape[1]
    flat = jnp.argmax(layers[..., 0])
    y, x = flat // width, flat % width
    cell = layers[y, x]
    return (
        jnp.stack([y, x]).astype(jnp.int32),
        jnp.argmax(cell[1:5]).astype(jnp.int32),
        cell[5:],
    )


def decode(obs, num_ingredients: int) -> View:
    """Decode a full-view default observation ``[H, W, C]`` for a two-player layout."""
    n = num_ingredients
    per_player = 7 + n
    a = 2 * per_player
    pos, facing, held = _decode_player(obs[..., :per_player])
    other_pos, other_facing, other_held = _decode_player(obs[..., per_player:a])
    static = (
        obs[..., a : a + 6] > 0
    )  # wall, goal, pot, recipe indicator, button, plate pile
    piles = obs[..., a + 6 : a + 6 + n] > 0
    return View(
        pos=pos,
        dir=facing,
        held=held,
        other_pos=other_pos,
        other_dir=other_facing,
        other_held=other_held,
        wall=static[..., 0],
        goal=static[..., 1],
        pot=static[..., 2],
        plate_pile=static[..., 5],
        ingredient_pile=piles,
        walkable=~(static.any(-1) | piles.any(-1)),
        items=obs[..., a + 6 + n : a + 8 + 2 * n],
        recipe=obs[..., a + 8 + 2 * n : a + 10 + 3 * n],
        pot_timer=obs[..., a + 10 + 3 * n],
    )


# --- what a player holds -------------------------------------------------------------------


def holds_nothing(held):
    return jnp.all(held == 0)


def holds_dish(held):
    return held[1] > 0


def holds_plate(held):
    return (held[0] > 0) & (held[1] == 0) & (held[2:].sum() == 0)


def holds_ingredient(held):
    return (held[0] == 0) & (held[1] == 0) & (held[2:].sum() > 0)


# --- pots ----------------------------------------------------------------------------------


def pot_contents(view: View):
    """Per cell: number of ingredients in the pot, whether it is cooking, whether cooked."""
    count = jnp.where(view.pot, view.items[..., 2:].sum(-1), 0)
    cooked = view.pot & (view.items[..., 1] > 0)
    cooking = view.pot & (view.pot_timer > 0)
    return count, cooking, cooked


# --- moving around -------------------------------------------------------------------------


def _neighbour(grid, action, fill):
    """``out[y, x] = grid[(y, x) + MOVE_DELTAS[action]]``, with ``fill`` outside the grid."""
    height, width = grid.shape
    dy, dx = _DELTAS[action]
    padded = jnp.pad(grid, 1, constant_values=fill)
    return padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]


def _at(grid, cell, fill):
    """``grid[cell]``, or ``fill`` if ``cell`` is outside the grid."""
    height, width = grid.shape
    inside = (cell[0] >= 0) & (cell[0] < height) & (cell[1] >= 0) & (cell[1] < width)
    return jnp.where(
        inside,
        grid[jnp.clip(cell[0], 0, height - 1), jnp.clip(cell[1], 0, width - 1)],
        fill,
    )


def next_to(mask):
    """Cells with a 4-neighbour in ``mask``."""
    return jnp.stack([_neighbour(mask, a, False) for a in range(4)]).any(0)


def distance_to(walkable, goals):
    """Shortest number of moves from every walkable cell to a cell in ``goals``."""

    def relax(carry):
        dist, _ = carry
        around = jnp.stack([_neighbour(dist, a, FAR) for a in range(4)]).min(0)
        new = jnp.where(walkable, jnp.minimum(dist, around + 1), FAR)
        return new, jnp.any(new != dist)

    # relax until nothing changes: as many rounds as the longest path, not one per cell
    dist = jnp.where(goals & walkable, 0, FAR)
    dist, _ = jax.lax.while_loop(lambda c: c[1], relax, (dist, jnp.array(True)))
    return dist


def _free_floor(view: View):
    """Floor the seat can plan through: everything walkable except the other player's cell."""
    return view.walkable.at[view.other_pos[0], view.other_pos[1]].set(False)


def _step_towards(view: View, dist):
    """The move to the neighbouring free cell with the smallest ``dist``; STAY if none helps."""
    free = _free_floor(view)
    costs = jnp.stack(
        [
            jnp.where(
                _at(free, view.pos + MOVE_DELTAS[a], False),
                _at(dist, view.pos + MOVE_DELTAS[a], FAR),
                FAR,
            )
            for a in range(4)
        ]
    )
    return jnp.where(costs.min() < FAR, jnp.argmin(costs), STAY).astype(jnp.int32)


def go_interact(view: View, targets, usable=None):
    """The move that brings the seat to a cell in ``targets`` and interacts with it.

    ``targets`` are object cells (not floor).  Next to a target, the seat turns to face it and
    then interacts; if ``usable`` is given, it interacts only with targets in ``usable`` and
    otherwise waits in front of the target.  Elsewhere it takes a shortest path around the
    other player.  Returns ``(action, distance, reachable)``.
    """
    usable = targets if usable is None else usable
    beside = jnp.stack(
        [_at(targets, view.pos + MOVE_DELTAS[a], False) for a in range(4)]
    )
    facing_action = DIRECTION_TO_ACTION[view.dir]
    faced = view.pos + MOVE_DELTAS[facing_action]
    in_front = jnp.where(_at(usable, faced, False), INTERACT, STAY)
    turn = jnp.argmax(beside)
    adjacent_action = jnp.where(beside[facing_action], in_front, turn)

    free = _free_floor(view)
    dist = distance_to(free, next_to(targets))
    here = dist[view.pos[0], view.pos[1]]
    action = jnp.where(beside.any(), adjacent_action, _step_towards(view, dist))
    distance = jnp.where(beside.any(), 0, here)
    reachable = targets.any() & (distance < FAR)
    action = jnp.where(reachable, action, STAY)
    return action.astype(jnp.int32), distance, reachable


def step_aside(view: View):
    """The move for a seat with nothing to do: leave the cells others need.

    A cell is "in use" if it is next to a pot, pile or goal.  From such a cell the seat walks
    to the nearest floor cell that is not; if it is already on one, or there is none, it stays.
    """
    stations = view.pot | view.goal | view.plate_pile | view.ingredient_pile.any(-1)
    free = _free_floor(view)
    quiet = free & ~next_to(stations)
    dist = distance_to(free, quiet)
    here = dist[view.pos[0], view.pos[1]]
    return jnp.where((here > 0) & (here < FAR), _step_towards(view, dist), STAY).astype(
        jnp.int32
    )
