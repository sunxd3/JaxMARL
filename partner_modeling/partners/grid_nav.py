"""Navigation on a static Overcooked-V2 grid, shared by the scripted agents.

``GridNav`` holds the layout masks, all-pairs shortest paths between walkable cells (BFS) and a
few helpers: distances to stations, a shortest-path controller that ignores other agents except
for occupied target cells, and pot-state decoding.  Everything is host-side NumPy and depends only
on the static grid.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from partner_modeling.env_compat import Actions, Direction, DynamicObject, StaticObject

INF = 10000                                       # "unreachable" distance
FAR = 100.0                                       # distance reported when no target is reachable
OCCUPIED_PENALTY = 100                            # extra cost of an access cell another agent stands on
MOVES = ((0, 1, Actions.right), (1, 0, Actions.down), (0, -1, Actions.left), (-1, 0, Actions.up))
NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))   # scan order for a station's access cells
FACE_ACTION = {Direction.UP: Actions.up, Direction.DOWN: Actions.down,
               Direction.RIGHT: Actions.right, Direction.LEFT: Actions.left}


def _neighbour_or(mask: np.ndarray) -> np.ndarray:
    """Cells 4-adjacent to any True cell of ``mask`` (no wrap-around)."""
    out = np.zeros_like(mask, dtype=bool)
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    return out


def _direction_to(dy: int, dx: int) -> Direction:
    """Facing direction for a unit offset (dy, dx) from the agent to its target."""
    if dy == -1:
        return Direction.UP
    if dy == 1:
        return Direction.DOWN
    return Direction.RIGHT if dx == 1 else Direction.LEFT


class GridNav:
    """Masks and shortest paths for one static grid (``layout.static_objects``)."""

    def __init__(self, layout):
        static = np.asarray(layout.static_objects).astype(np.int32)
        self.static_objects_np = static
        self.height, self.width = static.shape

        self.walkable_mask_np = static == StaticObject.EMPTY
        self.ingredient_pile_mask_np = static >= StaticObject.INGREDIENT_PILE_BASE
        self.plate_pile_mask_np = static == StaticObject.PLATE_PILE
        self.pot_mask_np = static == StaticObject.POT
        self.goal_mask_np = static == StaticObject.GOAL
        self.counter_mask_np = static == StaticObject.WALL
        self.pot_adjacent_counter_mask_np = self.counter_mask_np & _neighbour_or(self.pot_mask_np)

        self._shortest_paths()

    # ------------------------------------------------------------ shortest paths
    def _shortest_paths(self):
        """BFS from every walkable cell.

        ``dist_matrix[i, y, x]``: steps from walkable cell ``idx_to_pos[i]`` to (y, x), ``INF`` if
        unreachable or not walkable.  ``first_step[i, y, x]``: the first move of a shortest path
        (neighbours are expanded right, down, left, up, so ties go to the earliest such move;
        ``stay`` at the source and for unreachable cells).
        """
        H, W = self.height, self.width
        self.idx_to_pos = np.argwhere(self.walkable_mask_np)
        self.pos_to_idx = np.full((H, W), -1, dtype=np.int32)
        for i, (y, x) in enumerate(self.idx_to_pos):
            self.pos_to_idx[y, x] = i
        n = len(self.idx_to_pos)
        self.dist_matrix = np.full((n, H, W), INF, dtype=np.int32)
        self.first_step = np.full((n, H, W), int(Actions.stay), dtype=np.int32)
        for i, (sy, sx) in enumerate(self.idx_to_pos):
            dist, step = self.dist_matrix[i], self.first_step[i]
            dist[sy, sx] = 0
            frontier = [(sy, sx)]
            for y, x in frontier:                      # grows while iterating: FIFO order
                for dy, dx, a in MOVES:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and self.walkable_mask_np[ny, nx] and dist[ny, nx] == INF:
                        dist[ny, nx] = dist[y, x] + 1
                        step[ny, nx] = int(a) if (y, x) == (sy, sx) else step[y, x]
                        frontier.append((ny, nx))
        # cells next to anything reachable from i: the stations i can use
        self.station_reach = np.stack([_neighbour_or(d < INF) for d in self.dist_matrix]) if n else \
            np.zeros((0, H, W), dtype=bool)

    def _access_cells(self, ty: int, tx: int):
        """Walkable cells next to (ty, tx), in ``NEIGHBOURS`` order."""
        for dy, dx in NEIGHBOURS:
            ay, ax = ty + dy, tx + dx
            if 0 <= ay < self.height and 0 <= ax < self.width and self.walkable_mask_np[ay, ax]:
                yield ay, ax

    def _station_dist(self, idx: int, ty: int, tx: int) -> int:
        """Steps from walkable cell ``idx`` to the nearest access cell of station (ty, tx)."""
        return min((int(self.dist_matrix[idx, ay, ax]) for ay, ax in self._access_cells(ty, tx)), default=INF)

    # ------------------------------------------------------------ queries
    def reachable_targets(self, target_mask: np.ndarray, y: int, x: int) -> np.ndarray:
        """The cells of ``target_mask`` that can be used from (y, x)."""
        idx = self.pos_to_idx[y, x]
        if idx < 0:
            return np.zeros_like(target_mask, dtype=bool)
        return target_mask & self.station_reach[idx]

    def min_dist_to_target(self, target_mask: np.ndarray, y: int, x: int) -> float:
        """Steps from (y, x) to the nearest access cell of any target; ``FAR`` if none is reachable."""
        idx = self.pos_to_idx[y, x]
        if idx < 0 or not np.any(target_mask):
            return FAR
        d = min(self._station_dist(idx, ty, tx) for ty, tx in np.argwhere(target_mask))
        return float(d) if d < INF else FAR

    def closest_target(self, target_mask: np.ndarray, y: int, x: int) -> Tuple[int, int, bool]:
        """Nearest target (first in row-major order among equals) and whether one is reachable."""
        idx = self.pos_to_idx[y, x]
        if idx < 0 or not np.any(target_mask):
            return y, x, False
        best, best_d = None, INF
        for ty, tx in np.argwhere(target_mask):
            d = self._station_dist(idx, ty, tx)
            if d < best_d:
                best, best_d = (int(ty), int(tx)), d
        if best is None:
            return y, x, False
        return best[0], best[1], True

    # ------------------------------------------------------------ shortest-path controller
    def step_to_station(self, y: int, x: int, facing: int, ty: int, tx: int, others: np.ndarray) -> int:
        """Action that brings an agent at (y, x) to station (ty, tx) and uses it.

        Next to the station: interact if facing it, otherwise turn to face it.  Elsewhere: the
        first step of a shortest path to the cheapest access cell, where an access cell occupied by
        another agent (``others``: (N, 2) positions) costs ``OCCUPIED_PENALTY`` extra.
        """
        idx = self.pos_to_idx[y, x]
        access = list(self._access_cells(ty, tx))
        if idx < 0 or not access:
            return int(Actions.stay)
        dy, dx = ty - y, tx - x
        if abs(dy) + abs(dx) == 1:
            want = _direction_to(dy, dx)
            return int(Actions.interact) if facing == want else int(FACE_ACTION[want])
        occupied = {(int(p[0]), int(p[1])) for p in others}
        best, best_cost = None, INF
        for ay, ax in access:
            cost = int(self.dist_matrix[idx, ay, ax]) + (OCCUPIED_PENALTY if (ay, ax) in occupied else 0)
            if cost < best_cost:
                best, best_cost = (ay, ax), cost
        if best is None:
            return int(Actions.stay)
        return int(self.first_step[idx, best[0], best[1]])

    def step_next_to(self, y: int, x: int, facing: int, ty: int, tx: int, others: np.ndarray) -> int:
        """As ``step_to_station``, but once next to the station only turn to face it, then wait."""
        dy, dx = ty - y, tx - x
        if abs(dy) + abs(dx) == 1:
            want = _direction_to(dy, dx)
            return int(Actions.stay) if facing == want else int(FACE_ACTION[want])
        return self.step_to_station(y, x, facing, ty, tx, others)

    # ------------------------------------------------------------ pots and recipes
    def pot_states(self, grid: np.ndarray):
        """(is_pot, can_take_ingredient, cooking, cooked, full_not_started, ingredient_count) masks."""
        static, dyn, timer = grid[:, :, 0], grid[:, :, 1], grid[:, :, 2]
        is_pot = static == StaticObject.POT
        counts = np.zeros(dyn.shape, dtype=np.int32)
        for y, x in np.argwhere(is_pot):
            counts[y, x] = sum(_two_bit_fields(int(dyn[y, x]) >> 2))
        full = counts >= 3
        cooking = (timer > 0) & is_pot
        cooked = ((dyn & DynamicObject.COOKED) != 0) & is_pot
        idle = is_pot & ~cooking & ~cooked
        return is_pot, idle & ~full, cooking, cooked, idle & full, counts

    @staticmethod
    def decode_recipe(recipe: int) -> np.ndarray:
        """Per-ingredient counts (4,) of a recipe encoding (two bits per ingredient after 2 flag bits)."""
        fields = _two_bit_fields(int(recipe) >> 2)
        return np.array((fields + [0, 0, 0, 0])[:4], dtype=np.int32)


def _two_bit_fields(v: int):
    out = []
    while v > 0:
        out.append(v & 0x3)
        v >>= 2
    return out
