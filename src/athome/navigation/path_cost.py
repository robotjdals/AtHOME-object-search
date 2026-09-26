"""Grid shortest paths: 8-connected, no corner cutting, cost in meters."""

from __future__ import annotations

import heapq
import math
from typing import Dict, Iterable, List, Tuple

import numpy as np

from athome.navigation.grid import Cell

_MOVES = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
)


def shortest_paths(
    free: np.ndarray,
    start: Cell,
    goals: Iterable[Cell],
    resolution: float,
) -> Tuple[Dict[Cell, float], Dict[Cell, Cell]]:
    """Dijkstra from ``start`` until every reachable goal is settled.

    One run gives the path cost to all candidate goals of all locations,
    which is what search planning needs each step.
    Returns (cost per reached goal, parent map).
    """
    remaining = {g for g in goals if free[g]}
    height, width = free.shape
    dist = {start: 0.0}
    parent: Dict[Cell, Cell] = {}
    reached: Dict[Cell, float] = {}
    queue = [(0.0, start)]

    while queue and remaining:
        cost, cell = heapq.heappop(queue)
        if cost > dist[cell]:
            continue
        if cell in remaining:
            remaining.discard(cell)
            reached[cell] = cost * resolution

        row, col = cell
        for dr, dc, step in _MOVES:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < height and 0 <= nc < width) or not free[nr, nc]:
                continue
            if dr and dc and not (free[row + dr, col] and free[row, col + dc]):
                continue
            new_cost = cost + step
            if new_cost < dist.get((nr, nc), math.inf):
                dist[(nr, nc)] = new_cost
                parent[(nr, nc)] = cell
                heapq.heappush(queue, (new_cost, (nr, nc)))

    return reached, parent


def extract_path(parent: Dict[Cell, Cell], start: Cell, goal: Cell) -> List[Cell]:
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path
