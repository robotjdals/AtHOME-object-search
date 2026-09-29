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


_GRAPHS: Dict[int, tuple] = {}


def _grid_graph(free: np.ndarray):
    """CSR graph of the same moves as ``shortest_paths`` (8-connected, no
    corner cutting, weights 1 and sqrt(2) in cells), built once per grid."""
    cached = _GRAPHS.get(id(free))
    if cached is not None and cached[0] is free:
        return cached[1]
    from scipy.sparse import csr_matrix
    height, width = free.shape
    index = np.arange(height * width).reshape(height, width)
    rows, cols, weights = [], [], []
    for dr, dc, step in _MOVES:
        r0, r1 = max(0, -dr), height - max(0, dr)
        c0, c1 = max(0, -dc), width - max(0, dc)
        ok = free[r0:r1, c0:c1] & free[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        if dr and dc:
            ok &= free[r0 + dr:r1 + dr, c0:c1] & free[r0:r1, c0 + dc:c1 + dc]
        src = index[r0:r1, c0:c1][ok]
        rows.append(src)
        cols.append(src + dr * width + dc)
        weights.append(np.full(len(src), step))
    graph = csr_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(height * width, height * width))
    if len(_GRAPHS) > 16:
        _GRAPHS.clear()
    _GRAPHS[id(free)] = (free, graph)
    return graph


def shortest_costs(
    free: np.ndarray,
    start: Cell,
    goals: Iterable[Cell],
    resolution: float,
) -> Dict[Cell, float]:
    """Path cost [m] from ``start`` to every reachable goal: the costs of
    ``shortest_paths`` (same graph and Dijkstra), computed by
    scipy.sparse.csgraph.dijkstra in C; no parent map."""
    from scipy.sparse.csgraph import dijkstra
    height, width = free.shape
    if not free[start]:
        return {}
    dist = dijkstra(_grid_graph(free), directed=True, indices=start[0] * width + start[1])
    out = {}
    for g in goals:
        if free[g]:
            d = dist[g[0] * width + g[1]]
            if np.isfinite(d):
                out[g] = float(d) * resolution
    return out


def extract_path(parent: Dict[Cell, Cell], start: Cell, goal: Cell) -> List[Cell]:
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path
