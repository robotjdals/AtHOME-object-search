"""Observed fraction of each room (the searched region of Bayesian search).

After an unsuccessful search, the probability that the target is in a room
falls with the part of the room that was actually observed (Koopman 1946;
Stone 1975), not with the number of places visited: one 360-degree
observation often covers most of a small room. Each room is sampled with
points on its floor; a point counts as observed once a visit pose sees it
under the observation model (same range and occlusion as target
detection). The real robot uses the same with its room segmentation map and
occupancy line of sight.
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping

import numpy as np

Sees = Callable[[float, float, float, float], bool]   # (x, y, px, py) -> visible


class RoomCoverage:
    def __init__(self, room_points: Mapping[str, np.ndarray], sees: Sees):
        self._points = {rid: np.asarray(p, dtype=float).reshape(-1, 2) for rid, p in room_points.items()}
        self._seen = {rid: np.zeros(len(p), dtype=bool) for rid, p in self._points.items()}
        self._sees = sees

    def observe(self, x: float, y: float) -> None:
        """Mark the room points visible from an observation pose."""
        for rid, points in self._points.items():
            seen = self._seen[rid]
            for i in np.flatnonzero(~seen):
                if self._sees(x, y, points[i, 0], points[i, 1]):
                    seen[i] = True

    def fraction(self, room_id: str) -> float:
        seen = self._seen.get(room_id)
        return float(seen.mean()) if seen is not None and len(seen) else 0.0


def line_of_sight(walls, range_m: float) -> Sees:
    """Visibility of the wall line-of-sight observation model (wall_los.py)."""
    def sees(x, y, px, py):
        return (np.hypot(px - x, py - y) <= range_m
                and walls.first_hit((x, y), (px, py)) is None)
    return sees


def floor_points(polygons, spacing: float = 0.2) -> np.ndarray:
    """Regular samples inside the union of floor polygons (shapely)."""
    from shapely import contains_xy
    from shapely.ops import unary_union
    region = unary_union(list(polygons))
    if region.is_empty:
        return np.zeros((0, 2))
    x0, y0, x1, y1 = region.bounds
    xs, ys = np.meshgrid(np.arange(x0 + spacing / 2, x1, spacing), np.arange(y0 + spacing / 2, y1, spacing))
    xs, ys = xs.ravel(), ys.ravel()
    inside = contains_xy(region, xs, ys)
    return np.stack([xs[inside], ys[inside]], axis=1)


def room_points_from_labels(labels: np.ndarray, origin, resolution: float,
                            spacing: float = 0.2) -> dict:
    """Room segmentation grid (proposal 4-1, ``room_<i>`` = label i) -> floor
    samples per room, about ``spacing`` apart (row = +y, col = +x)."""
    step = max(1, int(round(spacing / resolution)))
    sub = labels[::step, ::step]
    rows, cols = np.nonzero(sub > 0)
    out = {}
    for label in np.unique(sub[rows, cols]):
        m = sub[rows, cols] == label
        xs = origin[0] + (cols[m] * step + 0.5) * resolution
        ys = origin[1] + (rows[m] * step + 0.5) * resolution
        out[f"room_{int(label)}"] = np.stack([xs, ys], axis=1)
    return out


def occupancy_line_of_sight(occupied: np.ndarray, origin, resolution: float, range_m: float) -> Sees:
    """Visibility on an occupancy grid: within range and no occupied cell on
    the ray (sampled every half cell), the 2D counterpart of the wall model."""
    def sees(x, y, px, py):
        d = float(np.hypot(px - x, py - y))
        if d > range_m:
            return False
        n = max(2, int(np.ceil(d / (resolution / 2))) + 1)
        t = np.linspace(0.0, 1.0, n)[:-1]         # the target cell itself may be occupied
        cols = np.floor((x + t * (px - x) - origin[0]) / resolution).astype(int)
        rows = np.floor((y + t * (py - y) - origin[1]) / resolution).astype(int)
        inside = (rows >= 0) & (rows < occupied.shape[0]) & (cols >= 0) & (cols < occupied.shape[1])
        return not occupied[rows[inside], cols[inside]].any()
    return sees
