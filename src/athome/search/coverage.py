"""Observed fraction of each room (the searched region of Bayesian search).

After an unsuccessful search, the probability that the target is in a room
falls with the part of the room that was actually observed (Koopman 1946;
Stone 1975), not with the number of places visited: one 360-degree
observation often covers most of a small room. Each room is sampled with
points on its floor; a point counts as observed once a visit pose sees it
under the observation model (same range and occlusion as target
detection). The real robot uses the same with its room segmentation map and
occupancy line of sight.

With an imperfect detector the searched region is the probability of
detection (POD) instead (Koopman 1946): a point seen from k distinct
viewpoints would have shown a target there with probability 1 - E[(1 - q)^k],
q ~ Beta of the detection model (athome.symbolic.ViewpointDetectionModel,
fitted to the robot: a viewpoint repeats its outcome, so looking again from
the same pose adds nothing). ``fraction`` is the mean POD of a room's points;
with a perfect detector (``detection`` None or mean 1) it is the observed
fraction.
"""
from __future__ import annotations

from typing import Callable, Mapping, Tuple

import numpy as np

Sees = Callable[[float, float, float, float], bool]   # (x, y, px, py) -> visible


class RoomCoverage:
    """``cache``: optional dict shared by trackers of the same rooms and
    observation model (e.g. all rollouts of a scene): the points visible
    from a pose do not depend on the episode, so they are computed once per
    (pose, room) and reused; the result is the same."""

    def __init__(self, room_points: Mapping[str, np.ndarray], sees: Sees, cache: dict = None,
                 detection: "Tuple[float, float] | None" = None):
        """``detection``: (mean, rho) of the viewpoint detection model, for the
        probability of detection; None: a perfect detector."""
        self._points = {rid: np.asarray(p, dtype=float).reshape(-1, 2) for rid, p in room_points.items()}
        self._seen = {rid: np.zeros(len(p), dtype=bool) for rid, p in self._points.items()}
        self._views = {rid: np.zeros(len(p), dtype=int) for rid, p in self._points.items()}
        self._poses = set()
        self._sees = sees
        self._cache = cache
        self._miss = None if detection is None or detection[0] >= 1.0 else _miss_probabilities(*detection)

    def observe(self, x: float, y: float) -> None:
        """Mark the room points visible from an observation pose (a pose seen
        before is the same viewpoint: nothing new)."""
        if (x, y) in self._poses:
            return
        self._poses.add((x, y))
        many = getattr(self._sees, "many", None)
        for rid, points in self._points.items():
            seen = self._seen[rid]
            if many is not None and self._cache is not None:
                key = (x, y, rid)
                if key not in self._cache:
                    self._cache[key] = many(x, y, points)
                visible = self._cache[key]
            elif self._miss is None:
                todo = np.flatnonzero(~seen)       # only unseen points can change
                visible = np.zeros(len(points), dtype=bool)
                if many is not None:               # one vectorized query (same test)
                    visible[todo[many(x, y, points[todo])]] = True
                else:
                    for i in todo:
                        visible[i] = self._sees(x, y, points[i, 0], points[i, 1])
            else:
                visible = (many(x, y, points) if many is not None else
                           np.array([self._sees(x, y, px, py) for px, py in points], dtype=bool))
            seen |= visible
            self._views[rid] += visible

    def fraction(self, room_id: str) -> float:
        seen = self._seen.get(room_id)
        if seen is None or not len(seen):
            return 0.0
        if self._miss is None:
            return float(seen.mean())
        views = np.minimum(self._views[room_id], len(self._miss) - 1)
        return float(np.mean(1.0 - self._miss[views]))


def _miss_probabilities(mean: float, rho: float, most: int = 64) -> np.ndarray:
    """E[(1 - q)^k] for k = 0..most, q ~ Beta(mean (1/rho - 1), (1 - mean)(1/rho - 1))
    (q = mean when rho = 0): the chance that k distinct viewpoints all miss."""
    if rho == 0.0:
        return (1.0 - mean) ** np.arange(most + 1)
    scale = 1.0 / rho - 1.0
    a, b = mean * scale, (1.0 - mean) * scale
    out = np.ones(most + 1)
    for k in range(1, most + 1):
        out[k] = out[k - 1] * (b + k - 1) / (a + b + k - 1)
    return out


def line_of_sight(walls, range_m: float) -> Sees:
    """Visibility of the wall line-of-sight observation model (wall_los.py).
    ``sees.many(x, y, points)`` answers the same test for many points at once."""
    def sees(x, y, px, py):
        return (np.hypot(px - x, py - y) <= range_m
                and walls.first_hit((x, y), (px, py)) is None)

    def many(x, y, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        near = np.hypot(points[:, 0] - x, points[:, 1] - y) <= range_m
        out = np.zeros(len(points), dtype=bool)
        idx = np.flatnonzero(near)
        if len(idx):
            out[idx] = ~walls.blocked((x, y), points[idx])
        return out
    sees.many = many
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
                            spacing: float = 0.2, room_ids=None) -> dict:
    """Room segmentation grid (proposal 4-1, ``room_<i>`` = label i, or
    ``room_ids[i - 1]`` when the map names its rooms) -> floor samples per
    room, about ``spacing`` apart (row = +y, col = +x)."""
    step = max(1, int(round(spacing / resolution)))
    sub = labels[::step, ::step]
    rows, cols = np.nonzero(sub > 0)
    out = {}
    for label in np.unique(sub[rows, cols]):
        m = sub[rows, cols] == label
        xs = origin[0] + (cols[m] * step + 0.5) * resolution
        ys = origin[1] + (rows[m] * step + 0.5) * resolution
        name = f"room_{int(label)}" if room_ids is None else str(room_ids[int(label) - 1])
        out[name] = np.stack([xs, ys], axis=1)
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
