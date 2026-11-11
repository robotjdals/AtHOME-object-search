"""Navigation goal candidates around an object footprint (proposal 5-2)."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from athome.navigation.grid import Cell, GridMap
from athome.schemas import Pose2D


def goal_candidates(
    grid: GridMap,
    bbox_min_xy: Sequence[float],
    bbox_max_xy: Sequence[float],
    offset: float,
    clearance: float = 0.0,
    max_offset: Optional[float] = None,
    sight: Optional[GridMap] = None,
) -> List[Tuple[Cell, Pose2D]]:
    """Free cells on a one-cell band at ``offset`` from the XY bbox,
    each facing the bbox center.

    ``offset`` = robot footprint distance + safety margin. Cells already
    have the robot radius applied through ``grid.free``. ``clearance`` is the
    extra distance a goal cell must keep from the non-free area so the robot
    can rotate in place there (circumscribed - inscribed radius; the robot
    observes 360 degrees at every goal).

    With ``max_offset`` the band is the nearest one-cell ring at or beyond
    ``offset`` (and below ``max_offset``) that has a usable cell: the goal is
    as close as allowed, and moves out only where the robot cannot stand at
    ``offset`` (nearest reachable point not closer than a minimum standoff,
    as OK-Robot and HomeRobot choose base poses; 1.0 m is the object-search
    success radius of GOAT / Habitat ObjectNav).

    With ``sight`` (walls and partitions only; non-free cells block) a goal
    must also see the bbox: the segment to its nearest point crosses no wall,
    as HomeRobot OVMM viewpoints must see their receptacle. Furniture does
    not block: the camera looks over it.
    """
    lo = np.asarray(bbox_min_xy[:2], dtype=float)
    hi = np.asarray(bbox_max_xy[:2], dtype=float)
    if not np.isfinite([lo, hi]).all() or np.any(hi < lo):
        raise ValueError("잘못된 bbox")

    rows, cols, x, y = grid.cell_centers()
    # Euclidean distance from each cell center to the rectangle.
    dx = np.maximum(np.maximum(lo[0] - x, x - hi[0]), 0)
    dy = np.maximum(np.maximum(lo[1] - y, y - hi[1]), 0)
    distance = np.hypot(dx, dy)
    usable = distance >= offset
    if clearance > 0:
        usable &= grid.clearance()[rows, cols] >= clearance
    if sight is not None:
        reach = usable if max_offset is None else usable & (distance < max_offset)
        seen = np.zeros_like(usable)
        seen[reach] = in_sight(sight, x[reach], y[reach], lo, hi)
        usable &= seen
    band = usable & (distance < offset + grid.resolution)
    if max_offset is not None:
        usable &= distance < max_offset
        ring = np.floor((distance - offset) / grid.resolution)
        band = usable & (ring == ring[usable].min()) if usable.any() else usable

    cx, cy = (lo + hi) / 2
    out = []
    for i in np.flatnonzero(band):
        px, py = float(x[i]), float(y[i])
        out.append((
            (int(rows[i]), int(cols[i])),
            Pose2D(px, py, math.atan2(cy - py, cx - px)),
        ))
    return out


def in_sight(sight: GridMap, x: np.ndarray, y: np.ndarray, lo, hi) -> np.ndarray:
    """For each point (x, y): the segment to the nearest point of the XY box
    [lo, hi] crosses no non-free cell of ``sight`` (sampled every half cell;
    outside the map does not block). Samples within half a cell of the box
    are not checked: a box against a wall reaches into the wall's cells, and
    a segment from the open side ends there, while one from behind the wall
    still crosses it."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dx, dy = np.clip(x, lo[0], hi[0]) - x, np.clip(y, lo[1], hi[1]) - y
    length = np.hypot(dx, dy)
    ux, uy = dx / np.maximum(length, 1e-9), dy / np.maximum(length, 1e-9)
    ok = np.ones(x.shape, dtype=bool)
    step = sight.resolution / 2
    rows_n, cols_n = sight.free.shape
    s = 0.0
    while True:
        active = ok & (s < length)
        if not active.any():
            return ok
        px, py = x[active] + ux[active] * s, y[active] + uy[active] * s
        to_box = np.hypot(np.maximum(np.maximum(lo[0] - px, px - hi[0]), 0),
                          np.maximum(np.maximum(lo[1] - py, py - hi[1]), 0))
        r = np.floor((py - sight.origin[1]) / sight.resolution).astype(int)
        c = np.floor((px - sight.origin[0]) / sight.resolution).astype(int)
        inside = (r >= 0) & (r < rows_n) & (c >= 0) & (c < cols_n) & (to_box >= step)
        blocked = np.zeros(r.shape, dtype=bool)
        blocked[inside] = ~sight.free[r[inside], c[inside]]
        ok[np.flatnonzero(active)[blocked]] = False
        s += step


def segment_hits_box(x0: float, y0: float, x1: float, y1: float, lo, hi) -> bool:
    """The segment (x0, y0)-(x1, y1) crosses the XY box [lo, hi] (slab test)."""
    t0, t1 = 0.0, 1.0
    for start, end, low, high in ((x0, x1, lo[0], hi[0]), (y0, y1, lo[1], hi[1])):
        d = end - start
        if abs(d) < 1e-12:
            if start < low or start > high:
                return False
            continue
        a, b = (low - start) / d, (high - start) / d
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t0 > t1:
            return False
    return True


def view_blockers(lo, hi, others: Sequence) -> list:
    """Of the XY boxes ``others``, those that can stand between a viewer and
    the box [lo, hi]: not the ones sharing its place (holding its center, or
    at least half inside it: a hanger on the desk, a guitar against it)."""
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    kept = []
    for a, b in others:
        if a[0] <= cx <= b[0] and a[1] <= cy <= b[1]:
            continue
        w = min(hi[0], b[0]) - max(lo[0], a[0])
        h = min(hi[1], b[1]) - max(lo[1], a[1])
        area = (b[0] - a[0]) * (b[1] - a[1])
        if w > 0 and h > 0 and area > 0 and w * h / area >= 0.5:
            continue
        kept.append((a, b))
    return kept


def spread_goals(
    ordered: Sequence[Pose2D], max_goals: int, min_separation: float
) -> Tuple[Pose2D, ...]:
    """Keep the cheapest goals that are at least ``min_separation`` apart,
    so a retry approaches from a meaningfully different spot."""
    kept: List[Pose2D] = []
    for pose in ordered:
        if all(
            math.hypot(pose.x - k.x, pose.y - k.y) >= min_separation
            for k in kept
        ):
            kept.append(pose)
            if len(kept) == max_goals:
                break
    return tuple(kept)
