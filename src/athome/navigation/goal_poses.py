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
