"""Observation headings at a search location (absolute yaws, in the order
they are looked at). ``full``: the turn of the training observation model
(heading_count headings from the goal yaw). ``front``: the goal yaw and its
neighbours ``step`` apart. ``cover``: the fewest headings ``step`` apart
whose camera views (``hfov``) cover the location's footprint as seen from
the goal pose; the goal pose faces the location, so the first heading needs
no turn for small locations.
"""

from __future__ import annotations

import math
from typing import List, Sequence

from athome.execution.visit import wrap_angle
from athome.schemas import Pose2D

STEP = math.radians(60.0)


def full(goal: Pose2D, count: int = 6) -> List[float]:
    return [wrap_angle(goal.yaw + 2 * math.pi * k / count) for k in range(count)]


def front(goal: Pose2D, count: int = 3, step: float = STEP) -> List[float]:
    """The goal yaw, then alternately right and left of it (yaw grows to the
    left, REP 103)."""
    offsets = [0.0] + [s * step * k for k in range(1, count) for s in (-1, 1)][:count - 1]
    return [wrap_angle(goal.yaw + o) for o in offsets]


def front_first(goal: Pose2D, count: int = 6) -> List[float]:
    """All ``count`` headings (the training model's full turn), the ones
    facing the location first: the goal yaw, then right up to a quarter of
    the headings (6: one step, 8: two), back to the left and on round through
    the back to the rest (yaw grows to the left). With an early stop the
    location is checked first; a full turn is always 360 degrees.
    8 headings: 0, R45, R90, L45, L90, L135, 180, R135."""
    step = 2 * math.pi / count
    right = count // 4
    offsets = [0.0] + [-k * step for k in range(1, right + 1)] + [k * step for k in range(1, count - right)]
    return [wrap_angle(goal.yaw + o) for o in offsets]


def _sweep(goal_yaw: float, offsets: Sequence[float]) -> List[float]:
    """Offsets sorted, started from the end nearer the goal yaw (one sweep)."""
    offsets = sorted(offsets)
    if abs(offsets[0]) > abs(offsets[-1]):
        offsets = offsets[::-1]
    return [wrap_angle(goal_yaw + o) for o in offsets]


def cover(goal: Pose2D, bbox_min, bbox_max, hfov: float, step: float = STEP,
          max_count: int = 6) -> List[float]:
    """Headings covering the footprint's corners as seen from ``goal``."""
    (x0, y0), (x1, y1) = bbox_min[:2], bbox_max[:2]
    if x0 <= goal.x <= x1 and y0 <= goal.y <= y1:
        return full(goal, max_count)
    bearings = [wrap_angle(math.atan2(y - goal.y, x - goal.x) - goal.yaw)
                for x in (x0, x1) for y in (y0, y1)]
    # The footprint is convex and does not contain the robot, so its corners
    # span less than pi: the span is between the extreme bearings.
    lo, hi = min(bearings), max(bearings)
    if hi - lo > math.pi:                   # the span crosses +-pi (behind the robot)
        shifted = [b + 2 * math.pi if b < 0 else b for b in bearings]
        lo, hi = min(shifted), max(shifted)
    span = hi - lo
    if span <= hfov:
        mid = (lo + hi) / 2
        # Facing the footprint already covers it: no turn.
        if abs(wrap_angle(mid)) + span / 2 <= hfov / 2:
            return [wrap_angle(goal.yaw)]
        return [wrap_angle(goal.yaw + mid)]
    count = min(max_count, math.ceil((span - hfov) / step) + 1)
    if count >= max_count:
        return full(goal, max_count)
    mid = (lo + hi) / 2
    return _sweep(goal.yaw, [mid + (k - (count - 1) / 2) * step for k in range(count)])
