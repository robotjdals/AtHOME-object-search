import math

import numpy as np
import pytest

from athome.config import goal_geometry, robot_geometry
from athome.navigation.goal_poses import goal_candidates
from athome.navigation.grid import GridMap


def test_tidybot_radii_and_goal_offset():
    raw = {"robot": {"footprint_m": [0.50, 0.54]}, "navigation": {"goal_safety_margin": 0.10}}
    geometry, inflation = robot_geometry(raw["robot"])
    assert inflation == pytest.approx(0.25)                       # inscribed
    assert geometry.circumscribed_radius == pytest.approx(math.hypot(0.25, 0.27))
    offset, clearance, max_offset = goal_geometry(raw)
    assert max_offset is None
    assert offset == pytest.approx(math.hypot(0.25, 0.27) + 0.10)
    assert clearance == pytest.approx(math.hypot(0.25, 0.27) - 0.25)


def test_inconsistent_inflation_radius_is_rejected():
    with pytest.raises(ValueError):
        robot_geometry({"footprint_m": [0.50, 0.54], "inflation_radius": 0.30})


def test_circular_config_is_unchanged():
    raw = {"robot": {"inflation_radius": 0.25}, "navigation": {"goal_offset": 0.45}}
    assert robot_geometry(raw["robot"]) == (None, 0.25)
    assert goal_geometry(raw) == (0.45, 0.0, None)


def test_goal_cells_need_room_to_turn():
    free = np.zeros((40, 40), dtype=bool)
    free[5:35, 5:35] = True
    free[18:22, 5:12] = False                                      # a box on the left
    grid = GridMap(free, (0.0, 0.0), 0.05)
    all_cells = goal_candidates(grid, (0.25, 0.9), (0.6, 1.1), 0.2)
    turnable = goal_candidates(grid, (0.25, 0.9), (0.6, 1.1), 0.2, clearance=0.12)
    assert turnable and len(turnable) < len(all_cells)
    assert all(grid.clearance()[cell] >= 0.12 for cell, _ in turnable)


def test_goal_moves_out_only_where_blocked():
    free = np.zeros((60, 60), dtype=bool)
    free[5:55, 5:55] = True
    grid = GridMap(free, (0.0, 0.0), 0.05)
    # Open on all sides: the ring at the offset is used.
    near = goal_candidates(grid, (1.2, 1.2), (1.4, 1.4), 0.3, max_offset=1.0)
    assert near == goal_candidates(grid, (1.2, 1.2), (1.4, 1.4), 0.3)
    # Box against the lower-left corner: nothing at the offset; the nearest usable ring is taken.
    boxed = GridMap(free & ~np.pad(np.ones((12, 12), bool), ((5, 43), (5, 43))), (0.0, 0.0), 0.05)
    assert goal_candidates(boxed, (0.25, 0.25), (0.8, 0.8), 0.05, clearance=0.4) == []
    moved = goal_candidates(boxed, (0.25, 0.25), (0.8, 0.8), 0.05, clearance=0.4, max_offset=1.0)
    assert moved
    rings = {round(float(np.hypot(max(p.x - 0.8, 0), max(p.y - 0.8, 0))), 1) for _, p in moved}
    assert min(rings) >= 0.05 and len({int((r - 0.05) // 0.05) for r in rings}) <= 2
