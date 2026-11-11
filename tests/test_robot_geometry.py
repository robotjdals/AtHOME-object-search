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


def _dist(p, lo, hi):
    return math.hypot(max(lo[0] - p.x, 0, p.x - hi[0]), max(lo[1] - p.y, 0, p.y - hi[1]))


def test_goals_behind_a_wall_are_left_out():
    grid = GridMap(np.ones((80, 80), dtype=bool), (0.0, 0.0), 0.05)
    walls = np.ones((80, 80), dtype=bool)
    walls[:, 30] = False                                           # a partition at x = 1.5 m
    sight = GridMap(walls, (0.0, 0.0), 0.05)
    lo, hi = (1.8, 1.8), (2.2, 2.2)
    every = goal_candidates(grid, lo, hi, 0.5)
    seen = goal_candidates(grid, lo, hi, 0.5, sight=sight)
    assert any(p.x < 1.5 for _, p in every)                        # the ring reaches past the partition
    assert seen and all(p.x > 1.5 for _, p in seen)
    assert len(seen) < len(every)


def test_sight_does_not_reject_a_box_against_a_wall():
    grid = GridMap(np.ones((60, 60), dtype=bool), (0.0, 0.0), 0.05)
    walls = np.ones((60, 60), dtype=bool)
    walls[:, 40:42] = False                                        # wall cells at x 2.0-2.1 m
    sight = GridMap(walls, (0.0, 0.0), 0.05)
    lo, hi = (1.6, 1.0), (2.05, 1.6)                               # the box ends inside the wall
    seen = goal_candidates(grid, lo, hi, 0.5, sight=sight)
    assert seen and all(p.x < 2.0 for _, p in seen)


def test_standalone_band_first_then_the_general_one():
    from athome.navigation.planner import NavigationConfig, NavigationPlanner
    free = np.ones((100, 100), dtype=bool)
    config = NavigationConfig(goal_offset=1.0, goal_max_offset=1.5,
                              standalone_goal_offset=1.5, standalone_goal_max_offset=2.0)
    planner = NavigationPlanner(GridMap(free, (0.0, 0.0), 0.05), config)
    lo, hi = (2.3, 2.3), (2.7, 2.7)
    planner.add_location("guitar", lo, hi, standalone=True)
    planner.add_location("desk", lo, hi)
    assert all(1.5 <= _dist(p, lo, hi) < 1.55 for _, p in planner.candidate_goals("guitar"))
    assert all(1.0 <= _dist(p, lo, hi) < 1.05 for _, p in planner.candidate_goals("desk"))
    # Room for 1.0 m but not 1.5 m around the object: the general band.
    small = np.zeros((100, 100), dtype=bool)
    small[28:72, 28:72] = True                                     # 1.27 m to the corners
    cramped = NavigationPlanner(GridMap(small, (0.0, 0.0), 0.05), config)
    cramped.add_location("guitar", lo, hi, standalone=True)
    goals = cramped.candidate_goals("guitar")
    assert goals and all(1.0 <= _dist(p, lo, hi) < 1.5 for _, p in goals)


def test_standalone_band_is_validated():
    from athome.config import standalone_goal_band
    raw = {"robot": {"inflation_radius": 0.25}, "navigation": {"goal_offset": 1.0}}
    assert standalone_goal_band(raw) == {}
    raw["navigation"].update(standalone_goal_offset=1.5, standalone_goal_max_offset=2.0)
    assert standalone_goal_band(raw) == {"standalone_goal_offset": 1.5, "standalone_goal_max_offset": 2.0}
    raw["navigation"]["standalone_goal_offset"] = 0.8
    with pytest.raises(ValueError):
        standalone_goal_band(raw)


def test_goals_stay_in_the_room_where_it_has_any():
    from athome.navigation.planner import NavigationConfig, NavigationPlanner
    from athome.navigation.rooms import RoomMap
    free = np.ones((100, 100), dtype=bool)
    labels = np.ones((100, 100), dtype=int)                        # room_1: x < 3 m
    labels[:, 60:] = 2                                             # room_2: x >= 3 m
    rooms = RoomMap(labels, (0.0, 0.0), 0.05)
    assert rooms.room_at(1.0, 1.0) == "room_1" and rooms.room_at(4.0, 1.0) == "room_2"
    assert rooms.room_at(-1.0, 1.0) is None
    config = NavigationConfig(goal_offset=1.0, goal_max_offset=1.5,
                              standalone_goal_offset=2.0, standalone_goal_max_offset=2.5)
    planner = NavigationPlanner(GridMap(free, (0.0, 0.0), 0.05), config, rooms=rooms)
    planner.add_location("guitar", (2.3, 2.3), (2.7, 2.7), True, "room_1")
    goals = [p for _, p in planner.candidate_goals("guitar")]
    assert goals and all(p.x < 3.0 for p in goals)
    assert any(1.9 < _dist(p, (2.3, 2.3), (2.7, 2.7)) for p in goals)   # the far band, inside the room
    # No goal in the room at all: any room, not none.
    elsewhere = NavigationPlanner(GridMap(free, (0.0, 0.0), 0.05), config, rooms=rooms)
    elsewhere.add_location("hall object", (2.3, 2.3), (2.7, 2.7), True, "room_9")
    assert elsewhere.candidate_goals("hall object")
    # Room for the near band only: inside the room, before the far band outside it.
    narrow = np.zeros((100, 100), dtype=int)
    narrow[30:70, 30:60] = 1                                       # room_1: nothing 2 m away inside it
    narrow[:, 60:] = 2
    near = NavigationPlanner(GridMap(free, (0.0, 0.0), 0.05), config, rooms=RoomMap(narrow, (0.0, 0.0), 0.05))
    near.add_location("guitar", (2.3, 2.3), (2.7, 2.7), True, "room_1")
    goals = [p for _, p in near.candidate_goals("guitar")]
    assert goals and all(1.5 <= p.x < 3.0 and 1.5 <= p.y < 3.5 and _dist(p, (2.3, 2.3), (2.7, 2.7)) < 1.5 for p in goals)


def test_goals_across_other_furniture_come_last():
    from athome.navigation.goal_poses import segment_hits_box
    from athome.navigation.planner import NavigationConfig, NavigationPlanner
    assert segment_hits_box(0, 0, 2, 2, (0.9, 0.9), (1.1, 1.1))
    assert not segment_hits_box(0, 0, 2, 0, (0.9, 0.9), (1.1, 1.1))
    assert not segment_hits_box(0, 0, 0.5, 0.5, (0.9, 0.9), (1.1, 1.1))     # the segment ends before it
    config = NavigationConfig(goal_offset=1.0, goal_max_offset=1.5, goal_clear_view=True)
    grid = GridMap(np.ones((100, 100), dtype=bool), (0.0, 0.0), 0.05)
    lo, hi = (2.3, 2.3), (2.7, 2.7)
    planner = NavigationPlanner(grid, config)
    planner.add_location("pillow desk", lo, hi, others=[((3.0, 2.0), (3.3, 3.0))])     # another desk on its right
    goals = [p for _, p in planner.candidate_goals("pillow desk")]
    assert goals and not any(segment_hits_box(p.x, p.y, 2.5, 2.5, (3.0, 2.0), (3.3, 3.0)) for p in goals)
    # Furniture all around: goals anyway.
    around = NavigationPlanner(grid, config)
    around.add_location("ringed", lo, hi, others=[((1.0, 1.0), (4.0, 4.0))])
    assert around.candidate_goals("ringed")
    # Off by default.
    off = NavigationPlanner(grid, NavigationConfig(goal_offset=1.0, goal_max_offset=1.5))
    off.add_location("pillow desk", lo, hi, others=[((3.0, 2.0), (3.3, 3.0))])
    assert any(segment_hits_box(p.x, p.y, 2.5, 2.5, (3.0, 2.0), (3.3, 3.0)) for _, p in off.candidate_goals("pillow desk"))


def test_furniture_sharing_the_place_does_not_block_the_view():
    from athome.navigation.goal_poses import view_blockers
    lo, hi = (0.0, 0.0), (1.0, 0.6)
    hanger = ((0.1, 0.1), (0.5, 0.4))          # inside the desk
    big_chair = ((-0.2, -0.2), (1.2, 0.9))     # oversized box around the desk's center
    next_desk = ((0.9, -1.0), (1.8, 0.3))      # touches a corner, stands beside it
    assert view_blockers(lo, hi, [hanger, big_chair, next_desk]) == [next_desk]
