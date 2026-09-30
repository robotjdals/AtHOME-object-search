import math

import numpy as np
import pytest

from athome.navigation import (
    GridMap,
    NavigationConfig,
    NavigationPlanner,
    StartNotFree,
    load_map_server,
    load_occupancy,
    traversable_from_occupancy,
)
from athome.navigation.goal_poses import goal_candidates
from athome.navigation.path_cost import extract_path, shortest_paths
from athome.schemas import Pose2D
from athome.testing import toy_env


def open_grid(rows=20, cols=20, res=0.1):
    return GridMap(np.ones((rows, cols), bool), (0.0, 0.0), res)


def test_cell_conversion_roundtrip():
    grid = open_grid()
    cell = grid.to_cell(1.23, 0.47)
    assert cell == (4, 12)
    x, y = grid.to_xy(cell)
    assert grid.to_cell(x, y) == cell


def test_inflation_blocks_cells_near_obstacles():
    occ = np.zeros((20, 20), np.int16)
    occ[10, 10] = 100
    free = traversable_from_occupancy(occ, 0.1, inflation_radius=0.25)
    assert not free[10, 10]
    assert not free[10, 12]       # 0.2 m away
    assert free[10, 13]           # 0.3 m away


def test_unknown_cells_blocked_by_default():
    occ = np.full((5, 5), -1, np.int16)
    assert not traversable_from_occupancy(occ, 0.1, 0.0).any()
    assert traversable_from_occupancy(occ, 0.1, 0.0, unknown_as_occupied=False).all()


def test_shortest_path_cost_and_no_corner_cutting():
    free = np.ones((5, 5), bool)
    free[1, 1] = False
    reached, parent = shortest_paths(free, (0, 0), [(2, 2)], 1.0)
    # Every diagonal touching the blocked cell is forbidden: 4 straight moves.
    assert reached[(2, 2)] == pytest.approx(4.0)
    path = extract_path(parent, (0, 0), (2, 2))
    assert path[0] == (0, 0) and path[-1] == (2, 2)
    assert (1, 1) not in path


def test_goal_candidates_face_the_object():
    grid = open_grid(40, 40)
    cands = goal_candidates(grid, (1.8, 1.8), (2.2, 2.2), offset=0.5)
    assert cands
    for _, pose in cands:
        d = math.hypot(max(1.8 - pose.x, pose.x - 2.2, 0), max(1.8 - pose.y, pose.y - 2.2, 0))
        assert 0.5 <= d < 0.6
        heading = math.atan2(2.0 - pose.y, 2.0 - pose.x)
        assert pose.yaw == pytest.approx(heading)


def test_planner_prefers_near_location_and_spreads_goals():
    planner = NavigationPlanner(toy_env.toy_grid())
    planner.add_location("table", (1.0, 1.0), (2.0, 1.8))
    planner.add_location("shelf", (9.3, 0.5), (9.8, 2.5))
    costs = planner.evaluate(Pose2D(2.5, 3.0, 0.0), ["table", "shelf"])
    assert costs["table"].cost < costs["shelf"].cost
    goals = costs["table"].goals
    assert 1 < len(goals) <= NavigationConfig().max_goals_per_location
    for a in goals:
        for b in goals:
            if a is not b:
                assert math.hypot(a.x - b.x, a.y - b.y) >= NavigationConfig().goal_separation


def test_unreachable_location_is_left_out():
    free = np.ones((40, 40), bool)
    free[:, 20] = False           # split the map
    planner = NavigationPlanner(GridMap(free, (0.0, 0.0), 0.1))
    planner.add_location("far", (3.0, 1.8), (3.4, 2.2))
    assert planner.evaluate(Pose2D(0.5, 2.0, 0.0), ["far"]) == {}


def test_start_outside_free_space_raises():
    planner = NavigationPlanner(toy_env.toy_grid())
    planner.add_location("table", (1.0, 1.0), (2.0, 1.8))
    with pytest.raises(StartNotFree):
        planner.evaluate(Pose2D(1.5, 1.4, 0.0), ["table"])   # inside the table


def test_load_map_server(tmp_path):
    rows, cols = 4, 6
    image = np.full((rows, cols), 254, np.uint8)   # free
    image[0, 0] = 0                                # occupied, top-left pixel
    (tmp_path / "m.pgm").write_bytes(
        b"P5\n# test\n%d %d\n255\n" % (cols, rows) + image.tobytes())
    (tmp_path / "m.yaml").write_text(
        "image: m.pgm\nresolution: 0.5\norigin: [-1.0, 2.0, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    grid = load_map_server(tmp_path / "m.yaml", inflation_radius=0.0)
    assert grid.origin == (-1.0, 2.0)
    # Top image row is the highest y, i.e. the last grid row.
    assert not grid.free[rows - 1, 0]
    assert grid.free.sum() == rows * cols - 1


def test_load_occupancy_values(tmp_path):
    rows, cols = 4, 6
    image = np.full((rows, cols), 254, np.uint8)   # free
    image[0, 0] = 0                                # occupied, top-left pixel
    image[0, 1] = 205                              # unknown (map_saver gray)
    (tmp_path / "m.pgm").write_bytes(b"P5\n%d %d\n255\n" % (cols, rows) + image.tobytes())
    (tmp_path / "m.yaml").write_text(
        "image: m.pgm\nresolution: 0.5\norigin: [-1.0, 2.0, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    m = load_occupancy(tmp_path / "m.yaml")
    assert m.origin == (-1.0, 2.0) and m.resolution == 0.5
    assert m.occupancy[rows - 1, 0] == 100 and m.occupancy[rows - 1, 1] == -1
    assert (m.occupancy == 0).sum() == rows * cols - 2
