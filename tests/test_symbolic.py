import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from athome.navigation.grid import GridMap
from athome.schemas import Pose2D
from athome.search.session import SearchDecision
from athome.search.policy import Stage
from athome.scene_graph.query import SceneGraph, floor_references
from athome.symbolic import GroundTruthObject, NavmeshSurface, SymbolicEnvironment
from athome.testing import toy_env


class FakeObserver:
    """Returns fixed pixel counts per (x, y, yaw); records every call."""

    def __init__(self, visible=None):
        self.visible = visible or (lambda x, y, yaw: {})
        self.calls = []

    def observe(self, x, y, floor_z, yaw):
        self.calls.append((x, y, floor_z, yaw))
        return self.visible(x, y, yaw)


def flat(z=0.):
    return lambda x, y: z


def decision(cost, goal):
    return SearchDecision(1, "cup", Stage.STANDALONE, "R", "location", cost, (goal,))


def env_with(grid, start, objects=(), observer=None, heading_count=4, floor=flat()):
    return SymbolicEnvironment(grid, start, list(objects), observer or FakeObserver(),
                               floor, heading_count)


def test_observes_only_at_goal_headings_like_visit_executor():
    grid = GridMap(np.ones((3, 5), dtype=bool), (0., 0.), 1.)
    # cup_42 is visible only when facing -X (yaw pi) from the goal.
    observer = FakeObserver(lambda x, y, yaw: {42: 2000} if abs(abs(yaw) - math.pi) < 1e-9 else {})
    env = env_with(grid, Pose2D(.5, .5, 0), [
        GroundTruthObject("cup_42", "cup", 42, (2.5, .5, 1.)),
        GroundTruthObject("cup_43", "cup", 43, (2.5, 2.5, 1.)),
    ], observer, floor=flat(2.6))
    outcome = env.visit(decision(4., Pose2D(4.5, .5, 0)))
    assert env.distance_m == 4.
    assert env.last_path == [(0, c) for c in range(5)]
    # No observation while driving: every call is at the goal with its floor Z.
    assert {(x, y, z) for x, y, z, _ in observer.calls} == {(4.5, .5, 2.6)}
    assert [yaw for *_, yaw in observer.calls] == pytest.approx(
        [0., math.pi / 2, math.pi, -math.pi / 2])
    assert len(outcome.observations) == 4
    assert [o.label for f in outcome.observations for o in f.objects] == ["cup"]
    assert env.last_detections == [{
        "object_id": "cup_42", "heading_index": 2, "yaw": pytest.approx(math.pi),
        "evidence": 2000, "observation_tick": 3}]
    assert env.pose.x == 4.5


def test_headings_start_from_goal_yaw():
    grid = GridMap(np.ones((1, 3), dtype=bool), (0., 0.), 1.)
    observer = FakeObserver()
    env = env_with(grid, Pose2D(.5, .5, 0), observer=observer, heading_count=2)
    env.visit(decision(2., Pose2D(2.5, .5, math.pi / 2)))
    assert [yaw for *_, yaw in observer.calls] == pytest.approx([math.pi / 2, -math.pi / 2])


def test_symbolic_rejects_blocked_path_instead_of_teleporting():
    free = np.ones((3, 5), dtype=bool)
    free[:, 2] = False
    env = env_with(GridMap(free, (0., 0.), 1.), Pose2D(.5, .5, 0))
    with pytest.raises(ValueError, match="경로"):
        env.visit(decision(4., Pose2D(4.5, .5, 0)))
    assert env.distance_m == 0
    assert env.pose == Pose2D(.5, .5, 0)


def test_symbolic_no_corner_cutting_and_cost_mismatch():
    free = np.ones((3, 3), dtype=bool)
    free[0, 1] = False
    env = env_with(GridMap(free, (0., 0.), 1.), Pose2D(.5, .5, 0))
    with pytest.raises(ValueError, match="비용"):
        env.visit(decision(2**.5, Pose2D(1.5, 1.5, 0)))
    env.visit(decision(2., Pose2D(1.5, 1.5, 0)))
    assert env.last_path == [(0, 0), (1, 0), (1, 1)]


def test_invalid_environment_arguments():
    grid = GridMap(np.ones((2, 2), bool), (0., 0.), 1.)
    with pytest.raises(ValueError):
        env_with(grid, Pose2D(.5, .5, 0), heading_count=0)
    with pytest.raises(ValueError, match="semantic"):
        env_with(grid, Pose2D(.5, .5, 0), [GroundTruthObject("a_1", "cup", 1, (0., 0., 0.)),
                                           GroundTruthObject("b_1", "cup", 1, (0., 0., 0.))])


def test_navmesh_surface_height():
    vertices = [[0., 0., 1.], [2., 0., 1.], [0., 2., 3.], [2., 2., 3.]]
    surface = NavmeshSurface(vertices, [[0, 1, 2], [1, 3, 2]])
    assert surface.height_at(1., 1.) == pytest.approx(2.)  # shared edge
    assert surface.height_at(.5, .5) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="밖"):
        surface.height_at(3., 3.)
    stacked = NavmeshSurface(vertices + [[0., 0., 5.], [2., 0., 5.], [0., 2., 5.]],
                             [[0, 1, 2], [4, 5, 6]])
    with pytest.raises(ValueError, match="층"):
        stacked.height_at(.5, .5)


def test_floor_reference_independent_of_low_nonfloor_object():
    data = toy_env.toy_graph()
    for room in data["rooms"]:
        room.pop("floor_z_m")
    data["objects"].append({"object_id": "floor_test", "room_id": "R_A",
                            "semantic_tag": "floor", "role": "standalone",
                            "bbox": {"min": [0., 0., -.1], "max": [5., 5., .1]}})
    data["objects"][0]["bbox"]["min"][2] = -100.
    assert floor_references(data) == {"R_A": 0.}
    assert SceneGraph(data).room_floor_z == {"R_A": 0.}


def test_masked_episode_and_absent_episode_use_same_planner():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("run_symbolic_review", scripts / "run_symbolic_review.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        grid = toy_env.toy_grid()
        cell = grid.to_cell(2.5, 3.)
        start = Pose2D(*grid.to_xy(cell), 0.)
        graph = SceneGraph(toy_env.toy_graph(hidden_categories=("cup",)))
        world = [GroundTruthObject("hidden_cup", "cup", 7, (start.x, start.y, 1.))]
        # The hidden cup is in view from every reached goal.
        found = runner.run_episode(graph, grid, start, world, "cup", .45, 30,
                                   FakeObserver(lambda x, y, yaw: {7: 5000}), flat(), 4)
        absent = runner.run_episode(graph, grid, start, [], "cup", .45, 30,
                                    FakeObserver(), flat(), 4)
        assert found["target_status"] == "found"
        assert found["found_object_id"] == "hidden_cup"
        assert absent["target_status"] == "failed"
        assert found["steps"][0]["location_id"] == absent["steps"][0]["location_id"]
        assert found["steps"][0]["stage"] != "known"
        assert "hidden_cup" not in graph.locations
    finally:
        sys.path.remove(str(scripts))


def test_isotropic_observer_called_once_per_visit():
    grid = GridMap(np.ones((1, 3), dtype=bool), (0., 0.), 1.)
    observer = FakeObserver(lambda x, y, yaw: {7: 1.2})
    observer.per_heading = False
    env = env_with(grid, Pose2D(.5, .5, 0), [GroundTruthObject("cup_7", "cup", 7, (2., 1., 1.))],
                   observer)
    outcome = env.visit(decision(2., Pose2D(2.5, .5, 0)))
    assert len(observer.calls) == 1 and len(outcome.observations) == 1
    assert env.last_detections[0]["heading_index"] is None
    assert env.last_detections[0]["evidence"] == 1.2
