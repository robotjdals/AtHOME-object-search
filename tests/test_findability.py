from athome.navigation import NavigationPlanner
from athome.scene_graph.query import SceneGraph
from athome.schemas import Pose2D
from athome.testing import toy_env
from athome.training.findability import findable_from

START = Pose2D(2.5, 3.0, 0.0)


class SeesNear:
    """Observes target 7 from goal poses close to (x, y)."""
    per_heading = False

    def __init__(self, x, y, radius=1.0):
        self.x, self.y, self.radius = x, y, radius

    def observe(self, x, y, floor_z, yaw):
        return {7: 1.0} if (x - self.x) ** 2 + (y - self.y) ** 2 <= self.radius ** 2 else {}


def planner(graph):
    nav = NavigationPlanner(toy_env.toy_grid())
    for lid, loc in graph.locations.items():
        nav.add_location(lid, loc.bbox_min, loc.bbox_max)
    return nav


def test_findable_when_a_reachable_goal_sees_the_target():
    graph = SceneGraph(toy_env.toy_graph(["remote"]))
    nav = planner(graph)
    lid, (cell, pose) = next((l, g[0]) for l in sorted(graph.locations)
                             if (g := nav.candidate_goals(l)))
    assert findable_from(nav, graph.locations, START, SeesNear(pose.x, pose.y, 0.01), [7]) is not None


def test_not_findable_when_no_goal_sees_it():
    graph = SceneGraph(toy_env.toy_graph(["remote"]))
    nav = planner(graph)
    assert findable_from(nav, graph.locations, START, SeesNear(-100.0, -100.0), [7]) is None
    assert findable_from(nav, graph.locations, START, SeesNear(0.0, 0.0, 1e9), []) is None
