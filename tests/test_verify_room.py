from athome.navigation import NavigationPlanner
from athome.scene_graph.query import SceneGraph
from athome.schemas import Pose2D
from athome.testing import toy_env
from athome.training.verify import GroundTruth

START = Pose2D(2.5, 3.0, 0.0)


class SeesNothing:
    per_heading = False

    def observe(self, x, y, floor_z, yaw):
        return {}


def truth(rule, gt_rooms):
    graph = SceneGraph(toy_env.toy_graph(["remote"]))
    nav = NavigationPlanner(toy_env.toy_grid())
    for lid, loc in graph.locations.items():
        nav.add_location(lid, loc.bbox_min, loc.bbox_max)
    return GroundTruth(graph, nav, SeesNothing(), [7], [], gt_rooms=gt_rooms, room_rule=rule)


def test_room_is_correct_when_it_contains_the_target():
    # The target is in R_B but seen from nowhere: containment still marks R_B.
    check = truth("containment", {"R_B"}).verify("room", "R_B", ["R_A", "R_B"], START, set())
    assert check["passed"] and check["valid_candidates"] == ["R_B"]
    wrong = truth("containment", {"R_B"}).verify("room", "R_A", ["R_A", "R_B"], START, set())
    assert not wrong["passed"]


def test_observation_rule_is_kept_for_comparison():
    check = truth("observation", {"R_B"}).verify("room", "R_B", ["R_A", "R_B"], START, set())
    assert not check["passed"] and check["valid_candidates"] == []
