from athome.execution.visit import VisitOutcome, VisitStatus
from athome.navigation import NavigationPlanner
from athome.scene_graph.query import SceneGraph
from athome.schemas import ObservationFrame, ObservedObject, Pose2D
from athome.search import SearchSession, SessionStatus, Stage, TargetStatus
from athome.search.policy import PolicyError
from athome.testing import toy_env

START = Pose2D(2.5, 3.0, 0.0)
TABLE = "workspace:R_A:table_1"
COUNTER = "workspace:R_A:counter_4"
FRIDGE = "standalone:fridge_6"
COFFEE = "workspace:R_B:coffee table_7"
SHELF = "workspace:R_B:shelf_9"

GRID = toy_env.toy_grid()


def session(targets, hide=(), **kwargs):
    graph = SceneGraph(toy_env.toy_graph(hide))
    return SearchSession(graph, NavigationPlanner(GRID), targets, **kwargs)


def observed(*labels):
    frame = ObservationFrame(
        1.0, tuple(ObservedObject(i, l, 0.9, (0, 0, 0)) for i, l in enumerate(labels)))
    return [frame]


def run(s, world, failing=()):
    """world: location_id -> labels seen there. Returns visited order."""
    pose, order = START, []
    while True:
        d = s.next_decision(pose)
        if d is None:
            return order
        order.append((d.target, d.stage, d.location_id))
        if d.location_id in failing:
            s.report(d, VisitOutcome(d.location_id, VisitStatus.FAILED))
            continue
        s.report(d, VisitOutcome(
            d.location_id, VisitStatus.COMPLETED,
            observations=observed(*world.get(d.location_id, ()))))
        pose = d.goals[0]


def test_known_target_goes_to_graph_location():
    s = session(["cup"])
    order = run(s, {TABLE: ["cup", "plate"]})
    assert order == [("cup", Stage.KNOWN, TABLE)]
    assert s.targets[0].status == TargetStatus.FOUND
    assert s.status == SessionStatus.DONE


def test_known_target_not_there_switches_to_unknown_without_revisit():
    s = session(["cup"])
    order = run(s, {COFFEE: ["cup"]})
    assert order[0] == ("cup", Stage.KNOWN, TABLE)
    assert s.targets[0].switched_to_unknown
    assert [o[2] for o in order].count(TABLE) == 1
    assert order[-1][2] == COFFEE
    assert s.targets[0].found_location == COFFEE


def test_unknown_searches_workspaces_before_standalone_room_by_room():
    s = session(["remote"], hide=["remote"])
    assert not s.targets[0].known
    order = run(s, {COFFEE: ["remote"]})
    assert [o[2] for o in order] == [TABLE, COUNTER, FRIDGE, COFFEE]
    assert [o[1] for o in order] == [
        Stage.WORKSPACE, Stage.WORKSPACE, Stage.STANDALONE, Stage.WORKSPACE]


def test_other_targets_found_on_the_way():
    s = session(["remote", "kettle"], hide=["remote", "kettle"])
    order = run(s, {COUNTER: ["kettle"], COFFEE: ["remote"]})
    assert s.targets[1].status == TargetStatus.FOUND
    assert s.targets[1].found_location == COUNTER
    # No step was spent on kettle itself.
    assert all(t == "remote" for t, _, _ in order)


def test_visited_is_shared_across_targets():
    s = session(["remote", "banana"], hide=["remote"])
    order = run(s, {COFFEE: ["remote"]})
    locations = [o[2] for o in order]
    assert len(locations) == len(set(locations))
    assert s.targets[1].status == TargetStatus.FAILED


def test_absent_target_fails_after_every_location():
    s = session(["banana"])
    order = run(s, {})
    assert s.targets[0].status == TargetStatus.FAILED
    assert set(o[2] for o in order) | s.covered == set(s.graph.locations)
    # Wall is not a search location.
    assert "standalone:wall_13" not in s.graph.locations


def test_ceiling_mounted_standalone_is_not_a_location():
    graph = toy_env.toy_graph()
    graph["objects"].append({
        "object_id": "ceiling fan_14", "room_id": "R_B", "semantic_tag": "ceiling fan",
        "bbox": {"min": [7.0, 3.0, 2.3], "max": [7.6, 3.6, 2.5]}, "role": "standalone",
    })
    assert "standalone:ceiling fan_14" not in SceneGraph(graph).locations
    assert "standalone:ceiling fan_14" in SceneGraph(
        graph, max_standalone_base_height=3.0).locations


def completed_at(d, pose):
    return VisitOutcome(d.location_id, VisitStatus.COMPLETED, final_pose=pose)


def test_nearby_standalone_counts_as_observed():
    s = session(["banana"])
    d = s.next_decision(START)
    # Observation pose next to the lamp (center 8.7, 5.2).
    record = s.report(d, completed_at(d, Pose2D(8.2, 4.6, 0.0)))
    assert record.covered == ["standalone:lamp_11"]
    assert "standalone:lamp_11" not in s.visited
    locations = [o[2] for o in run(s, {})]
    assert "standalone:lamp_11" not in locations


def test_workspaces_are_never_covered_and_coverage_can_be_disabled():
    s = session(["banana"])
    d = s.next_decision(START)
    # Right next to the coffee table workspace.
    s.report(d, completed_at(d, Pose2D(7.0, 1.6, 0.0)))
    assert COFFEE not in s.covered

    s = session(["banana"], standalone_coverage_radius=0.0)
    d = s.next_decision(START)
    assert s.report(d, completed_at(d, Pose2D(8.2, 4.6, 0.0))).covered == []


def test_failed_visit_is_excluded_not_visited():
    s = session(["remote"], hide=["remote"])
    order = run(s, {COFFEE: ["remote"]}, failing={TABLE})
    assert TABLE in s.excluded and TABLE not in s.visited
    assert [o[2] for o in order].count(TABLE) == 1
    assert s.targets[0].status == TargetStatus.FOUND


def test_max_steps_stops_the_command():
    s = session(["banana"], max_steps=2)
    order = run(s, {})
    assert len(order) == 2
    assert s.status == SessionStatus.MAX_STEPS
    assert s.targets[0].detail == "최대 탐색 횟수 도달"


class BadPolicy:
    def select(self, stage, target, candidates, context):
        return "not-a-candidate"


class BrokenPolicy:
    def select(self, stage, target, candidates, context):
        raise PolicyError("timeout")


def test_policy_output_outside_candidates_is_not_applied():
    s = session(["remote"], hide=["remote"], policy=BadPolicy())
    order = run(s, {TABLE: ["remote"]})
    assert order[0][2] == TABLE
    assert s.history[0].policy_fallback
    assert "not-a-candidate" in s.history[0].fallback_reason


def test_policy_error_falls_back_to_min_cost():
    s = session(["remote"], hide=["remote"], policy=BrokenPolicy())
    order = run(s, {TABLE: ["remote"]})
    assert order[0][2] == TABLE
    assert s.history[0].fallback_reason == "timeout"


class RecordingPolicy:
    def __init__(self):
        self.calls = []

    def select(self, stage, target, candidates, context):
        self.calls.append((stage, context))
        return min(candidates, key=lambda c: c.cost).candidate_id


def test_policy_receives_planning_context():
    policy = RecordingPolicy()
    s = session(["remote"], hide=["remote"], policy=policy)
    run(s, {COFFEE: ["remote"]})
    stage, ctx = policy.calls[0]
    assert stage == Stage.ROOM and ctx.room_id is None and ctx.visited_count == 0
    stage, ctx = policy.calls[1]
    assert stage == Stage.WORKSPACE and ctx.room_label == "kitchen"
    last_stage, last_ctx = policy.calls[-1]
    assert last_ctx.explored_rooms == ("R_A",)
    assert last_ctx.last_location == FRIDGE


def test_paused_visit_does_not_change_state():
    s = session(["cup"])
    d = s.next_decision(START)
    s.report(d, VisitOutcome(d.location_id, VisitStatus.PAUSED))
    assert s.steps == 0 and not s.visited and not s.excluded
    assert s.next_decision(START).location_id == d.location_id
