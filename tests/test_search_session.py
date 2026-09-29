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
    assert set(o[2] for o in order) == set(s.graph.locations)
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


def test_nearby_standalone_remains_open_until_visited():
    s = session(["banana"])
    d = s.next_decision(START)
    s.report(d, completed_at(d, Pose2D(8.2, 4.6, 0.0)))
    assert s.visited == {d.location_id}
    assert "standalone:lamp_11" in [o[2] for o in run(s, {})]


def test_failed_visit_remains_open_and_counts_as_attempt():
    s = session(["remote"], hide=["remote"])
    d = s.next_decision(START)
    s.report(d, VisitOutcome(d.location_id, VisitStatus.FAILED))
    assert not s.visited and s.steps == 1
    assert not hasattr(s, "excluded") and not hasattr(s, "covered")
    assert s.next_decision(START).location_id == d.location_id


def test_observation_on_failed_visit_finds_target_without_marking_visited():
    s = session(["cup", "remote"], hide=["remote"])
    d = s.next_decision(START)
    record = s.report(d, VisitOutcome(
        d.location_id, VisitStatus.FAILED, observations=observed("remote")))
    assert record.found == ["remote"]
    assert not s.visited and s.targets[1].status == TargetStatus.FOUND


def test_observation_on_pause_is_retained_without_consuming_step():
    s = session(["cup"])
    d = s.next_decision(START)
    s.report(d, VisitOutcome(d.location_id, VisitStatus.PAUSED,
                            observations=observed("cup")))
    assert s.targets[0].status == TargetStatus.FOUND
    assert s.steps == 0 and not s.visited


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
    assert s.steps == 0 and not s.visited
    assert s.next_decision(START).location_id == d.location_id


def test_observed_location_objects_count_as_searched():
    # Map IDs 0/1 stand for the fridge and the counter seen from the table.
    ids = {0: "fridge_6", 1: "counter_4"}
    s = session(["remote"], hide=["remote"], observed_object_ids=ids.get)
    order = run(s, {TABLE: ["fridge", "counter"], COFFEE: ["remote"]})
    assert [o[2] for o in order] == [TABLE, COFFEE]
    assert s.history[0].covered == [FRIDGE, COUNTER]
    assert {FRIDGE, COUNTER} <= s.visited


def test_without_resolver_observations_do_not_cover_locations():
    s = session(["remote"], hide=["remote"])
    order = run(s, {TABLE: ["fridge", "counter"], COFFEE: ["remote"]})
    assert [o[2] for o in order] == [TABLE, COUNTER, FRIDGE, COFFEE]


def test_workspace_only_locations():
    graph = SceneGraph(toy_env.toy_graph(), standalone_locations=False)
    assert FRIDGE not in graph.locations and TABLE in graph.locations


def test_time_budget_stops_the_search():
    clock = {"t": 0.0}
    s = session(["banana"], time_budget_s=10.0, elapsed_s=lambda: clock["t"])
    pose, visits = START, 0
    while True:
        d = s.next_decision(pose)
        if d is None:
            break
        s.report(d, VisitOutcome(d.location_id, VisitStatus.COMPLETED, observations=observed()))
        pose, visits = d.goals[0], visits + 1
        clock["t"] += 4.0          # each visit takes 4 s
    assert visits == 3
    assert s.status == SessionStatus.TIME_BUDGET
    assert s.targets[0].detail == "시간 예산 초과"


def test_known_target_outside_search_locations_goes_to_the_object():
    # A lamp is not a Search Location under the mpcat40 policy, but a Known
    # lamp is still reached directly; unknown-search candidates are unchanged.
    graph = SceneGraph(toy_env.toy_graph(), excluded_categories={"wall", "lamp"})
    assert "standalone:lamp_11" not in graph.locations
    assert graph.known_locations("lamp") == ["object:lamp_11"]
    s = SearchSession(graph, NavigationPlanner(GRID), ["lamp"])
    order = run(s, {"object:lamp_11": ["lamp"]})
    assert order == [("lamp", Stage.KNOWN, "object:lamp_11")]
    assert s.targets[0].status == TargetStatus.FOUND


class PreferRoomB:
    """Room stage: R_A first, then R_B; locations: nearest."""
    def __init__(self):
        self.rooms = 0

    def select(self, stage, target, candidates, context):
        if stage == Stage.ROOM:
            self.rooms += 1
            ids = [c.candidate_id for c in candidates]
            return "R_A" if self.rooms == 1 else ("R_B" if "R_B" in ids else ids[0])
        return min(candidates, key=lambda c: (c.cost, c.candidate_id)).candidate_id


def test_room_is_reselected_after_every_visit():
    s = session(["remote"], hide=["remote"], policy=PreferRoomB())
    order = run(s, {COFFEE: ["remote"]})
    # One visit in R_A, then the planner moves to R_B although R_A has open locations.
    assert order[0][2] == TABLE
    assert order[1][2] not in (COUNTER, FRIDGE)          # left R_A with open locations
    assert s.targets[0].status == TargetStatus.FOUND


def test_commit_to_room_keeps_the_room_until_exhausted():
    s = session(["remote"], hide=["remote"], policy=PreferRoomB(), commit_to_room=True)
    order = run(s, {COFFEE: ["remote"]})
    assert [o[2] for o in order][:3] == [TABLE, COUNTER, FRIDGE]


def test_grpo_rollout_gives_the_room_stage_observed_fraction():
    from athome.search.coverage import RoomCoverage
    from athome.search.policy import MinCostPolicy
    from athome.training.grpo import rollout

    class Env:
        pose, distance_m, visits = START, 0.0, 0

        def visit(self, d):
            self.pose, self.visits = d.goals[0], self.visits + 1
            return VisitOutcome(d.location_id, VisitStatus.COMPLETED, observations=observed())

    rooms = []

    class Spy(MinCostPolicy):
        def select(self, stage, target, candidates, context):
            if stage == Stage.ROOM:
                rooms.append([c.info.get("observed_fraction") for c in candidates])
            return super().select(stage, target, candidates, context)

    graph = SceneGraph(toy_env.toy_graph(["remote"]))
    coverage = RoomCoverage({"R_A": [(2.5, 3.0)], "R_B": [(9.0, 9.0)]}, lambda x, y, px, py: True)
    policies = {s: Spy() for s in Stage}
    rollout(graph, NavigationPlanner(GRID), Env(), "remote", policies, 1.0, coverage=coverage)
    assert rooms and all(f is not None for r in rooms for f in r)


def test_shuffled_policy_presents_a_seeded_order_and_keeps_the_choice():
    from athome.search.policy import Candidate, MinCostPolicy, PlanningContext, ShuffledPolicy

    class First:
        def select(self, stage, target, candidates, context):
            return candidates[0].candidate_id

    cands = [Candidate(f"_{i}", float(i)) for i in range(6)]
    orders = set()
    for seed in range(5):
        p = ShuffledPolicy(First(), seed)
        chosen = p.select(Stage.ROOM, "cup", cands, PlanningContext())
        assert chosen == p.last_order[0]
        orders.add(tuple(p.last_order))
    assert len(orders) > 1
    again = ShuffledPolicy(First(), 3)
    again.select(Stage.ROOM, "cup", cands, PlanningContext())
    assert tuple(again.last_order) in orders
    # Order-invariant policies choose the same candidate.
    assert ShuffledPolicy(MinCostPolicy(), 1).select(Stage.ROOM, "cup", cands, PlanningContext()) == "_0"


def test_teacher_record_keeps_current_room_and_prompt_versions():
    from athome.inference.prompts import PROMPT_VERSION, TEACHER_PROMPT_VERSION
    from athome.search.policy import Candidate, PlanningContext
    from athome.training.teacher import MinCostTeacher

    teacher = MinCostTeacher()
    cands = [Candidate("_1", 1.0, {"room_label": "kitchen", "workspaces": [], "standalone_categories": []}),
             Candidate("_2", 2.0, {"room_label": "bedroom", "workspaces": [], "standalone_categories": []})]
    teacher.select(Stage.ROOM, "cup", cands, PlanningContext(current_room="_2"))
    record = teacher.records[-1]
    assert record.context["current_room"] == "_2"
    assert (record.prompt_version, record.teacher_prompt_version) == (PROMPT_VERSION, TEACHER_PROMPT_VERSION)


def test_standalone_candidates_are_grouped_by_category():
    from athome.inference.prompts import build_messages
    from athome.navigation import LocationCost
    from athome.search.policy import PlanningContext
    from athome.search.session import group_candidate, standalone_groups

    graph = SceneGraph(toy_env.toy_graph())
    standalone = [l for l in graph.locations.values() if l.kind == "standalone"]
    loc = standalone[0]
    # Two instances of one category (same location object reused under two IDs).
    graph.locations["standalone:copy"] = loc.__class__(**{**loc.__dict__, "location_id": "standalone:copy"})
    costs = [LocationCost(loc.location_id, 3.0, ()), LocationCost("standalone:copy", 1.0, ())]
    groups = standalone_groups(graph, loc.room_id, costs)
    (gid, members), = groups.items()
    assert [m.location_id for m in members] == ["standalone:copy", loc.location_id]
    cand = group_candidate(gid, members)
    assert cand.cost == 1.0 and cand.info["count"] == 2 and cand.info["category"] == loc.category
    text = build_messages(Stage.STANDALONE, "cup", [cand], PlanningContext())[0][1]["content"]
    assert f"Object: {loc.category} x2" in text and "A* path cost: 1.0 m (nearest)" in text


def test_standalone_stage_offers_groups_and_visits_the_nearest_instance():
    seen = []

    class Spy:
        def select(self, stage, target, candidates, context):
            if stage == Stage.STANDALONE:
                seen.append([c.candidate_id for c in candidates])
            return min(candidates, key=lambda c: c.cost).candidate_id

    s = session(["remote"], hide=["remote"], policy=Spy())
    order = run(s, {})
    assert seen and all(cid.startswith("standalone_group:") for ids in seen for cid in ids)
    assert any(stage == Stage.STANDALONE for _, stage, _ in order)
    assert all(lid.startswith("standalone:") for _, stage, lid in order if stage == Stage.STANDALONE)
