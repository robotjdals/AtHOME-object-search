from athome.navigation import NavigationPlanner
from athome.scene_graph.query import STANDALONE, SceneGraph
from athome.schemas import Pose2D
from athome.search.baselines import (CooccurrencePolicy, FlatSearchSession, PlacementStats,
                                     WithoutObservation, flat_graph)
from athome.search.policy import Candidate, MinCostPolicy, Stage
from athome.testing import toy_env

GRID = toy_env.toy_grid()


class Recorder:
    def __init__(self):
        self.calls = []

    def select(self, stage, target, candidates, context):
        self.calls.append((stage, candidates))
        return MinCostPolicy().select(stage, target, candidates, context)


def test_flat_graph_turns_every_object_into_its_own_location():
    flat = SceneGraph(flat_graph(toy_env.toy_graph(["remote"])))
    assert not any(l.kind != STANDALONE for l in flat.locations.values())
    # Children of furniture (cup, plate, kettle, book) are locations themselves; the wall is not.
    categories = sorted(l.category for l in flat.locations.values())
    assert categories == sorted(["table", "cup", "plate", "counter", "kettle", "fridge", "coffee table",
                                 "shelf", "book", "lamp", "sofa"])
    assert all(not l.child_categories for l in flat.locations.values())


def test_flat_session_decides_once_over_the_whole_scene():
    graph = SceneGraph(flat_graph(toy_env.toy_graph(["remote"])))
    policy = Recorder()
    session = FlatSearchSession(graph, NavigationPlanner(GRID), ["remote"], policy=policy)
    decision = session.next_decision(Pose2D(2.5, 3.0, 0.0))
    stage, candidates = policy.calls[0]
    assert stage == Stage.STANDALONE and decision.stage == Stage.STANDALONE
    assert {c.info["category"] for c in candidates} >= {"cup", "sofa", "book"}     # both rooms at once
    reachable = session.navigation.evaluate(Pose2D(2.5, 3.0, 0.0), list(graph.locations))
    assert {c.candidate_id for c in candidates} == set(reachable)


def test_without_observation_strips_the_room_state():
    policy = Recorder()
    room = Candidate("R_A", 1.0, {"room_label": "kitchen", "observed_fraction": 0.4,
                                  "searched_locations": 1, "total_locations": 3})
    WithoutObservation(policy).select(Stage.ROOM, "cup", [room], None)
    assert policy.calls[0][1][0].info == {"room_label": "kitchen"}


def test_cooccurrence_prefers_where_the_target_was_placed():
    stats = PlacementStats()
    graph = toy_env.toy_graph()
    for _ in range(3):
        stats.add_scene(graph, {"kettle": [{"room_id": "R_A", "unmasked_graph_anchor":
                                            {"type": "workspace", "id": "workspace:R_A:counter_4"}}]})
    policy = CooccurrencePolicy(stats)
    rooms = [Candidate("R_A", 5.0, {"room_label": "kitchen"}), Candidate("R_B", 1.0, {"room_label": "living room"})]
    assert policy.select(Stage.ROOM, "kettle", rooms) == "R_A"
    spots = [Candidate("w:table", 1.0, {"category": "table"}), Candidate("w:counter", 2.0, {"category": "counter"})]
    assert policy.select(Stage.WORKSPACE, "kettle", spots) == "w:counter"
    # A target the statistics never saw: nearest first.
    assert policy.select(Stage.ROOM, "drone", rooms) == "R_B"
    # The searched share of a room lowers it (Teacher's after-search rule).
    searched = [Candidate("R_A", 5.0, {"room_label": "kitchen", "observed_fraction": 0.99}), rooms[1]]
    assert policy.select(Stage.ROOM, "kettle", searched) == "R_B"


def test_prompt_without_observation_has_no_search_state_line():
    from athome.inference.prompts import build_messages
    from athome.search.policy import PlanningContext

    room = Candidate("R_A", 1.0, {"room_label": "kitchen", "searched_locations": 1, "total_locations": 3})
    policy = Recorder()
    WithoutObservation(policy).select(Stage.ROOM, "cup", [room, Candidate("R_B", 2.0, {"room_label": "bath"})], None)
    stripped, _ = build_messages(Stage.ROOM, "cup", policy.calls[0][1], PlanningContext())
    full, _ = build_messages(Stage.ROOM, "cup", [room], PlanningContext())
    assert "Searched here" not in stripped[-1]["content"] and "Observed" not in stripped[-1]["content"]
    assert "Searched here: 1 of 3 locations" in full[-1]["content"]


def test_given_room_type_fixes_the_label_and_is_shown_to_the_labeler():
    from athome.scene_graph.semantic_labeling import PROMPTS, label_rooms

    room = {"room_id": "room_1", "objects": [
        {"object_id": "counter top_51", "semantic_tag": "counter top", "child_candidate_ids": [],
         "bbox": {"size": [1.0, 0.6, 0.9]}}]}
    seen = []

    def sample(messages, schema, n, temperature):
        seen.append((messages, schema))
        return [{"room_id": "room_1", "room_label": "bathroom", "workspace_sources": [
            {"source_object_id": "counter top_51", "function_label": "personal_care"}]}] * n

    out = label_rooms({"rooms": [room]}, sample, given_labels={"room_1": "bathroom"})
    messages, schema = seen[0]
    assert messages[0]["content"] == PROMPTS["v2_given"]
    assert '"room_label_given":"bathroom"' in messages[1]["content"]
    assert schema["properties"]["room_label"]["enum"] == ["bathroom"]
    assert out["protocol"] == "v2_given" and out["rooms"][0]["room_label"] == "bathroom"
