from athome.inference.prompts import PROMPT_VERSION, build_messages
from athome.search.policy import Candidate, PlanningContext, Stage


def room(cid, cost, label):
    return Candidate(cid, cost, {"room_label": label, "workspaces": [], "standalone_categories": []})


def test_room_stage_marks_the_current_room_only():
    candidates = [room("_1", 2.0, "bedroom"), room("_2", 0.5, "bathroom")]
    text = build_messages(Stage.ROOM, "towel", candidates, PlanningContext(current_room="_2"))[0][1]["content"]
    assert "- C2 (current room)" in text and "- C1 (current room)" not in text
    first = build_messages(Stage.ROOM, "towel", candidates, PlanningContext())[0][1]["content"]
    assert "(current room)" not in first
    assert PROMPT_VERSION == "0.6"


def test_room_stage_shows_search_coverage():
    c = Candidate("_1", 1.0, {"room_label": "bedroom", "workspaces": [], "standalone_categories": [],
                              "searched_locations": 3, "total_locations": 10})
    text = build_messages(Stage.ROOM, "bag", [c, room("_2", 2.0, "kitchen")], PlanningContext())[0][1]["content"]
    assert "Searched here: 3 of 10 locations, target not seen" in text


def test_observed_fraction_replaces_visit_count():
    c = Candidate("_1", 1.0, {"room_label": "bedroom", "workspaces": [], "standalone_categories": [],
                              "searched_locations": 1, "total_locations": 13, "observed_fraction": 1.0})
    text = build_messages(Stage.ROOM, "bag", [c, room("_2", 2.0, "kitchen")], PlanningContext())[0][1]["content"]
    first = text.split("- C2")[0]
    assert "Observed: 100% of this room, target not seen" in first and "Searched here" not in first


def test_repeated_objects_are_counted():
    c = Candidate("_1", 1.0, {"room_label": "storage",
                              "workspaces": [{"category": "shelf", "function_label": "general_storage"}] * 3
                              + [{"category": "table", "function_label": None}],
                              "standalone_categories": ["box", "box", "lamp"]})
    text = build_messages(Stage.ROOM, "bag", [c], PlanningContext())[0][1]["content"]
    assert "Workspaces: shelf (general_storage) x3, table" in text
    assert "Standalone objects: box x2, lamp" in text
