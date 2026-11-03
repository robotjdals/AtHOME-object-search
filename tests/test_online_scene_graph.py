import numpy as np
import pytest

from athome.execution.visit import VisitOutcome, VisitStatus
from athome.navigation import NavigationPlanner
from athome.scene_graph.online import OnlineConfig, OnlineSceneGraph
from athome.scene_graph.query import SceneGraph
from athome.scene_graph.room_segmentation import RoomSegmentation
from athome.schemas import ObservationFrame, ObservedObject, Pose2D
from athome.search import SearchSession, Stage
from athome.testing import toy_env

RES = toy_env.RESOLUTION
COFFEE = "workspace:R_B:coffee table_7"
# Training label rates (workspace_source_priors.json format).
PRIORS = {"categories": {
    "desk": {"present": 258, "source": 258, "rate": 1.0, "function_label": "desk_work"},
    "cabinet": {"present": 1773, "source": 674, "rate": 0.38, "function_label": "general_storage"},
    "bag": {"present": 300, "source": 0, "rate": 0.0}}}


def toy_rooms(extra_room=False):
    """R_A left of the dividing wall (x < 5), R_B right of it; optionally a
    third room (x > 9.5) that the graph does not have."""
    labels = np.zeros(toy_env.toy_occupancy().shape, dtype=np.int32)
    labels[:, :100] = 1
    labels[:, 102:] = 2
    if extra_room:
        labels[:, 190:] = 3
    return RoomSegmentation(labels, (0.0, 0.0), RES)


def online(extra_room=False, priors=PRIORS, **config):
    names = ["R_A", "R_B", "R_C"] if extra_room else ["R_A", "R_B"]
    return OnlineSceneGraph(toy_env.toy_graph(), toy_rooms(extra_room), OnlineConfig(**config),
                            room_names=names, source_priors=priors)


def seen(label, lo, hi, object_id=-1):
    center = tuple((a + b) / 2 for a, b in zip(lo, hi))
    size = tuple(b - a for a, b in zip(lo, hi))
    return ObservedObject(object_id, label, 0.9, center, bbox_center=center, bbox_size=size)


def frame(stamp, *objects):
    return ObservationFrame(stamp, tuple(objects))


def twice(g, *objects):
    g.observe(frame(1.0, *objects))
    return g.observe(frame(2.0, *objects))


MUG_ON_COFFEE_TABLE = ((7.0, 2.2, 0.45), (7.1, 2.3, 0.55))
BAG_ON_FLOOR = ((8.0, 3.0, 0.0), (8.4, 3.3, 0.4))
NEW_DESK = ((8.0, 0.5, 0.0), (9.0, 1.1, 0.75))
LAPTOP_ON_NEW_DESK = ((8.3, 0.7, 0.75), (8.6, 0.9, 0.78))


def test_new_object_on_a_workspace_becomes_its_child_after_two_frames():
    g = online()
    assert g.observe(frame(1.0, seen("mug", *MUG_ON_COFFEE_TABLE))) == []
    changes = g.observe(frame(2.0, seen("mug", *MUG_ON_COFFEE_TABLE)))
    assert [c["event"] for c in changes] == ["object_added"]
    assert changes[0]["role"] == "child" and changes[0]["parent_id"] == COFFEE
    graph = SceneGraph(g.graph())
    assert "mug" in graph.locations[COFFEE].child_categories
    assert graph.known_locations("mug") == [COFFEE]


def test_one_frame_is_not_enough():
    g = online()
    g.observe(frame(1.0, seen("mug", *MUG_ON_COFFEE_TABLE)))
    assert SceneGraph(g.graph()).known_locations("mug") == []


def test_new_object_on_the_floor_is_a_standalone_location():
    g = online()
    changes = twice(g, seen("bag", *BAG_ON_FLOOR))
    assert changes[0]["role"] == "standalone" and changes[0]["room_id"] == "R_B"
    assert f"standalone:{changes[0]['object_id']}" in SceneGraph(g.graph()).locations


def test_new_furniture_of_a_workspace_category_becomes_a_workspace():
    g = online()
    changes = twice(g, seen("desk", *NEW_DESK))
    assert changes[0]["event"] == "workspace_added"
    assert changes[0]["function_label"] == "desk_work"
    loc = SceneGraph(g.graph()).locations[changes[0]["workspace_id"]]
    assert loc.kind == "workspace" and loc.function_label == "desk_work" and loc.room_id == "R_B"


def test_objects_already_on_new_furniture_move_under_it():
    g = online()
    laptop = twice(g, seen("laptop", *LAPTOP_ON_NEW_DESK))[0]
    assert laptop["role"] == "standalone"
    changes = twice(g, seen("desk", *NEW_DESK))
    moved = [c for c in changes if c["event"] == "object_moved"]
    assert moved[0]["object_id"] == laptop["object_id"] and moved[0]["role"] == "child"
    assert SceneGraph(g.graph()).locations[changes[0]["workspace_id"]].child_categories == ("laptop",)


def test_furniture_the_labels_rarely_chose_stays_standalone():
    changes = twice(online(), seen("cabinet", *NEW_DESK))
    assert changes[0]["event"] == "object_added" and changes[0]["role"] == "standalone"


def test_without_priors_new_furniture_stays_standalone():
    changes = twice(online(priors=None), seen("desk", *NEW_DESK))
    assert changes[0]["role"] == "standalone"


def test_premap_link_and_nearby_same_category_are_the_existing_object():
    g = online()
    linked = seen("cup", (1.6, 1.6, 0.75), (1.7, 1.7, 0.85), object_id=2)    # perception: cup_2
    nearby = seen("cup", (1.25, 1.2, 0.75), (1.35, 1.3, 0.85))                # no link, 5 cm away
    for t in (1.0, 2.0, 3.0):
        assert g.observe(frame(t, linked, nearby)) == []
    assert g.added == []


def test_part_of_a_large_object_is_the_same_object():
    """A partly seen table (box center 0.57 m from the mapped center, inside
    the mapped box) is the table, not a second one (Hydra's containment test)."""
    g = online()
    corner = seen("table", (1.85, 1.65, 0.6), (2.05, 1.85, 0.75))
    for t in (1.0, 2.0, 3.0):
        assert g.observe(frame(t, corner)) == []


def test_premap_link_far_from_the_object_is_not_that_object():
    changes = twice(online(), seen("cup", *MUG_ON_COFFEE_TABLE, object_id=2))
    assert changes[0]["event"] == "object_added" and changes[0]["parent_id"] == COFFEE


def test_objects_in_rooms_the_graph_does_not_have_are_rejected_once():
    g = online(extra_room=True)
    box = ((9.6, 3.0, 0.0), (9.7, 3.1, 0.2))
    changes = twice(g, seen("bag", *box))
    assert changes[0]["event"] == "object_rejected" and changes[0]["reason"] == "room_not_in_graph"
    assert g.observe(frame(3.0, seen("bag", *box))) == []
    assert g.observe(frame(4.0, seen("bag", *box))) == []


def test_dynamic_observations_are_ignored():
    g = online()
    person = ObservedObject(-1, "person", 0.9, (8.0, 3.0, 0.8), is_static=False)
    twice(g, person)
    assert g.added == []


def test_detections_below_the_found_check_confidence_are_ignored():
    g = online(min_confidence=0.15)
    center = tuple((a + b) / 2 for a, b in zip(*BAG_ON_FLOOR))
    size = tuple(b - a for a, b in zip(*BAG_ON_FLOOR))
    weak = ObservedObject(-1, "bag", 0.1, center, bbox_center=center, bbox_size=size)
    twice(g, weak)
    assert g.added == []
    twice(g, seen("bag", *BAG_ON_FLOOR))
    assert len(g.added) == 1


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        OnlineConfig.from_dict({"enabled": True, "min_frame": 2})


# --- search session on an updated graph ---------------------------------------

def make_session(targets):
    graph = SceneGraph(toy_env.toy_graph())
    return SearchSession(graph, NavigationPlanner(toy_env.toy_grid()), targets)


def test_target_seen_on_the_way_becomes_known():
    s = make_session(["bag"])
    assert not s.targets[0].known
    g = online()
    oid = twice(g, seen("bag", *BAG_ON_FLOOR))[0]["object_id"]
    s.update_graph(SceneGraph(g.graph()))
    d = s.next_decision(Pose2D(2.5, 3.0, 0.0))
    assert d.stage == Stage.KNOWN and d.location_id == f"standalone:{oid}"
    assert s.targets[0].became_known


def test_new_workspace_is_a_candidate_of_the_unknown_search():
    s = make_session(["laptop"])
    g = online()
    wid = twice(g, seen("desk", *NEW_DESK))[0]["workspace_id"]
    s.update_graph(SceneGraph(g.graph()))
    assert wid in s.graph.locations
    assert s.navigation.has_candidates(wid)


def test_visited_location_that_left_the_graph_keeps_the_context():
    s = make_session(["remote"])
    d = s.next_decision(Pose2D(2.5, 3.0, 0.0))
    s.report(d, VisitOutcome(d.location_id, VisitStatus.COMPLETED))
    graph = online().graph()
    visited = s.graph.goal(d.location_id)
    if visited.kind == "standalone":
        graph["objects"] = [o for o in graph["objects"] if o["object_id"] != visited.object_id]
    else:
        graph["workspaces"] = [w for w in graph["workspaces"] if w["workspace_id"] != d.location_id]
    s.update_graph(SceneGraph(graph))
    assert s.next_decision(d.goals[0]) is not None
