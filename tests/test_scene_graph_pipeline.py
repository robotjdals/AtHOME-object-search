import math

import numpy as np
import pytest

from athome.execution.visit import VisitOutcome, VisitStatus
from athome.navigation import NavigationPlanner
from athome.scene_graph.object_rooms import assign_rooms
from athome.scene_graph.pipeline import build_scene_graph
from athome.scene_graph.query import SceneGraph
from athome.scene_graph.room_segmentation import segment_rooms
from athome.scene_graph.semantic_labeling import label_rooms, validate_label
from athome.scene_graph.static_features import (
    load_static_features,
    oriented_to_aabb,
)
from athome.schemas import ObservationFrame, ObservedObject, Pose2D
from athome.search import SearchSession, TargetStatus
from athome.testing import toy_env

RES = toy_env.RESOLUTION


def raw_free():
    return toy_env.toy_occupancy() < 50


@pytest.fixture
def features(tmp_path):
    toy_env.write_toy_static_features(tmp_path / "static.json", hidden_categories=["remote"])
    return load_static_features(tmp_path / "static.json")


def room_of(seg, x, y):
    return int(seg.labels[int(y / RES), int(x / RES)])


def test_two_rooms_split_at_the_door():
    seg = segment_rooms(raw_free(), (0, 0), RES)
    assert len(seg.room_ids) == 2
    kitchen, living = room_of(seg, 2.5, 3.0), room_of(seg, 7.5, 3.5)
    assert kitchen and living and kitchen != living
    assert seg.area(kitchen) == pytest.approx(seg.area(living), rel=0.2)


def test_single_room_with_furniture_is_not_split():
    occ = toy_env.toy_occupancy()
    occ[:, 100:102] = 0            # remove the dividing wall
    seg = segment_rooms(occ < 50, (0, 0), RES)
    assert len(seg.room_ids) == 1


def test_narrow_closet_without_marker_is_still_a_room():
    occ = np.full((60, 120), 100, np.int16)
    occ[5:55, 5:75] = 0            # 3.5 x 2.5 m room
    occ[20:40, 80:115] = 0         # 1.75 x 1.0 m closet, clearance < 0.6 m
    occ[28:33, 75:80] = 0          # door
    seg = segment_rooms(occ < 50, (0, 0), RES)
    assert len(seg.room_ids) == 2


def test_oriented_box_to_aabb():
    lo, hi = oriented_to_aabb((0, 0, 0.5), (2.0, 1.0, 1.0), math.pi / 2)
    assert lo == pytest.approx((-0.5, -1.0, 0.0))
    assert hi == pytest.approx((0.5, 1.0, 1.0))


def test_furniture_centroid_on_obstacle_uses_nearest_room(features):
    seg = segment_rooms(raw_free(), (0, 0), RES)
    result = assign_rooms(features.objects, seg)
    by_label = {o.label: o.object_id for o in features.objects}
    assert result.methods[by_label["sofa"]] == "nearest"
    assert result.rooms[by_label["sofa"]] == f"room_{room_of(seg, 7.5, 3.5)}"
    assert result.rooms[by_label["cup"]] == result.rooms[by_label["table"]]
    assert not result.unassigned


def test_pipeline_builds_searchable_graph(features):
    result = build_scene_graph(features, raw_free(), (0, 0), RES, toy_env.toy_labeler)
    graph = SceneGraph(result.graph)
    labels = sorted(r.label for r in graph.rooms.values())
    assert labels == ["kitchen", "living_room"]
    # Same labeling protocol and source policy as the HM3D training graphs.
    assert result.graph["provenance"]["labeling_protocol"]["name"] == "v2"
    assert result.graph["provenance"]["labeling_protocol"]["n"] == 9
    assert result.graph["association_policy"]["child_excluded_categories"]
    assert result.labels["review"]["method"] == "real_environment_source_policy"
    ws = {l.category: l for l in graph.locations.values() if l.kind == "workspace"}
    assert set(ws) == {"table", "counter", "coffee table", "shelf"}
    assert "cup" in ws["table"].child_categories
    # Inner shelf tiers are not workspace surfaces: the book stays standalone.
    assert "book" not in ws["shelf"].child_categories
    assert any(l.category == "book" for l in graph.locations.values())

    # The generated graph drives a search: remote was not in the features.
    s = SearchSession(graph, NavigationPlanner(toy_env.toy_grid()), ["remote"])
    assert not s.targets[0].known
    pose = Pose2D(2.5, 3.0, 0.0)
    while (d := s.next_decision(pose)) is not None:
        seen = ("remote",) if ws["coffee table"].location_id == d.location_id else ()
        frame = ObservationFrame(1.0, tuple(ObservedObject(0, l, 0.9, (0, 0, 0)) for l in seen))
        s.report(d, VisitOutcome(d.location_id, VisitStatus.COMPLETED, observations=[frame]))
        pose = d.goals[0]
    assert s.targets[0].status == TargetStatus.FOUND


def test_labeler_output_is_validated(features):
    room = {"room_id": "room_1", "objects": [
        {"object_id": "table_0", "semantic_tag": "table",
         "bbox": {"size": [1, 1, 1]}, "child_candidate_ids": []}]}
    ok = {"room_id": "room_1", "room_label": "kitchen",
          "workspace_sources": [{"source_object_id": "table_0", "function_label": "dining"}]}
    assert validate_label(room, ok)["room_label"] == "kitchen"
    for bad in (
        {**ok, "room_id": "room_2"},
        {**ok, "room_label": "spaceship"},
        {**ok, "workspace_sources": [{"source_object_id": "ghost_9", "function_label": "x"}]},
    ):
        with pytest.raises(ValueError):
            validate_label(room, bad)
    with pytest.raises(ValueError):
        label_rooms({"rooms": [room]}, lambda m, s, n, t: [{**ok, "room_label": "spaceship"}] * n)


SAMPLE = {
    "object_id": 0, "semantic_label": "lamp",
    "clip_feature": [0.012, -0.084, 0.031],
    "point_cloud": [[2.91, 0.21, -0.31], [2.95, 0.24, -0.27], [3.01, 0.30, -0.23]],
    "bbox_3d": {"center": [2.969, 0.262, -0.264], "extent": [0.156, 0.131, 0.164], "yaw": 1.524},
    "centroid": [2.969, 0.262, -0.269],
}


def test_perception_format_file_and_directory(tmp_path):
    import json

    second = {**SAMPLE, "object_id": 1, "semantic_label": "cup"}
    (tmp_path / "list.json").write_text(json.dumps([SAMPLE, second]))
    per_object = tmp_path / "objects"
    per_object.mkdir()
    for o in (SAMPLE, second):
        (per_object / f"{o['object_id']}.json").write_text(json.dumps(o))

    for source in (tmp_path / "list.json", per_object):
        f = load_static_features(source)
        assert [o.object_id for o in f.objects] == ["lamp_0", "cup_1"]
        lamp = f.objects[0]
        # yaw ~ 90 deg: the box footprint swaps its x/y extents.
        assert lamp.bbox_max[0] - lamp.bbox_min[0] == pytest.approx(0.131, abs=0.01)
        assert lamp.bbox_max[1] - lamp.bbox_min[1] == pytest.approx(0.156, abs=0.01)
        assert lamp.bbox_max[2] - lamp.bbox_min[2] == pytest.approx(0.164)
        assert f.features.shape == (2, 3)
        assert f.points(lamp).shape == (3, 3)
        assert f.map_version is None


def test_points_used_when_centroid_is_outside_rooms(tmp_path):
    import json

    seg = segment_rooms(raw_free(), (0, 0), RES)
    # Centroid far outside the map, points inside the living room.
    obj = {**SAMPLE, "centroid": [50.0, 50.0, 0.3],
           "bbox_3d": {"center": [50.0, 50.0, 0.3], "extent": [0.1, 0.1, 0.1], "yaw": 0.0},
           "point_cloud": [[7.5, 3.5, 0.3], [7.6, 3.4, 0.3], [7.4, 3.6, 0.3]]}
    (tmp_path / "o.json").write_text(json.dumps([obj]))
    f = load_static_features(tmp_path / "o.json")
    result = assign_rooms(f.objects, seg, f.points)
    assert result.methods["lamp_0"] == "points"
    assert result.rooms["lamp_0"] == f"room_{room_of(seg, 7.5, 3.5)}"


def test_training_scope_rules_apply_to_real_graph(features):
    import json
    from pathlib import Path

    from athome.scene_graph.pipeline import BuildConfig

    scope = json.loads((Path(__file__).resolve().parents[1]
                        / "configs/data/scene_scope.json").read_text(encoding="utf-8"))
    result = build_scene_graph(features, raw_free(), (0, 0), RES, toy_env.toy_labeler,
                               BuildConfig(scene_scope=scope, floor_z_m=0.0))
    # The book on an inner shelf tier is outside the scope, as in HM3D graphs.
    tag = {o.object_id: o.label for o in features.objects}
    excluded = result.graph["provenance"]["scope_excluded_objects"]
    assert [(tag[o], tag[c]) for o, c in excluded.items()] == [("book", "shelf")]
    assert all(o["object_id"] not in excluded for o in result.graph["objects"])
    graph = SceneGraph(result.graph)
    assert not any(l.category == "book" for l in graph.locations.values())
    assert not graph.known_locations("book")
    # Floor height reaches every room, so the 1.5 m search-height scope applies.
    assert graph.room_floor_z == {rid: 0.0 for rid in graph.rooms}
