import pytest

from athome.data.hm3d.scope import apply_storage_scope, find_contained

SCOPE = {"storage_container_categories": ["shelf"],
         "containment": {"min_footprint_inside": 0.9, "vertical_tolerance_m": 0.02,
                         "min_below_container_top_m": 0.05}}


def obj(oid, tag, lo, hi, room="R"):
    return {"object_id": oid, "semantic_tag": tag, "room_id": room,
            "bbox": {"min": list(lo), "max": list(hi)}}


def test_inside_shelf_removed_on_top_kept_non_container_ignored():
    shelf = obj("shelf_1", "shelf", (0, 0, 0), (1, 0.4, 2))
    inside = obj("book_2", "book", (0.1, 0.1, 0.5), (0.3, 0.3, 0.8))
    on_top = obj("box_3", "box", (0.1, 0.1, 2.0), (0.4, 0.3, 2.3))
    sofa = obj("sofa_4", "sofa", (3, 0, 0), (5, 1, 1))
    cushion = obj("pillow_5", "pillow", (3.2, 0.2, 0.4), (3.6, 0.6, 0.7))
    other_room = obj("book_6", "book", (0.1, 0.1, 0.5), (0.3, 0.3, 0.8), room="Q")
    objects = [shelf, inside, on_top, sofa, cushion, other_room]
    assert find_contained(objects, SCOPE) == {"book_2": "shelf_1"}

    room_map = {"coordinate_frame": "athome_z_up", "objects": objects,
                "rooms": [{"room_id": "R", "object_ids": [o["object_id"] for o in objects[:5]]}]}
    scoped, _ = apply_storage_scope(room_map, SCOPE)
    assert "book_2" not in {o["object_id"] for o in scoped["objects"]}
    assert "book_2" not in scoped["rooms"][0]["object_ids"]
    assert scoped["excluded_objects"][-1]["reasons"] == ["inside_storage_container"]
    assert len(room_map["objects"]) == 6  # input untouched


def test_partially_protruding_object_kept():
    shelf = obj("shelf_1", "shelf", (0, 0, 0), (1, 0.4, 2))
    sticking_out = obj("bag_2", "bag", (0.8, 0.1, 0.5), (1.4, 0.3, 0.8))
    assert find_contained([shelf, sticking_out], SCOPE) == {}


def test_requires_z_up():
    with pytest.raises(ValueError):
        apply_storage_scope({"coordinate_frame": "habitat_native", "objects": [], "rooms": []}, SCOPE)
