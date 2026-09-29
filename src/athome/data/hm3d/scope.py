"""Project scope for HM3D room-object maps (athome_z_up).

Objects stored inside storage furniture (inner shelf tiers, cabinet
compartments) are outside the project scope: the real robot does not treat
them as separate objects, so the training scene graph must not contain them.

An object is inside a container when, in the same room,
- at least ``min_footprint_inside`` of its XY footprint lies in the container's,
- its bottom is not below the container's bottom (``vertical_tolerance_m``), and
- its top is at least ``min_below_container_top_m`` below the container's top
  (objects resting on the top surface are kept).
Only listed open-storage categories are containers; sofas, beds or stairs
would otherwise "contain" cushions and items lying on them.
"""
from __future__ import annotations

from copy import deepcopy

from athome.data.hm3d.target_masking import normalize_tag


def footprint_inside(a, b):
    """Fraction of bbox ``a``'s XY footprint inside bbox ``b``."""
    (ax0, ay0), (ax1, ay1) = a["min"][:2], a["max"][:2]
    (bx0, by0), (bx1, by1) = b["min"][:2], b["max"][:2]
    area = (ax1 - ax0) * (ay1 - ay0)
    if area <= 0:
        return 0.0
    w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    h = max(0.0, min(ay1, by1) - max(ay0, by0))
    return w * h / area


def _volume(box):
    return ((box["max"][0] - box["min"][0]) * (box["max"][1] - box["min"][1])
            * (box["max"][2] - box["min"][2]))


def find_contained(objects, scope):
    """object_id -> container object_id (smallest enclosing container)."""
    rule = scope["containment"]
    categories = {normalize_tag(c) for c in scope["storage_container_categories"]}
    containers = [o for o in objects if normalize_tag(o["semantic_tag"]) in categories]
    result = {}
    for obj in objects:
        box = obj["bbox"]
        enclosing = [
            c for c in containers
            if c is not obj and c["room_id"] == obj["room_id"]
            and footprint_inside(box, c["bbox"]) >= rule["min_footprint_inside"]
            and box["min"][2] >= c["bbox"]["min"][2] - rule["vertical_tolerance_m"]
            and box["max"][2] <= c["bbox"]["max"][2] - rule["min_below_container_top_m"]
            and _volume(c["bbox"]) > _volume(box)
        ]
        if enclosing:
            result[obj["object_id"]] = min(enclosing, key=lambda c: _volume(c["bbox"]))["object_id"]
    return result


def apply_storage_scope(room_object_map, scope):
    """Return (scoped map, contained dict). Contained objects move to
    ``excluded_objects`` with reason ``inside_storage_container``."""
    if room_object_map.get("coordinate_frame") != "athome_z_up":
        raise ValueError("athome_z_up room-object map만 입력할 수 있습니다.")
    contained = find_contained(room_object_map["objects"], scope)
    result = deepcopy(room_object_map)
    result["objects"] = [o for o in result["objects"] if o["object_id"] not in contained]
    for room in result["rooms"]:
        room["object_ids"] = [i for i in room["object_ids"] if i not in contained]
    by_id = {o["object_id"]: o for o in room_object_map["objects"]}
    result["excluded_objects"] = list(result.get("excluded_objects", [])) + [
        {"object_id": oid, "category": by_id[oid]["semantic_tag"],
         "source_region_id": by_id[oid]["room_id"],
         "reasons": ["inside_storage_container"], "container_id": cid}
        for oid, cid in sorted(contained.items())
    ]
    result["scene_scope"] = deepcopy(scope)
    return result, contained
