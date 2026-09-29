"""Rule-based review of LLM workspace-source labels, and graph validation.

The same policy applies to unmasked and target-masked scenes:
- structural surfaces must never be selected (error: the LLM broke the prompt);
- trays are portable containers, not workspace surfaces (excluded);
- bench/piano/toilet surfaces are kept but marked unverified.
"""
from __future__ import annotations

from copy import deepcopy

STRUCTURAL = {"wall", "floor", "ceiling", "staircase wall"}
EXCLUDED_SOURCES = {"tray"}
UNVERIFIED_SURFACES = {"bench", "piano", "toilet"}
ASSOCIATION_POLICY = {"version": "0.2", "child_excluded_categories": sorted(STRUCTURAL)}


def category(obj):
    return " ".join(obj["semantic_tag"].casefold().split())


def apply_source_policy(labels, objects, method):
    """Return reviewed copy of ``labels``; ``objects``: object_id -> input object."""
    labels = deepcopy(labels)
    edits, unverified = [], []
    for room in labels["rooms"]:
        retained = []
        for source in room["workspace_sources"]:
            oid = source["source_object_id"]
            obj = objects[oid]
            if "room_id" in obj and obj["room_id"] != room["room_id"]:
                raise ValueError(f"{oid}: Source Room 불일치")
            tag = category(obj)
            if tag in STRUCTURAL:
                raise ValueError(f"{oid}: 구조물이 Source로 선정됨")
            if tag in EXCLUDED_SOURCES:
                edits.append({
                    "room_id": room["room_id"], "source_object_id": oid,
                    "original_label": deepcopy(source),
                    "decision": "exclude_as_workspace_source",
                    "reason": "portable_tray_excluded_by_source_policy",
                })
                continue
            retained.append(source)
            if tag in UNVERIFIED_SURFACES:
                unverified.append({
                    "room_id": room["room_id"], "source_object_id": oid, "category": tag,
                    "decision": "retain_provisionally", "surface_geometry_verified": False,
                })
        room["workspace_sources"] = retained
    labels["status"] = "reviewed_for_graph_assembly"
    labels["review"] = {
        "method": method, "is_ground_truth": False, "surface_geometry_verified": False,
        "changes": edits, "provisional_sources": unverified,
    }
    return labels


def validate_workspace_graph(graph, input_object_ids, name=""):
    """Object preservation and Room -> Workspace -> Child reference checks."""
    graph_objects = {o["object_id"]: o for o in graph["objects"]}
    workspaces = {w["workspace_id"]: w for w in graph["workspaces"]}
    if len(graph_objects) != len(graph["objects"]):
        raise ValueError(f"{name}: 객체 중복")
    if set(graph_objects) != set(input_object_ids):
        raise ValueError(f"{name}: 객체 누락 또는 추가")
    linked_children, source_ids = [], set()
    for wid, workspace in workspaces.items():
        sid = workspace["source_object_id"]
        source_ids.add(sid)
        source = graph_objects[sid]
        if source["role"] != "source" or source["parent_type"] != "room":
            raise ValueError(f"{name}/{sid}: Source 연결 오류")
        if source["workspace_id"] != wid:
            raise ValueError(f"{name}/{sid}: Workspace 참조 오류")
        for cid in workspace["child_object_ids"]:
            child = graph_objects[cid]
            if (child["role"] != "child" or child["parent_id"] != wid
                    or child["room_id"] != workspace["room_id"]
                    or category(child) in STRUCTURAL):
                raise ValueError(f"{name}/{cid}: Child 연결 오류")
            linked_children.append(cid)
    role_children = {oid for oid, obj in graph_objects.items() if obj["role"] == "child"}
    if (len(linked_children) != len(set(linked_children))
            or set(linked_children) != role_children or source_ids & role_children):
        raise ValueError(f"{name}: Child 배정 중복 또는 누락")
