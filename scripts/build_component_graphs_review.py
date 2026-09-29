"""Build component review graphs without modifying source data."""
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from statistics import median

from athome.data.hm3d.layout import current_layout
from athome.data.hm3d.target_masking import normalize_tag

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = current_layout()
BASE = LAYOUT.scene_dir


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def index_by(items, key):
    result = {item[key]: item for item in items}
    require(len(result) == len(items), f"Duplicate {key}")
    return result


def main():
    paths = {
        "graph": LAYOUT.graph,
        "membership": LAYOUT.membership,
        "report": LAYOUT.room_report,
        "selection": LAYOUT.stair_selection,
        "floor_plan": LAYOUT.floors,
        "decisions": LAYOUT.decisions,
    }
    raw = {key: path.read_bytes() for key, path in paths.items()}
    hashes = {
        key: hashlib.sha256(value).hexdigest()
        for key, value in raw.items()
    }
    data = {key: json.loads(value) for key, value in raw.items()}
    graph = data["graph"]
    membership = data["membership"]
    decisions = data["decisions"]

    require(graph["coordinate_frame"] == "athome_z_up", "Wrong frame")
    for key in ("graph", "membership", "report", "selection", "floor_plan"):
        require(
            decisions["source_sha256"][key] == hashes[key],
            f"Source changed since review: {key}",
        )
    for field, key in (
        ("source_graph_sha256", "graph"),
        ("source_component_report_sha256", "report"),
    ):
        require(membership[field] == hashes[key], f"Stale membership: {key}")
    for field, key in (
        ("graph_sha256", "graph"),
        ("selection_sha256", "selection"),
        ("floor_plan_sha256", "floor_plan"),
    ):
        require(data["report"][field] == hashes[key], f"Stale report: {key}")

    objects = index_by(graph["objects"], "object_id")
    workspaces = index_by(graph["workspaces"], "workspace_id")
    rooms = index_by(graph["rooms"], "room_id")
    pending = index_by(membership["unresolved_objects"], "object_id")
    reviews = index_by(decisions["objects"], "object_id")
    require(set(pending) == set(reviews), "Review coverage mismatch")

    components = index_by(membership["components"], "component")
    require(set(components) == {c["component"] for c in data["report"]["components"]},
            "Unexpected components")
    selected = {}
    for name, component in components.items():
        ids = component["provisional_object_ids"]
        require(len(ids) == len(set(ids)), f"{name}: duplicate objects")
        selected[name] = set(ids)
        require(selected[name] <= objects.keys(), f"{name}: unknown objects")
        require(not selected[name] & pending.keys(), "Assignment overlap")

    deferred = []
    for oid, record in reviews.items():
        require(oid in objects, f"Unknown reviewed object: {oid}")
        require(
            record["source_room_id"] == objects[oid]["room_id"],
            f"Room mismatch: {oid}",
        )
        name = record["proposed_component"]
        if record["status"] == "proposed_assignment":
            require(name in selected, f"Invalid component: {oid}")
            require(
                name in pending[oid]["room_component_candidates"],
                f"Invalid component candidate: {oid}",
            )
            selected[name].add(oid)
        else:
            require(
                record["status"] == "deferred" and name is None,
                f"Invalid review status: {oid}",
            )
            deferred.append(record)

    # A Workspace (source surface + objects on it) stays in one component:
    # it follows its objects when they were all assigned to the same one,
    # otherwise the whole Workspace is deferred.
    extra_workspaces = {name: [] for name in selected}
    for item in membership["workspace_review"]:
        linked = item["linked_object_ids"]
        homes = {name for name, ids in selected.items() for oid in linked if oid in ids}
        if len(homes) == 1 and all(oid in selected[next(iter(homes))] for oid in linked):
            extra_workspaces[homes.pop()].append(item["workspace_id"])
            continue
        for oid in linked:
            for ids in selected.values():
                ids.discard(oid)
            if oid not in {d["object_id"] for d in deferred}:
                deferred.append({"object_id": oid, "source_room_id": objects[oid]["room_id"],
                                 "status": "deferred", "proposed_component": None,
                                 "reason": "workspace_objects_not_in_one_component",
                                 "workspace_id": item["workspace_id"]})

    included = set().union(*selected.values())
    require(sum(map(len, selected.values())) == len(included), "Component object overlap")
    deferred_ids = {item["object_id"] for item in deferred}
    require(not included & deferred_ids, "Deferred object included")

    report_components = {c["component"]: c for c in data["report"]["components"]}
    artifacts = {}
    counts = {}
    for name in sorted(selected):
        oids = selected[name]
        workspace_list = components[name]["provisional_workspace_ids"] + extra_workspaces[name]
        wids = set(workspace_list)
        require(len(wids) == len(workspace_list), "Duplicate workspace IDs")
        require(wids <= workspaces.keys(), f"{name}: unknown workspace")
        rids = {objects[oid]["room_id"] for oid in oids}
        require(rids <= rooms.keys(), f"{name}: unknown room")

        for wid in wids:
            ws = workspaces[wid]
            linked = [ws["source_object_id"], *ws["child_object_ids"]]
            require(len(linked) == len(set(linked)), f"Duplicate links: {wid}")
            require(set(linked) <= oids, f"Missing workspace objects: {wid}")
            require(
                all(objects[oid]["room_id"] == ws["room_id"] for oid in linked),
                f"Workspace room mismatch: {wid}",
            )
            for oid in ws["child_object_ids"]:
                obj = objects[oid]
                require(
                    obj["parent_type"] == "workspace"
                    and obj["parent_id"] == wid
                    and obj["workspace_id"] == wid,
                    f"Child reference mismatch: {oid}",
                )

        for oid in oids:
            obj = objects[oid]
            if obj["parent_type"] == "room":
                require(obj["parent_id"] == obj["room_id"], f"Bad parent: {oid}")
            elif obj["parent_type"] == "workspace":
                wid = obj["parent_id"]
                require(wid in wids, f"Missing parent workspace: {oid}")
                require(
                    oid in workspaces[wid]["child_object_ids"],
                    f"Missing reverse child link: {oid}",
                )
            else:
                raise RuntimeError(f"Unknown parent type: {oid}")
            wid = obj.get("workspace_id")
            require(wid is None or wid in wids, f"Missing workspace: {oid}")

        room_records = []
        for rid in sorted(rids):
            room = copy.deepcopy(rooms[rid])
            for field, allowed in (
                ("workspace_ids", wids),
                ("source_object_ids", oids),
                ("standalone_object_ids", oids),
                ("direct_object_ids", oids),
            ):
                room[field] = [value for value in room[field] if value in allowed]
                require(
                    len(room[field]) == len(set(room[field])),
                    f"Duplicate room references: {rid}/{field}",
                )

            expected_ws = {wid for wid in wids if workspaces[wid]["room_id"] == rid}
            expected_direct = {
                oid for oid in oids
                if objects[oid]["parent_type"] == "room"
                and objects[oid]["parent_id"] == rid
            }
            expected_sources = {
                workspaces[wid]["source_object_id"] for wid in expected_ws
            }
            expected_standalone = {
                oid for oid in oids
                if objects[oid]["room_id"] == rid
                and objects[oid]["role"] == "standalone"
            }
            for field, expected in (
                ("workspace_ids", expected_ws),
                ("direct_object_ids", expected_direct),
                ("source_object_ids", expected_sources),
                ("standalone_object_ids", expected_standalone),
            ):
                require(set(room[field]) == expected, f"Room link mismatch: {rid}/{field}")
            if not any(objects[oid]["room_id"] == rid and normalize_tag(objects[oid]["semantic_tag"]) == "floor"
                       for oid in oids):
                # The room's floor object is not in this component (e.g. a floor
                # shared by two components is deferred): reference height from the
                # floors this component stands on (room candidates of the report).
                heights = sorted(c["floor_center_z_m"] for c in report_components[name]["room_candidates"]
                                 if c["room_id"] == rid)
                require(bool(heights), f"No floor evidence: {name}/{rid}")
                room["floor_z_m"] = float(median(heights))
            room_records.append(room)

        result = copy.deepcopy(graph)
        result["rooms"] = room_records
        result["workspaces"] = [copy.deepcopy(workspaces[wid]) for wid in sorted(wids)]
        result["objects"] = [copy.deepcopy(objects[oid]) for oid in sorted(oids)]
        result["component_review"] = {
            "status": "development_review_not_final",
            "component": name,
            "source_sha256": hashes,
            "room_assignment_verified": False,
            "navigation_accessibility_verified": False,
            "search_location_policy_applied": False,
            "note": "Room IDs are source IDs; use component + room_id across files.",
        }
        artifacts[f"{name}.workspace_graph.review.json"] = result
        counts[name] = {
            "rooms": len(room_records),
            "workspaces": len(wids),
            "objects": len(oids),
        }

    artifacts["manifest.review.json"] = {
        "status": "development_review_not_final",
        "scene_id": LAYOUT.scene_id,
        "source_sha256": hashes,
        "counts": counts,
        "deferred_objects": deferred,
        "outside_current_component_scope_object_ids": sorted(
            set(objects) - included - deferred_ids
        ),
        "note": "Out-of-scope objects are not classified as inaccessible.",
    }

    # Validate everything before publishing the output directory.
    output = BASE / "component_graphs.review"
    if output.exists():
        require(
            {p.name for p in output.iterdir()} == set(artifacts),
            f"Existing output contents differ: {output}",
        )
        for filename, content in artifacts.items():
            require(
                json.loads((output / filename).read_text(encoding="utf-8")) == content,
                f"Existing output differs; not overwritten: {filename}",
            )
        print("동일한 결과가 이미 저장되어 있습니다.")
    else:
        temporary = Path(tempfile.mkdtemp(prefix=".component_graphs-", dir=BASE))
        try:
            for filename, content in artifacts.items():
                (temporary / filename).write_text(
                    json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
            os.rename(temporary, output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    for name, count in counts.items():
        print(f"{name}: Room {count['rooms']}, Workspace {count['workspaces']}, Object {count['objects']}")
    print("보류 객체:", len(deferred))
    print("객체 중복 및 Room·Workspace 참조 검사: 통과")
    print("저장 위치:", output)
    print("원본 파일 변경 없음")


if __name__ == "__main__":
    main()
