"""Restrict rebuilt masked graphs to the reviewed component scope."""
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/hm3d/wcojb4TFT35"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index(items, key):
    result = {item[key]: item for item in items}
    require(len(result) == len(items), f"Duplicate {key}")
    return result


def main():
    config_path = ROOT / "configs/data/pilot_targets.json"
    catalog_path = ROOT / "outputs/hm3d/wcojb4TFT35.target_catalog.json"
    targets = read(config_path)["target_categories"]
    require(len(targets) == len(set(targets)), "Duplicate target categories")
    catalog = index(read(catalog_path)["targets"], "target_category")

    source_paths = {
        "graph": ROOT / "outputs/hm3d/wcojb4TFT35.workspace_graph.v2.json",
        "membership": BASE / "component_membership.review.json",
        "report": BASE / "component_room_candidates.review.json",
        "selection": BASE / "stair_triangle_selection.review.json",
        "floor_plan": BASE / "floor_environments.json",
        "decisions": BASE / "component_membership.decisions.review.json",
    }
    source_hashes = {key: sha(path) for key, path in source_paths.items()}

    components = {}
    for name in ("A", "B"):
        path = BASE / "component_graphs.review" / f"{name}.workspace_graph.review.json"
        graph = read(path)
        require(
            graph["component_review"]["source_sha256"] == source_hashes,
            f"{name}: component sources changed",
        )
        require(graph["component_review"]["component"] == name, "Wrong component")
        components[name] = (path, index(graph["objects"], "object_id"))

    require(
        not set(components["A"][1]) & set(components["B"][1]),
        "A/B object overlap",
    )

    artifacts = {}
    rows = []
    input_hashes = {}

    for target in targets:
        require(
            isinstance(target, str)
            and target
            and all(c.isascii() and (c.isalnum() or c in "_-") for c in target),
            f"Invalid target filename: {target}",
        )
        require(target in catalog, f"Missing catalog target: {target}")
        instances = index(catalog[target]["instances"], "object_id")
        gt_ids = set(instances)

        path = BASE / "masked_graphs" / f"{target}.workspace_graph.json"
        masked = read(path)
        input_hashes[target] = sha(path)
        require(masked["coordinate_frame"] == "athome_z_up", "Wrong frame")
        require(
            masked["provenance"]["target_category"] == target,
            f"{target}: provenance mismatch",
        )
        objects = index(masked["objects"], "object_id")
        workspaces = index(masked["workspaces"], "workspace_id")
        rooms = index(masked["rooms"], "room_id")
        require(not set(objects) & gt_ids, f"{target}: GT object remains")

        for name, (component_path, original_objects) in components.items():
            scope = set(original_objects)
            expected = scope - gt_ids
            kept = set(objects) & scope
            require(
                kept == expected,
                f"{target}/{name}: non-target objects missing: {sorted(expected-kept)}",
            )

            for oid in kept:
                require(
                    objects[oid]["room_id"] == original_objects[oid]["room_id"]
                    and objects[oid]["bbox"] == original_objects[oid]["bbox"],
                    f"{target}/{name}: object geometry or room changed: {oid}",
                )

            selected_ws = {}
            for wid, ws in workspaces.items():
                linked = [ws["source_object_id"], *ws["child_object_ids"]]
                require(len(linked) == len(set(linked)), f"Duplicate links: {wid}")
                require(set(linked) <= objects.keys(), f"Missing object: {wid}")

                if not set(linked) & kept:
                    continue
                require(
                    set(linked) <= kept,
                    f"{target}/{name}: workspace crosses component boundary: {wid}",
                )
                require(
                    all(objects[oid]["room_id"] == ws["room_id"] for oid in linked),
                    f"Workspace room mismatch: {wid}",
                )
                source = objects[ws["source_object_id"]]
                require(
                    source["role"] == "source"
                    and source["parent_type"] == "room"
                    and source["workspace_id"] == wid,
                    f"Invalid source: {wid}",
                )
                for cid in ws["child_object_ids"]:
                    child = objects[cid]
                    require(
                        child["role"] == "child"
                        and child["parent_type"] == "workspace"
                        and child["parent_id"] == wid
                        and child["workspace_id"] == wid,
                        f"Invalid child: {cid}",
                    )
                selected_ws[wid] = ws

            rids = {objects[oid]["room_id"] for oid in kept}
            require(rids <= rooms.keys(), "Missing room")
            for oid in kept:
                obj = objects[oid]
                if obj["parent_type"] == "room":
                    require(obj["parent_id"] == obj["room_id"], f"Bad room parent: {oid}")
                else:
                    wid = obj["parent_id"]
                    require(
                        obj["parent_type"] == "workspace"
                        and wid in selected_ws
                        and oid in selected_ws[wid]["child_object_ids"],
                        f"Bad workspace parent: {oid}",
                    )
                wid = obj.get("workspace_id")
                require(wid is None or wid in selected_ws, f"Dangling workspace: {oid}")

            room_records = []
            for rid in sorted(rids):
                room = copy.deepcopy(rooms[rid])
                room_ws = {
                    wid for wid, ws in selected_ws.items() if ws["room_id"] == rid
                }
                expected_lists = {
                    "workspace_ids": room_ws,
                    "source_object_ids": {
                        selected_ws[wid]["source_object_id"] for wid in room_ws
                    },
                    "standalone_object_ids": {
                        oid for oid in kept
                        if objects[oid]["room_id"] == rid
                        and objects[oid]["role"] == "standalone"
                    },
                    "direct_object_ids": {
                        oid for oid in kept
                        if objects[oid]["parent_type"] == "room"
                        and objects[oid]["parent_id"] == rid
                    },
                }
                for field, expected_ids in expected_lists.items():
                    allowed = set(selected_ws) if field == "workspace_ids" else kept
                    filtered = [oid for oid in room[field] if oid in allowed]
                    require(
                        len(filtered) == len(set(filtered))
                        and set(filtered) == expected_ids,
                        f"Room references differ: {target}/{name}/{rid}/{field}",
                    )
                    room[field] = sorted(filtered)
                room_records.append(room)

            result = copy.deepcopy(masked)
            result["rooms"] = room_records
            result["workspaces"] = [
                copy.deepcopy(selected_ws[wid]) for wid in sorted(selected_ws)
            ]
            result["objects"] = [copy.deepcopy(objects[oid]) for oid in sorted(kept)]

            # Keep audit metadata out of the planner graph artifact.
            result.pop("provenance", None)
            result.pop("component_review", None)
            artifacts[f"{name}.{target}.workspace_graph.json"] = result

            rows.append({
                "component": name,
                "target": target,
                "rooms": len(room_records),
                "workspaces": len(selected_ws),
                "objects": len(kept),
                "target_instances_in_scope": len(scope & gt_ids),
                "component_graph_sha256": sha(component_path),
                "masked_graph_sha256": input_hashes[target],
            })

    artifacts["manifest.review.json"] = {
        "status": "development_masked_graphs_not_final",
        "source_sha256": source_hashes,
        "config_sha256": sha(config_path),
        "catalog_sha256": sha(catalog_path),
        "masked_graph_sha256": input_hashes,
        "results": rows,
        "navigation_rechecked": False,
        "observation_checked": False,
        "sft_samples_generated": False,
        "note": (
            "Audit metadata only; do not pass this manifest to the planner. "
            "Zero in-scope targets must not be treated as positive search episodes."
        ),
    }

    output = BASE / "component_masked_graphs.review"
    if output.exists():
        require(
            {p.name for p in output.iterdir()} == set(artifacts),
            "Existing output contains different files",
        )
        for filename, content in artifacts.items():
            require(read(output / filename) == content,
                    f"Existing output differs: {filename}")
        print("동일한 결과가 이미 저장되어 있습니다.")
    else:
        temporary = Path(tempfile.mkdtemp(prefix=".component_masked-", dir=BASE))
        try:
            for filename, content in artifacts.items():
                (temporary / filename).write_text(
                    json.dumps(content, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n",
                    encoding="utf-8",
                )
            os.rename(temporary, output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    for row in rows:
        print(
            f"{row['component']} / {row['target']}: "
            f"Workspace {row['workspaces']}, Object {row['objects']}, "
            f"영역 내 GT {row['target_instances_in_scope']}"
        )
    print("목표 객체 제거·객체 보존·그래프 참조 검사: 통과")
    print("저장:", output)
    print("원본 그래프·GT catalog·격자 지도 변경 없음")


if __name__ == "__main__":
    main()
