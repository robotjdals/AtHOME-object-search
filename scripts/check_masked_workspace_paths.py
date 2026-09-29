import hashlib
import json
import math
from pathlib import Path

import numpy as np

from check_workspace_paths import astar
from athome.data.hm3d.layout import current_layout


ROOT = Path(__file__).resolve().parents[1]
LAYOUT = current_layout()
SCENE = LAYOUT.scene_id
BASE = LAYOUT.scene_dir
GRID_DIR = LAYOUT.grid_dir
OUTPUT_DIR = GRID_DIR / "masked_workspace_checks"

GRAPH_PATH = LAYOUT.graph
MEMBERSHIP_PATH = BASE / "component_membership.review.json"
CANDIDATE_PATH = (
    GRID_DIR / "workspace_goal_candidates.clearance_0p1.review.json"
)
PATH_REVIEW_PATH = GRID_DIR / "path_review" / "workspace_paths.review.json"

TARGETS = json.loads((LAYOUT.targets).read_text(encoding="utf-8"))["target_categories"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def index_by(items, key):
    result = {}
    for item in items:
        value = item[key]
        require(value not in result, f"Duplicate {key}: {value}")
        result[value] = item
    return result


membership = read_json(MEMBERSHIP_PATH)
baseline = read_json(CANDIDATE_PATH)
path_review = read_json(PATH_REVIEW_PATH)
original_graph = read_json(GRAPH_PATH)

require(
    membership["source_graph_sha256"] == sha256(GRAPH_PATH),
    "Original graph differs from the membership source.",
)
require(
    baseline["membership_sha256"] == sha256(MEMBERSHIP_PATH),
    "Membership changed after baseline candidate generation.",
)
require(
    baseline["source_graph_sha256"] == sha256(GRAPH_PATH),
    "Baseline candidates refer to another graph.",
)
require(
    path_review["candidate_file_sha256"] == sha256(CANDIDATE_PATH),
    "Path review refers to another candidate file.",
)

original_objects = index_by(original_graph["objects"], "object_id")
original_workspaces = index_by(original_graph["workspaces"], "workspace_id")

# Final component membership (component graphs, after membership decisions).
object_component = {}
for name in LAYOUT.component_names():
    component_graph = read_json(
        BASE / "component_graphs.review" / f"{name}.workspace_graph.review.json")
    for obj in component_graph["objects"]:
        object_id = obj["object_id"]
        require(
            object_id not in object_component,
            f"Object assigned to multiple components: {object_id}",
        )
        object_component[object_id] = name

unresolved_ids = {
    item["object_id"]
    for item in read_json(BASE / "component_graphs.review/manifest.review.json")["deferred_objects"]
}
require(
    not unresolved_ids.intersection(object_component),
    "An unresolved object also has a component assignment.",
)

baseline_components = index_by(baseline["components"], "component")
review_components = index_by(path_review["components"], "component")
maps = {}

for name, settings in baseline_components.items():
    grid_path = GRID_DIR / f"{name}.grid.npz"
    metadata_path = GRID_DIR / f"{name}.metadata.json"

    require(
        sha256(grid_path) == settings["grid_sha256"],
        f"Grid changed: {name}",
    )
    require(
        sha256(metadata_path) == settings["metadata_sha256"],
        f"Grid metadata changed: {name}",
    )

    with np.load(grid_path, allow_pickle=False) as data:
        free = data["free"].astype(bool)
        origin = data["origin_xy_m"].astype(float)
        resolution = float(data["resolution_m"])

    start = tuple(map(int, review_components[name]["start_row_col"]))
    require(
        0 <= start[0] < free.shape[0]
        and 0 <= start[1] < free.shape[1]
        and free[start],
        f"Invalid validation start: {name}",
    )

    rows, cols = np.nonzero(free)
    xy = origin + resolution * np.column_stack((cols + 0.5, rows + 0.5))

    maps[name] = {
        "free": free,
        "origin": origin,
        "resolution": resolution,
        "start": start,
        "rows": rows,
        "cols": cols,
        "xy": xy,
        "settings": settings,
    }


def check_workspace(workspace, source, name):
    grid = maps[name]
    settings = grid["settings"]
    lower = np.asarray(source["bbox"]["min"], dtype=float)[:2]
    upper = np.asarray(source["bbox"]["max"], dtype=float)[:2]
    center = np.asarray(source["bbox"]["center"], dtype=float)[:2]

    require(
        np.isfinite(np.concatenate((lower, upper, center))).all()
        and np.all(upper >= lower),
        f"Invalid source footprint: {source['object_id']}",
    )

    xy = grid["xy"]
    delta = np.maximum(np.maximum(lower - xy, xy - upper), 0.0)
    distances = np.linalg.norm(delta, axis=1)
    outside = np.any((xy < lower) | (xy > upper), axis=1)

    offset = float(settings["footprint_offset_m"])
    width = float(settings["candidate_band_width_m"])
    selected = np.flatnonzero(
        outside & (distances >= offset) & (distances < offset + width)
    )

    candidates = []
    for index in selected:
        row = int(grid["rows"][index])
        col = int(grid["cols"][index])
        x, y = map(float, xy[index])
        candidates.append({
            "row": row,
            "col": col,
            "x_m": x,
            "y_m": y,
            "yaw_rad": math.atan2(center[1] - y, center[0] - x),
            "footprint_distance_m": float(distances[index]),
        })

    result = {
        "workspace_id": workspace["workspace_id"],
        "source_object_id": source["object_id"],
        "room_id": workspace["room_id"],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }

    if not candidates:
        external = distances[outside]
        result.update({
            "status": "no_candidate_at_configured_offset",
            "nearest_external_free_cell_distance_m": (
                float(external.min()) if external.size else None
            ),
        })
        return result

    goals = {(item["row"], item["col"]) for item in candidates}
    cost, path = astar(
        grid["free"], grid["start"], goals, grid["resolution"]
    )

    if cost is None:
        result["status"] = "no_path_from_validation_start"
        return result

    cells = [tuple(map(int, cell)) for cell in path]
    require(
        cells and cells[0] == grid["start"] and cells[-1] in goals,
        f"Invalid path endpoints: {source['object_id']}",
    )

    measured_cost = 0.0
    for row, col in cells:
        require(grid["free"][row, col], "Path enters a blocked cell.")

    for previous, current in zip(cells, cells[1:]):
        dr = current[0] - previous[0]
        dc = current[1] - previous[1]
        require(
            max(abs(dr), abs(dc)) == 1,
            "Invalid path step.",
        )
        if dr and dc:
            require(
                grid["free"][previous[0] + dr, previous[1]]
                and grid["free"][previous[0], previous[1] + dc],
                "Path cuts a blocked corner.",
            )
        measured_cost += math.hypot(dr, dc) * grid["resolution"]

    require(
        math.isclose(float(cost), measured_cost, abs_tol=1e-8),
        "Path cost does not match path length.",
    )

    chosen = next(
        item for item in candidates
        if (item["row"], item["col"]) == cells[-1]
    )
    result.update({
        "status": "path_found",
        "astar_cost_m": float(cost),
        "selected_candidate": chosen,
        "path_row_col": [list(cell) for cell in cells],
    })
    return result


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary = []

for target in TARGETS:
    graph_path = LAYOUT.masked_graphs / f"{target}.workspace_graph.json"
    graph = read_json(graph_path)
    objects = index_by(graph["objects"], "object_id")
    workspaces = index_by(graph["workspaces"], "workspace_id")

    # Masking may remove objects, but must not move retained objects.
    for object_id, obj in objects.items():
        require(
            object_id in original_objects,
            f"Unexpected object in masked graph: {object_id}",
        )
        original = original_objects[object_id]
        require(
            obj["room_id"] == original["room_id"],
            f"Room changed: {object_id}",
        )
        for field in ("center", "size", "min", "max"):
            require(
                np.allclose(
                    obj["bbox"][field],
                    original["bbox"][field],
                    rtol=0,
                    atol=1e-7,
                ),
                f"Geometry changed: {object_id}/{field}",
            )

    grouped = {name: [] for name in maps}
    excluded = []

    for workspace in workspaces.values():
        source_id = workspace["source_object_id"]
        require(source_id in objects, f"Missing source: {source_id}")

        name = object_component.get(source_id)
        if name is None:
            excluded.append({
                "workspace_id": workspace["workspace_id"],
                "source_object_id": source_id,
                "reason": (
                    "unresolved_component_membership"
                    if source_id in unresolved_ids
                    else "outside_current_A_B_scope"
                ),
            })
            continue

        for child_id in workspace["child_object_ids"]:
            require(child_id in objects, f"Missing child: {child_id}")
            require(
                object_component.get(child_id) == name,
                f"Child crosses component scope: {source_id} -> {child_id}",
            )

        grouped[name].append(workspace)

    report = {
        "status": "provisional_masked_workspace_path_review",
        "scene_id": SCENE,
        "target": target,
        "coordinate_frame": "athome_z_up",
        "masked_graph_sha256": sha256(graph_path),
        "membership_sha256": sha256(MEMBERSHIP_PATH),
        "baseline_candidates_sha256": sha256(CANDIDATE_PATH),
        "baseline_path_review_sha256": sha256(PATH_REVIEW_PATH),
        "scope": "provisional_A_B_workspace_sources_only",
        "start_policy": "reuse_baseline_validation_start",
        "movement": "8_connected_no_corner_cutting",
        "visibility_checked": False,
        "physical_robot_feasibility_verified": False,
        "room_assignment_verified": False,
        "components": [],
        "out_of_scope_workspaces": excluded,
    }

    total = found = missing = no_path = 0
    print(f"\n=== Target: {target} ===")

    for name, selected_workspaces in grouped.items():
        results = []
        original_ids = {
            wid for wid, workspace in original_workspaces.items()
            if object_component.get(workspace["source_object_id"]) == name
        }
        current_ids = {item["workspace_id"] for item in selected_workspaces}

        for workspace in selected_workspaces:
            result = check_workspace(
                workspace, objects[workspace["source_object_id"]], name
            )
            results.append(result)
            total += 1

            if result["status"] == "path_found":
                found += 1
            elif result["status"] == "no_candidate_at_configured_offset":
                missing += 1
                print(f"  {name}: no candidate: {result['source_object_id']}")
            else:
                no_path += 1
                print(f"  {name}: no path: {result['source_object_id']}")

        grid = maps[name]
        report["components"].append({
            "component": name,
            "grid_sha256": grid["settings"]["grid_sha256"],
            "metadata_sha256": grid["settings"]["metadata_sha256"],
            "start_row_col": list(grid["start"]),
            "footprint_offset_m": grid["settings"]["footprint_offset_m"],
            "candidate_band_width_m": grid["settings"]["candidate_band_width_m"],
            "added_workspace_ids": sorted(current_ids - original_ids),
            "removed_workspace_ids": sorted(original_ids - current_ids),
            "workspaces": results,
        })
        print(
            f"  {name}: {len(results)} workspaces, "
            f"added={len(current_ids - original_ids)}, "
            f"removed={len(original_ids - current_ids)}"
        )

    row = {
        "target": target,
        "workspaces_checked": total,
        "paths_found": found,
        "no_candidate": missing,
        "no_path_from_validation_start": no_path,
        "outside_scope": len(excluded),
        "unresolved_workspace_sources": sum(
            item["reason"] == "unresolved_component_membership"
            for item in excluded
        ),
    }
    report["summary"] = row
    save_json(OUTPUT_DIR / f"{target}.review.json", report)
    summary.append(row)

    print(
        f"  Checked={total}, paths={found}, "
        f"no_candidate={missing}, no_path={no_path}"
    )

save_json(
    OUTPUT_DIR / "summary.review.json",
    {
        "status": "development_review_only",
        "results": summary,
        "unresolved_object_count": len(unresolved_ids),
        "sft_samples_generated": False,
    },
)

print("\n=== Completed ===")
print("Targets checked:", len(summary))
print("Unresolved membership objects:", len(unresolved_ids))
print("Saved:", OUTPUT_DIR)
print("Source graphs, grids, and membership files were not modified.")
print("These costs apply only to the recorded validation starts.")
