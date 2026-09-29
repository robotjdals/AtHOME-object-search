"""Check component review graphs using shared navigation functions."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from athome.navigation.grid import GridMap
from athome.navigation.goal_poses import goal_candidates
from athome.navigation.path_cost import shortest_paths, extract_path
from athome.data.hm3d.layout import current_layout

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = current_layout()
BASE = LAYOUT.scene_dir
GRID = LAYOUT.grid_dir
GRAPHS = BASE / "component_graphs.review"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def indexed(items, key):
    result = {item[key]: item for item in items}
    require(len(result) == len(items), f"Duplicate {key}")
    return result


def main():
    candidate_path = GRID / "workspace_goal_candidates.clearance_0p1.review.json"
    start_path = GRID / "path_review/workspace_paths.review.json"
    baseline = read(candidate_path)
    starts = read(start_path)
    require(
        starts["candidate_file_sha256"] == sha(candidate_path),
        "Recorded starts refer to another candidate file.",
    )
    settings = indexed(baseline["components"], "component")
    starts_by_name = indexed(starts["components"], "component")
    require(set(settings) == set(starts_by_name) == set(LAYOUT.component_names()), "Wrong components")

    sources = {
        "graph": LAYOUT.graph,
        "membership": LAYOUT.membership,
        "report": LAYOUT.room_report,
        "selection": LAYOUT.stair_selection,
        "floor_plan": LAYOUT.floors,
        "decisions": LAYOUT.decisions,
    }
    source_hashes = {key: sha(path) for key, path in sources.items()}
    require(
        baseline["source_graph_sha256"] == source_hashes["graph"]
        and baseline["membership_sha256"] == source_hashes["membership"]
        and baseline["component_report_sha256"] == source_hashes["report"],
        "Baseline candidate sources changed.",
    )

    report = {
        "status": "development_navigation_review",
        "algorithm": "dijkstra",
        "movement": "8_connected_no_corner_cutting",
        "coordinate_frame": "athome_z_up",
        "cost_unit": "meter",
        "start_policy": "recorded_validation_cell_without_snapping",
        "candidate_file_sha256": sha(candidate_path),
        "start_review_sha256": sha(start_path),
        "source_sha256": source_hashes,
        "visibility_checked": False,
        "physical_robot_feasibility_verified": False,
        "components": [],
    }

    for name in LAYOUT.component_names():
        graph_path = GRAPHS / f"{name}.workspace_graph.review.json"
        graph = read(graph_path)
        require(graph["component_review"]["component"] == name, "Wrong graph")
        require(
            graph["component_review"]["source_sha256"] == source_hashes,
            f"{name}: component graph sources changed.",
        )

        cfg = settings[name]
        grid_path = GRID / f"{name}.grid.npz"
        meta_path = GRID / f"{name}.metadata.json"
        require(sha(grid_path) == cfg["grid_sha256"], f"{name}: grid changed")
        require(sha(meta_path) == cfg["metadata_sha256"], f"{name}: metadata changed")
        meta = read(meta_path)
        require(
            meta["coordinate_frame"] == "athome_z_up"
            and meta["source_report_sha256"] == source_hashes["report"],
            f"{name}: incompatible metadata",
        )

        with np.load(grid_path, allow_pickle=False) as saved:
            free = saved["free"].copy()
            origin = np.asarray(saved["origin_xy_m"], dtype=float)
            resolution = float(saved["resolution_m"].item())
        require(
            free.ndim == 2 and free.dtype == bool and free.any(),
            f"{name}: invalid free grid",
        )
        require(
            origin.shape == (2,) and np.isfinite(origin).all()
            and math.isfinite(resolution) and resolution > 0,
            f"{name}: invalid grid coordinates",
        )
        require(
            list(free.shape) == meta["array_shape_rows_cols"]
            and np.allclose(origin, meta["origin_xy_m"], rtol=0, atol=1e-9)
            and math.isclose(resolution, meta["resolution_m"], abs_tol=1e-9),
            f"{name}: grid metadata mismatch",
        )
        grid = GridMap(free, tuple(origin), resolution)
        recorded_start = starts_by_name[name]["start_row_col"]
        require(
            len(recorded_start) == 2
            and all(type(value) is int for value in recorded_start),
            f"{name}: invalid start indices",
        )
        start = tuple(recorded_start)
        require(grid.is_free(start), f"{name}: start is not free")

        offset = float(cfg["footprint_offset_m"])
        require(math.isfinite(offset) and offset > 0, "Invalid goal offset")
        require(
            math.isclose(float(cfg["candidate_band_width_m"]), resolution,
                         rel_tol=0, abs_tol=1e-9),
            "Candidate band differs from shared goal generator.",
        )

        objects = indexed(graph["objects"], "object_id")
        workspaces = indexed(graph["workspaces"], "workspace_id")
        baseline_workspaces = indexed(cfg["workspaces"], "workspace_id")
        require(set(workspaces) == set(baseline_workspaces),
                f"{name}: workspace set changed")

        candidates = {}
        for wid, ws in workspaces.items():
            source_id = ws["source_object_id"]
            require(source_id in objects, f"Missing source: {wid}")
            require(
                baseline_workspaces[wid]["source_object_id"] == source_id,
                f"Source mismatch: {wid}",
            )
            bbox = objects[source_id]["bbox"]
            lo = np.asarray(bbox["min"], dtype=float)
            hi = np.asarray(bbox["max"], dtype=float)
            require(
                lo.shape == hi.shape == (3,)
                and np.isfinite([lo, hi]).all() and np.all(hi >= lo),
                f"Invalid bbox: {source_id}",
            )
            candidates[wid] = goal_candidates(grid, lo[:2], hi[:2], offset,
                                              float(cfg.get("goal_clearance_m", 0.0)),
                                              cfg.get("goal_max_offset_m"))

            expected = {
                (item["row"], item["col"])
                for item in baseline_workspaces[wid]["candidates"]
            }
            actual = {cell for cell, _ in candidates[wid]}
            require(actual == expected, f"Candidate cells differ: {wid}")

        goals = {cell for items in candidates.values() for cell, _ in items}
        costs, parents = shortest_paths(free, start, goals, resolution)
        results = []

        for wid in sorted(workspaces):
            reachable = [
                (costs[cell], cell, pose)
                for cell, pose in candidates[wid] if cell in costs
            ]
            record = {
                "workspace_id": wid,
                "source_object_id": workspaces[wid]["source_object_id"],
                "candidate_count": len(candidates[wid]),
                "reachable_candidate_count": len(reachable),
            }
            if not reachable:
                record["status"] = (
                    "no_candidate_at_configured_offset"
                    if not candidates[wid] else "no_path_from_validation_start"
                )
            else:
                cost, goal, pose = min(reachable, key=lambda item: (item[0], item[1]))
                path = extract_path(parents, start, goal)
                require(path[0] == start and path[-1] == goal, "Invalid endpoints")
                require(all(grid.is_free(cell) for cell in path), "Blocked path")
                measured = 0.0
                for previous, current in zip(path, path[1:]):
                    dr, dc = current[0] - previous[0], current[1] - previous[1]
                    require(max(abs(dr), abs(dc)) == 1, "Invalid path step")
                    if dr and dc:
                        require(
                            grid.is_free((previous[0] + dr, previous[1]))
                            and grid.is_free((previous[0], previous[1] + dc)),
                            "Corner cutting",
                        )
                    measured += math.hypot(dr, dc) * resolution
                require(math.isclose(cost, measured, rel_tol=0, abs_tol=1e-8),
                        "Path length mismatch")
                record.update({
                    "status": "path_found",
                    "path_cost_m": float(cost),
                    "goal_pose": {
                        "x_m": float(pose.x),
                        "y_m": float(pose.y),
                        "yaw_rad": float(pose.yaw),
                    },
                    "path_row_col": [list(cell) for cell in path],
                    "path_xy_m": [list(grid.to_xy(cell)) for cell in path],
                })
            results.append(record)
            detail = (
                f"{record['path_cost_m']:.3f} m"
                if record["status"] == "path_found" else record["status"]
            )
            print(f"{name} / {record['source_object_id']}: {detail}")

        report["components"].append({
            "component": name,
            "component_graph_sha256": sha(graph_path),
            "grid_sha256": sha(grid_path),
            "metadata_sha256": sha(meta_path),
            "start_row_col": list(start),
            "start_xy_m": list(grid.to_xy(start)),
            "footprint_offset_m": offset,
            **{k: cfg[k] for k in ("goal_clearance_m", "goal_max_offset_m") if k in cfg},
            "workspaces": results,
        })

    all_results = [
        item for component in report["components"] for item in component["workspaces"]
    ]
    report["summary"] = {
        "checked": len(all_results),
        "path_found": sum(item["status"] == "path_found" for item in all_results),
        "no_candidate": sum(
            item["status"] == "no_candidate_at_configured_offset"
            for item in all_results
        ),
        "no_path": sum(
            item["status"] == "no_path_from_validation_start"
            for item in all_results
        ),
    }

    output = BASE / "component_navigation.review.json"
    if output.exists():
        require(read(output) == report, f"Different existing report: {output}")
        print("동일한 결과가 이미 저장되어 있습니다.")
    else:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")

    print("\n요약:", report["summary"])
    print("저장:", output)
    print("원본 그래프·격자 지도 변경 없음")


if __name__ == "__main__":
    main()
