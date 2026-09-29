import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from athome.config import goal_geometry, robot_geometry
from athome.navigation.grid import GridMap
from athome.data.hm3d.layout import current_layout


ROOT = Path(__file__).resolve().parents[1]
LAYOUT = current_layout()
BASE = LAYOUT.scene_dir
GRAPH_PATH = LAYOUT.graph
MEMBERSHIP_PATH = LAYOUT.membership
REPORT_PATH = LAYOUT.room_report
SELECTION_PATH = LAYOUT.stair_selection
FLOORS_PATH = LAYOUT.floors


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--extra-clearance", type=float, default=0.10)
    args = parser.parse_args()
    if args.grid_dir is None:
        args.grid_dir = LAYOUT.grid_dir

    if not np.isfinite(args.extra_clearance) or args.extra_clearance < 0:
        raise ValueError("Extra clearance must be finite and nonnegative.")

    graph = read_json(GRAPH_PATH)
    membership = read_json(MEMBERSHIP_PATH)
    report = read_json(REPORT_PATH)

    if membership["source_graph_sha256"] != digest(GRAPH_PATH):
        raise RuntimeError("Graph changed after membership review.")
    if membership["source_component_report_sha256"] != digest(REPORT_PATH):
        raise RuntimeError("Component report changed after membership review.")

    for field, path in (
        ("selection_sha256", SELECTION_PATH),
        ("floor_plan_sha256", FLOORS_PATH),
        ("graph_sha256", GRAPH_PATH),
    ):
        if report[field] != digest(path):
            raise RuntimeError(f"Source changed: {path.name}")

    objects = {obj["object_id"]: obj for obj in graph["objects"]}
    workspaces = {
        ws["workspace_id"]: ws for ws in graph["workspaces"]
    }
    if (
        len(objects) != len(graph["objects"])
        or len(workspaces) != len(graph["workspaces"])
    ):
        raise ValueError("Duplicate object or workspace IDs.")

    result = {
        "status": "review_candidates_not_final",
        "coordinate_frame": "athome_z_up",
        "pose_dimensions": "x_y_yaw",
        "yaw_unit": "radian",
        "source_graph_sha256": digest(GRAPH_PATH),
        "membership_sha256": digest(MEMBERSHIP_PATH),
        "component_report_sha256": digest(REPORT_PATH),
        "visibility_checked": False,
        "physical_robot_feasibility_verified": False,
        "astar_cost_computed": False,
        "components": [],
    }

    for component in membership["components"]:
        name = component["component"]
        grid_path = args.grid_dir / f"{name}.grid.npz"
        meta_path = args.grid_dir / f"{name}.metadata.json"
        metadata = read_json(meta_path)

        if metadata["source_report_sha256"] != digest(REPORT_PATH):
            raise RuntimeError(f"{name}: grid and component report differ.")
        if metadata["coordinate_frame"] != "athome_z_up":
            raise RuntimeError(f"{name}: unexpected coordinate frame.")

        settings = metadata["source_navmesh_settings"]
        if settings is None:
            raise RuntimeError(f"{name}: source agent radius is unknown.")

        radius = float(settings["agent_radius"])
        offset = radius + args.extra_clearance
        goal_clearance, goal_max_offset = 0.0, None
        if LAYOUT.robot is not None:
            # Proposal 5-2: d_off = r_robot + d_safe and the in-place rotation
            # clearance, the values the real robot uses (athome.config).
            robot = yaml.safe_load(LAYOUT.robot.read_text(encoding="utf-8"))
            offset, goal_clearance, goal_max_offset = goal_geometry(robot)
            if not np.isclose(radius, robot_geometry(robot["robot"])[1], atol=1e-6):
                raise RuntimeError(f"{name}: NavMesh 반경이 로봇 설정과 다릅니다.")

        with np.load(grid_path, allow_pickle=False) as data:
            free = data["free"].astype(bool)
            labels = data["component_labels"].copy()
            origin = data["origin_xy_m"].astype(float)
            resolution = float(data["resolution_m"].item())

        if free.ndim != 2 or labels.shape != free.shape:
            raise ValueError(f"{name}: invalid grid shape.")
        if not np.isclose(resolution, metadata["resolution_m"]):
            raise ValueError(f"{name}: resolution mismatch.")
        if not np.allclose(origin, metadata["origin_xy_m"]):
            raise ValueError(f"{name}: origin mismatch.")
        if not free.any() or np.any(labels[free] <= 0):
            raise ValueError(f"{name}: invalid traversable cell labels.")

        rows, cols = np.nonzero(free)
        grid_clearance = GridMap(free, tuple(origin), resolution).clearance()
        x = origin[0] + (cols + 0.5) * resolution
        y = origin[1] + (rows + 0.5) * resolution

        component_result = {
            "component": name,
            "grid_sha256": digest(grid_path),
            "metadata_sha256": digest(meta_path),
            "agent_radius_m": radius,
            "extra_clearance_m": args.extra_clearance if LAYOUT.robot is None else offset - radius,
            "footprint_offset_m": offset,
            **({"goal_clearance_m": goal_clearance, "goal_max_offset_m": goal_max_offset}
               if LAYOUT.robot is not None else {}),
            "candidate_band_width_m": resolution,
            "workspaces": [],
        }

        print(f"\n=== Component {name} ===")
        print(f"Footprint offset: {offset:.3f} m")
        print(f"Candidate band width: {resolution:.3f} m")

        # Final workspace set of the component graph (after membership decisions).
        component_graph = read_json(
            LAYOUT.scene_dir / "component_graphs.review" / f"{name}.workspace_graph.review.json")
        for wid in [ws["workspace_id"] for ws in component_graph["workspaces"]]:
            if wid not in workspaces:
                raise ValueError(f"Unknown workspace: {wid}")

            workspace = workspaces[wid]
            source_id = workspace["source_object_id"]
            if source_id not in {obj["object_id"] for obj in component_graph["objects"]}:
                raise ValueError(f"Source membership mismatch: {source_id}")

            source = objects[source_id]
            lower = np.asarray(source["bbox"]["min"], dtype=float)
            upper = np.asarray(source["bbox"]["max"], dtype=float)

            if (
                lower.shape != (3,)
                or upper.shape != (3,)
                or not np.isfinite([lower, upper]).all()
                or np.any(upper <= lower)
            ):
                raise ValueError(f"Invalid source bounding box: {source_id}")

            # Euclidean distance from each cell center to the XY rectangle.
            dx = np.maximum(np.maximum(lower[0] - x, x - upper[0]), 0)
            dy = np.maximum(np.maximum(lower[1] - y, y - upper[1]), 0)
            distance = np.hypot(dx, dy)

            inside = (
                (x >= lower[0]) & (x <= upper[0])
                & (y >= lower[1]) & (y <= upper[1])
            )

            # Keep a one-cell-wide band outside the configured offset.
            eligible = (
                ~inside
                & (distance >= offset)
                & (distance < offset + resolution)
            )
            if goal_clearance > 0:  # room to turn in place (goal_poses.goal_candidates)
                eligible &= grid_clearance[rows, cols] >= goal_clearance
            if goal_max_offset is not None:
                # Nearest usable ring up to goal_max_offset (same rule as
                # athome.navigation.goal_poses.goal_candidates).
                usable = ~inside & (distance >= offset) & (distance < goal_max_offset)
                if goal_clearance > 0:
                    usable &= grid_clearance[rows, cols] >= goal_clearance
                ring = np.floor((distance - offset) / resolution)
                eligible = usable & (ring == ring[usable].min()) if usable.any() else usable
            indices = np.flatnonzero(eligible)
            center = (lower[:2] + upper[:2]) / 2
            candidates = []

            for index in indices:
                row, col = int(rows[index]), int(cols[index])
                px, py = float(x[index]), float(y[index])
                candidates.append({
                    "candidate_id": f"{name}:{wid}:r{row}:c{col}",
                    "row": row,
                    "col": col,
                    "x_m": px,
                    "y_m": py,
                    "yaw_rad": float(np.arctan2(
                        center[1] - py, center[0] - px
                    )),
                    "footprint_distance_m": float(distance[index]),
                    "grid_component_id": int(labels[row, col]),
                })

            count = len(candidates)
            component_result["workspaces"].append({
                "workspace_id": wid,
                "source_object_id": source_id,
                "room_id": workspace["room_id"],
                "status": (
                    "candidates_generated"
                    if count else "no_candidate_at_configured_offset"
                ),
                "candidates": candidates,
            })
            print(f"{source_id}: {count} candidates")

        result["components"].append(component_result)

    all_records = [
        ws for component in result["components"]
        for ws in component["workspaces"]
    ]
    missing = [
        ws["source_object_id"]
        for ws in all_records if not ws["candidates"]
    ]

    clearance_tag = format(args.extra_clearance, ".6g").replace(".", "p")
    output_path = (
        args.grid_dir
        / f"workspace_goal_candidates.clearance_{clearance_tag}.review.json"
    )
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(output_path)

    print("\n=== Summary ===")
    print("Workspaces:", len(all_records))
    print("With candidates:", len(all_records) - len(missing))
    print("Without candidates:", len(missing))
    print("Sources without candidates:", missing)
    print("Saved:", output_path)
    print("Graphs, grids, and selection files were not modified.")


if __name__ == "__main__":
    main()
