import hashlib
import heapq
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import distance_transform_edt
from athome.data.hm3d.layout import current_layout


ROOT = Path(__file__).resolve().parents[1]
LAYOUT = current_layout()
BASE = LAYOUT.scene_dir
GRID_DIR = LAYOUT.grid_dir
INPUT = GRID_DIR / "workspace_goal_candidates.clearance_0p1.review.json"
GRAPH = LAYOUT.graph
OUTPUT = GRID_DIR / "path_review"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def astar(free, start, goals, resolution):
    goals = set(goals)
    goal_array = np.asarray(sorted(goals), dtype=int)
    height, width = free.shape

    def heuristic(cell):
        delta = np.abs(goal_array - np.asarray(cell))
        lo = delta.min(axis=1)
        hi = delta.max(axis=1)
        return float(np.min(hi + (math.sqrt(2) - 1) * lo)) * resolution

    distance = {start: 0.0}
    parent = {}
    queue = [(heuristic(start), 0.0, start)]

    while queue:
        _, cost, cell = heapq.heappop(queue)
        if cost > distance[cell]:
            continue

        if cell in goals:
            path = [cell]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return cost, list(reversed(path))

        row, col = cell
        for dr, dc in (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if not free[nr, nc]:
                continue
            if dr and dc:
                if not free[row + dr, col] or not free[row, col + dc]:
                    continue

            step = resolution * (math.sqrt(2) if dr and dc else 1.0)
            new_cost = cost + step
            neighbor = (nr, nc)
            if new_cost < distance.get(neighbor, math.inf):
                distance[neighbor] = new_cost
                parent[neighbor] = cell
                heapq.heappush(
                    queue,
                    (new_cost + heuristic(neighbor), new_cost, neighbor),
                )

    return None, []


def main():
    inputs = json.loads(INPUT.read_text(encoding="utf-8"))
    if inputs["source_graph_sha256"] != digest(GRAPH):
        raise RuntimeError("Graph changed after candidate generation.")

    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    objects = {obj["object_id"]: obj for obj in graph["objects"]}
    OUTPUT.mkdir(parents=True, exist_ok=True)

    report = {
        "status": "development_path_check",
        "candidate_file_sha256": digest(INPUT),
        "start_policy": "maximum_padded_grid_distance_transform",
        "movement": "8_connected_no_corner_cutting",
        "visibility_checked": False,
        "components": [],
    }

    for component in inputs["components"]:
        name = component["component"]
        grid_path = GRID_DIR / f"{name}.grid.npz"
        meta_path = GRID_DIR / f"{name}.metadata.json"
        if (
            digest(grid_path) != component["grid_sha256"]
            or digest(meta_path) != component["metadata_sha256"]
        ):
            raise RuntimeError(f"{name}: grid or metadata changed.")

        with np.load(grid_path, allow_pickle=False) as data:
            free = data["free"].astype(bool)
            origin = data["origin_xy_m"].astype(float)
            resolution = float(data["resolution_m"].item())

        if not free.any():
            raise ValueError(f"{name}: no free cells.")

        clearance = distance_transform_edt(
            np.pad(free, 1, constant_values=False)
        )[1:-1, 1:-1]
        start = tuple(
            int(i) for i in np.unravel_index(np.argmax(clearance), free.shape)
        )

        def xy(cell):
            row, col = cell
            return (
                float(origin[0] + (col + 0.5) * resolution),
                float(origin[1] + (row + 0.5) * resolution),
            )

        height, width = free.shape
        extent = [
            origin[0], origin[0] + width * resolution,
            origin[1], origin[1] + height * resolution,
        ]

        def map_axes():
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(
                free, origin="lower", extent=extent,
                cmap="gray", vmin=0, vmax=1, interpolation="nearest",
            )
            ax.set_aspect("equal")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            return fig, ax

        fig, ax = map_axes()
        sx, sy = xy(start)
        ax.plot(sx, sy, "*", color="red", markersize=14, label="Start")
        records = []

        print(f"\n=== Component {name} ===")
        print("Start row/col:", start)
        print("Start XY:", [round(sx, 3), round(sy, 3)])

        free_rows, free_cols = np.nonzero(free)
        free_x = origin[0] + (free_cols + 0.5) * resolution
        free_y = origin[1] + (free_rows + 0.5) * resolution

        for number, workspace in enumerate(component["workspaces"], 1):
            source_id = workspace["source_object_id"]
            source = objects[source_id]
            candidates = workspace["candidates"]
            record = {
                "workspace_id": workspace["workspace_id"],
                "source_object_id": source_id,
            }

            if not candidates:
                lower = np.asarray(source["bbox"]["min"], dtype=float)
                upper = np.asarray(source["bbox"]["max"], dtype=float)
                dx = np.maximum(
                    np.maximum(lower[0] - free_x, free_x - upper[0]), 0
                )
                dy = np.maximum(
                    np.maximum(lower[1] - free_y, free_y - upper[1]), 0
                )
                distance = np.hypot(dx, dy)
                outside = (
                    (free_x < lower[0]) | (free_x > upper[0])
                    | (free_y < lower[1]) | (free_y > upper[1])
                )

                nearest = None
                if outside.any():
                    valid = np.flatnonzero(outside)
                    nearest = int(valid[np.argmin(distance[valid])])

                record["status"] = "no_candidate_at_configured_offset"
                record["nearest_external_free_cell_distance_m"] = (
                    float(distance[nearest]) if nearest is not None else None
                )

                detail_fig, detail_ax = map_axes()
                detail_ax.add_patch(Rectangle(
                    lower[:2], upper[0] - lower[0], upper[1] - lower[1],
                    fill=False, edgecolor="red", linewidth=2,
                    label="Source footprint",
                ))

                offset = component["footprint_offset_m"]
                band = component["candidate_band_width_m"]

                # Show exact distance contours around the rectangle.
                gx = np.linspace(lower[0] - 1, upper[0] + 1, 200)
                gy = np.linspace(lower[1] - 1, upper[1] + 1, 200)
                xx, yy = np.meshgrid(gx, gy)
                ddx = np.maximum(
                    np.maximum(lower[0] - xx, xx - upper[0]), 0
                )
                ddy = np.maximum(
                    np.maximum(lower[1] - yy, yy - upper[1]), 0
                )
                detail_ax.contour(
                    xx, yy, np.hypot(ddx, ddy),
                    levels=[offset, offset + band],
                    colors=["orange", "cyan"],
                )
                if nearest is not None:
                    detail_ax.plot(
                        free_x[nearest], free_y[nearest], "o",
                        color="magenta", label="Nearest external free cell",
                    )
                detail_ax.set_xlim(lower[0] - 1, upper[0] + 1)
                detail_ax.set_ylim(lower[1] - 1, upper[1] + 1)
                detail_ax.set_title(f"{name}: {source_id}\nNo candidate in band")
                detail_ax.legend()
                detail_fig.tight_layout()
                detail_path = OUTPUT / f"{name}_missing_{number:02d}.png"
                detail_fig.savefig(detail_path, dpi=160)
                plt.close(detail_fig)
                record["diagnostic_image"] = detail_path.name

                print(
                    f"{source_id}: NO CANDIDATE; "
                    f"nearest external free-cell distance = "
                    f"{record['nearest_external_free_cell_distance_m']}"
                )

            else:
                goals = []
                by_cell = {}
                for candidate in candidates:
                    cell = (candidate["row"], candidate["col"])
                    row, col = cell
                    if not (
                        0 <= row < height and 0 <= col < width
                        and free[row, col]
                    ):
                        raise ValueError(f"Invalid candidate: {source_id}")
                    goals.append(cell)
                    by_cell[cell] = candidate

                cost, path = astar(free, start, goals, resolution)
                if not path:
                    record["status"] = "no_path_from_this_start"
                    print(f"{source_id}: NO PATH")
                else:
                    # Independently check the returned grid path.
                    measured = 0.0
                    for a, b in zip(path, path[1:]):
                        dr, dc = b[0] - a[0], b[1] - a[1]
                        if (
                            max(abs(dr), abs(dc)) != 1
                            or not free[b]
                            or (dr and dc and (
                                not free[a[0] + dr, a[1]]
                                or not free[a[0], a[1] + dc]
                            ))
                        ):
                            raise RuntimeError("Invalid A* path.")
                        measured += math.hypot(dr, dc) * resolution
                    if not math.isclose(measured, cost, abs_tol=1e-8):
                        raise RuntimeError("Path cost mismatch.")

                    record.update({
                        "status": "path_found",
                        "astar_cost_m": cost,
                        "selected_candidate": by_cell[path[-1]],
                        "path_row_col": [list(cell) for cell in path],
                    })
                    coordinates = np.asarray([xy(cell) for cell in path])
                    ax.plot(coordinates[:, 0], coordinates[:, 1], linewidth=1)
                    ax.text(*coordinates[-1], str(number), color="blue")
                    print(f"{source_id}: {cost:.3f} m")

            records.append(record)

        ax.set_title(f"{name}: workspace paths from one development start")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT / f"{name}.paths.png", dpi=160)
        plt.close(fig)

        report["components"].append({
            "component": name,
            "start_row_col": list(start),
            "start_xy_m": [sx, sy],
            "workspaces": records,
        })

    output_path = OUTPUT / "workspace_paths.review.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print("\nSaved:", output_path)
    print("Images:", OUTPUT)
    print("Source graphs, grids, and candidates were not modified.")


if __name__ == "__main__":
    main()
