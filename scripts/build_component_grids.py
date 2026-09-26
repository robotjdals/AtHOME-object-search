import hashlib
import json
from pathlib import Path

import habitat_sim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from scipy.ndimage import label
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/hm3d/wcojb4TFT35"
REPORT_PATH = BASE / "component_room_candidates.review.json"
SELECTION_PATH = BASE / "stair_triangle_selection.review.json"
GRAPH_PATH = ROOT / "outputs/hm3d/wcojb4TFT35.workspace_graph.v2.json"
FLOORS_PATH = BASE / "floor_environments.json"

RESOLUTION = 0.05


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    raw_report = REPORT_PATH.read_bytes()
    report = json.loads(raw_report)
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))

    for field, path in (
        ("selection_sha256", SELECTION_PATH),
        ("graph_sha256", GRAPH_PATH),
        ("floor_plan_sha256", FLOORS_PATH),
    ):
        if report[field] != digest(path.read_bytes()):
            raise RuntimeError(
                f"検査" if False else
                f"이전 검사 이후 파일이 변경됐습니다: {path.name}"
            )

    nav_path = ROOT / selection["navmesh_path"]
    if digest(nav_path.read_bytes()) != report["navmesh_sha256"]:
        raise RuntimeError("원본 NavMesh가 변경됐습니다.")

    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(nav_path)):
        raise RuntimeError("NavMesh 로딩 실패")

    island = selection["island_id"]
    native = np.asarray(
        pf.build_navmesh_vertices(island), dtype=float
    )
    vertices = np.column_stack((
        native[:, 0], -native[:, 2], native[:, 1]
    ))
    faces = np.asarray(
        pf.build_navmesh_vertex_indices(island), dtype=np.int64
    ).reshape(-1, 3)

    signature = digest(
        vertices.astype("<f8").tobytes()
        + faces.astype("<i8").tobytes()
    )
    if signature != report["geometry_sha256"]:
        raise RuntimeError("삼각형 순서 또는 좌표가 변경됐습니다.")

    extra = set(report["temporary_extra_excluded_triangle_ids"])
    expected = set(selection["retained_triangle_ids"]) - extra
    exported = [
        i for component in report["components"]
        for i in component["triangle_ids"]
    ]
    if (
        any(type(i) is not int for i in exported)
        or len(exported) != len(set(exported))
        or set(exported) != expected
        or not expected
        or min(expected) < 0
        or max(expected) >= len(faces)
    ):
        raise ValueError("이동 영역의 삼각형 목록이 선택 기록과 다릅니다.")

    settings = pf.nav_mesh_settings
    setting_names = [
        "agent_radius", "agent_height",
        "agent_max_climb", "agent_max_slope",
        "cell_size", "cell_height",
    ]
    source_settings = (
        {name: float(getattr(settings, name)) for name in setting_names}
        if settings is not None else None
    )

    version = digest(raw_report)[:12]
    output_dir = BASE / "component_grids" / f"{version}_5cm"
    output_dir.mkdir(parents=True, exist_ok=True)

    for component in report["components"]:
        name = component["component"]
        ids = np.asarray(component["triangle_ids"], dtype=np.int64)
        selected_faces = faces[ids]
        triangles = vertices[selected_faces]

        polygons = []
        for triangle in triangles:
            polygon = Polygon(triangle[:, :2])
            if not polygon.is_valid or polygon.area <= 0:
                raise ValueError(
                    f"영역 {name}: XY 투영이 유효하지 않은 삼각형"
                )
            polygons.append(polygon)

        footprint = unary_union(polygons)
        if footprint.is_empty or not footprint.is_valid:
            raise ValueError(f"영역 {name}: 투영 영역이 유효하지 않습니다.")

        # 서로 다른 높이의 면이 XY에서 중첩되면 단일 지도로
        # 투영하기 전에 별도 검토가 필요합니다.
        projected_sum = sum(p.area for p in polygons)
        overlap_area = projected_sum - footprint.area
        if overlap_area > max(1e-8, projected_sum * 1e-7):
            raise RuntimeError(
                f"영역 {name}: XY 중첩 면적 {overlap_area:.6f}m². "
                "단일 2D 지도로 저장하지 않습니다."
            )

        xmin, ymin, xmax, ymax = footprint.bounds
        origin = np.floor(
            np.array([xmin, ymin]) / RESOLUTION
        ) * RESOLUTION - RESOLUTION
        width, height = (
            np.ceil(
                (np.array([xmax, ymax]) - origin) / RESOLUTION
            ).astype(int) + 1
        )

        free = np.zeros((height, width), dtype=bool)
        prepared = prep(footprint)

        for row in range(height):
            y0 = origin[1] + row * RESOLUTION
            for col in range(width):
                x0 = origin[0] + col * RESOLUTION
                cell = box(
                    x0, y0,
                    x0 + RESOLUTION, y0 + RESOLUTION,
                )
                free[row, col] = prepared.covers(cell)

        if not free.any():
            raise RuntimeError(f"영역 {name}: 통행 가능한 셀이 없습니다.")

        # 상하좌우 연결을 검사합니다.
        structure = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ])
        grid_labels, count = label(free, structure=structure)
        sizes = np.bincount(grid_labels.ravel())[1:]
        component_areas = sorted(
            (sizes * RESOLUTION ** 2).tolist(),
            reverse=True,
        )

        used, inverse = np.unique(
            selected_faces.ravel(), return_inverse=True
        )
        local_vertices = vertices[used]
        local_faces = inverse.reshape(-1, 3)

        # free[row, col] が通行可能性を表します。
        np.savez_compressed(
            output_dir / f"{name}.grid.npz",
            free=free,
            component_labels=grid_labels,
            origin_xy_m=origin,
            resolution_m=np.array(RESOLUTION),
        )
        np.savez_compressed(
            output_dir / f"{name}.mesh.npz",
            vertices=local_vertices,
            faces=local_faces,
            original_triangle_ids=ids,
        )

        z = triangles[:, :, 2]
        metadata = {
            "status": "review_grid_not_final",
            "component": name,
            "coordinate_frame": "athome_z_up",
            "source_report_sha256": digest(raw_report),
            "navmesh_sha256": report["navmesh_sha256"],
            "geometry_sha256": signature,
            "source_navmesh_settings": source_settings,
            "resolution_m": RESOLUTION,
            "origin_xy_m": origin.tolist(),
            "array_shape_rows_cols": [int(height), int(width)],
            "cell_center_formula":
                "origin_xy + resolution * [col + 0.5, row + 0.5]",
            "free_value": True,
            "blocked_value": False,
            "blocked_meaning": "not certified traversable",
            "rasterization": "whole_cell_covered_by_projected_navmesh",
            "additional_inflation_m": 0.0,
            "navmesh_z_range_m": [float(z.min()), float(z.max())],
            "projected_navmesh_area_m2": float(footprint.area),
            "grid_free_area_m2": float(free.sum() * RESOLUTION ** 2),
            "grid_components_4_connected": int(count),
            "grid_component_areas_m2": component_areas,
            "extra_excluded_triangle_ids": sorted(extra),
            "room_assignment_verified": False,
            "robot_feasibility_verified": False,
            "shapely_version": shapely.__version__,
        }
        (output_dir / f"{name}.metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(
            free, origin="lower", cmap="gray", vmin=0, vmax=1,
            extent=[
                origin[0], origin[0] + width * RESOLUTION,
                origin[1], origin[1] + height * RESOLUTION,
            ],
            interpolation="nearest",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(
            f"{name} | 0.05 m/cell | "
            f"4-connected components: {count}\n"
            "White: traversable / Black: blocked"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.grid.png", dpi=180)
        plt.close(fig)

        print(f"\n=== 영역 {name} ===")
        print("격자 크기:", free.shape)
        print(f"원본 XY 투영 면적: {footprint.area:.3f} m²")
        print(f"통행 가능 셀 면적: {free.sum() * RESOLUTION**2:.3f} m²")
        print("격자 연결 영역 수:", count)
        print(
            "연결 영역별 면적:",
            [round(area, 4) for area in component_areas],
        )
        if count > 1:
            print("격자에서 분리가 발생했습니다. 자동 연결하지 않았습니다.")

    print("\n저장 폴더:", output_dir)
    print("기존 선택 파일과 원본 NavMesh는 변경하지 않았습니다.")
    print("이 지도는 원본 NavMesh 설정을 사용하는 검토본입니다.")


if __name__ == "__main__":
    main()
