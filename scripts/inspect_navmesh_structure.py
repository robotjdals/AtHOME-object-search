import itertools
import json
from pathlib import Path

import habitat_sim
import numpy as np
import plotly.graph_objects as go
import trimesh


ROOT = Path(__file__).resolve().parents[1]
SCENE_ID = "wcojb4TFT35"
DATA = ROOT / "data/scene_datasets/hm3d/minival" / f"00802-{SCENE_ID}"
OUTPUT = ROOT / "outputs/hm3d" / SCENE_ID
GRAPH = ROOT / "outputs/hm3d" / f"{SCENE_ID}.workspace_graph.v2.json"

COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#8C6D31",
]


def to_athome(points):
    """Habitat 좌표를 AtHOME 좌표로 변환합니다."""
    points = np.asarray(points, dtype=float)
    return np.column_stack((
        points[:, 0], -points[:, 2], points[:, 1]
    ))


def load_scene_points():
    scene = trimesh.load_scene(
        str(DATA / f"{SCENE_ID}.semantic.glb"),
        process=False,
    )
    parts = []

    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices):
            parts.append(
                trimesh.transform_points(mesh.vertices, transform)
            )

    if not parts:
        raise ValueError("메시 정점을 찾지 못했습니다.")

    points = np.concatenate(parts)
    if not np.isfinite(points).all():
        raise ValueError("메시에 유효하지 않은 좌표가 있습니다.")

    # 현재 장면에서 확인한 좌표 처리:
    # GLB 노드 변환 이후에는 추가 축 변환을 적용하지 않습니다.
    rng = np.random.default_rng(0)
    if len(points) > 80_000:
        points = points[
            rng.choice(len(points), 80_000, replace=False)
        ]
    return points


def add_structure_boxes(fig, objects, categories, label, color):
    xs, ys, zs = [], [], []
    centers, labels = [], []
    bits = list(itertools.product((0, 1), repeat=3))

    for obj in objects:
        category = str(obj.get("semantic_tag", "")).strip().lower()
        if category not in categories or not obj.get("bbox"):
            continue

        bbox = obj["bbox"]
        lower = np.asarray(bbox["min"], dtype=float)
        upper = np.asarray(bbox["max"], dtype=float)
        if (
            lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite([lower, upper]).all()
            or np.any(upper < lower)
        ):
            raise ValueError(f"잘못된 Bounding Box: {obj['object_id']}")

        corners = np.array([
            np.where(np.asarray(bit, dtype=bool), upper, lower)
            for bit in bits
        ])

        for i, first in enumerate(bits):
            for j in range(i + 1, len(bits)):
                if sum(a != b for a, b in zip(first, bits[j])) != 1:
                    continue
                xs.extend([corners[i, 0], corners[j, 0], None])
                ys.extend([corners[i, 1], corners[j, 1], None])
                zs.extend([corners[i, 2], corners[j, 2], None])

        centers.append((lower + upper) / 2)
        labels.append(
            f"{obj.get('room_id', '미배정')} / {obj['object_id']}"
        )

    if not centers:
        return

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=3),
        name=label,
        legendgroup=label,
        hoverinfo="skip",
    ))

    centers = np.asarray(centers)
    fig.add_trace(go.Scatter3d(
        x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
        mode="markers",
        marker=dict(size=3, color=color),
        text=labels,
        hovertemplate="%{text}<br>높이 %{z:.3f} m<extra></extra>",
        legendgroup=label,
        showlegend=False,
    ))


def camera(x, y, z, up):
    return dict(
        eye=dict(x=x, y=y, z=z),
        up=up,
        projection=dict(type="orthographic"),
    )


def main():
    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(DATA / f"{SCENE_ID}.basis.navmesh")):
        raise RuntimeError("NavMesh 로딩 실패")

    points = load_scene_points()
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=1, color="#777777", opacity=0.12),
        name="원본 메시",
        hoverinfo="skip",
    ))

    records = []
    print("=== NavMesh 연결 영역별 확인 ===")

    for island in range(pf.num_islands):
        # 정점과 인덱스에 동일한 연결 영역 번호를 지정합니다.
        vertices = to_athome(pf.build_navmesh_vertices(island))
        faces = np.asarray(
            pf.build_navmesh_vertex_indices(island),
            dtype=np.int64,
        ).reshape(-1, 3)

        if len(vertices) == 0 or len(faces) == 0:
            raise ValueError(f"연결 영역 {island}: 비어 있는 메시")
        if not np.isfinite(vertices).all():
            raise ValueError(f"연결 영역 {island}: 잘못된 좌표")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError(f"연결 영역 {island}: 잘못된 삼각형 참조")

        area = float(pf.island_area(island))
        z_min = float(vertices[:, 2].min())
        z_max = float(vertices[:, 2].max())

        records.append({
            "island_id": island,
            "area_m2": area,
            "vertex_count": len(vertices),
            "triangle_count": len(faces),
            "min_xyz_m": vertices.min(axis=0).tolist(),
            "max_xyz_m": vertices.max(axis=0).tolist(),
        })

        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=COLORS[island % len(COLORS)],
            opacity=0.8,
            flatshading=True,
            name=f"연결 영역 {island} ({area:.2f} m²)",
            showlegend=True,
            hovertemplate=(
                f"연결 영역 {island}<br>"
                "X: %{x:.3f} m<br>"
                "Y: %{y:.3f} m<br>"
                "높이: %{z:.3f} m<extra></extra>"
            ),
        ))

        print(
            f"영역 {island}: 면적 {area:.3f} m², "
            f"높이 {z_min:.3f} ~ {z_max:.3f} m, "
            f"삼각형 {len(faces)}개"
        )

    add_structure_boxes(
        fig, graph["objects"], {"floor"}, "바닥 경계", "crimson"
    )
    add_structure_boxes(
        fig, graph["objects"], {"stair", "stairs"},
        "계단 경계", "darkorange"
    )

    views = [
        ("입체 보기", camera(1.5, 1.5, 1.1, dict(x=0, y=0, z=1))),
        ("위에서 보기", camera(0, 0, 2.5, dict(x=0, y=1, z=0))),
        ("X축에서 보기", camera(2.5, 0, 0, dict(x=0, y=0, z=1))),
        ("Y축에서 보기", camera(0, 2.5, 0, dict(x=0, y=0, z=1))),
    ]

    fig.update_layout(
        title=(
            f"{SCENE_ID} — 원본 메시·NavMesh 연결 영역 검토"
            "<br><sup>색상은 연결 영역을 나타내며 층 번호가 아닙니다."
            " 범례를 클릭하면 표시를 전환할 수 있습니다.</sup>"
        ),
        scene=dict(
            xaxis_title="AtHOME X (m)",
            yaxis_title="AtHOME Y (m)",
            zaxis_title="높이 Z (m)",
            aspectmode="data",
            camera=views[0][1],
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0, y=1.08,
            buttons=[
                dict(
                    label=name,
                    method="relayout",
                    args=[{"scene.camera": view}],
                )
                for name, view in views
            ],
        )],
        legend=dict(groupclick="togglegroup"),
        margin=dict(l=0, r=20, t=140, b=0),
        height=900,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    html_path = OUTPUT / "navmesh_structure.review.html"
    report_path = OUTPUT / "navmesh_structure.report.json"

    fig.write_html(
        str(html_path),
        include_plotlyjs=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
    report = {
        "scene_id": SCENE_ID,
        "coordinate_frame": "athome_z_up",
        "status": "inspection_only",
        "floor_assignment_verified": False,
        "navigable_area_m2": float(pf.navigable_area),
        "islands": records,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n연결 영역 수:", pf.num_islands)
    print("검토 화면:", html_path)
    print("검토 보고서:", report_path)
    print("원본 데이터와 층 배정은 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
