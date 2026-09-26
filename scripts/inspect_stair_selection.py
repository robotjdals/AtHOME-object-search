import hashlib
import json
from pathlib import Path

import habitat_sim
import numpy as np
import plotly.graph_objects as go

from inspect_navmesh_structure import load_scene_points


ROOT = Path(__file__).resolve().parents[1]
SCENE_ID = "wcojb4TFT35"
BASE = ROOT / "outputs/hm3d" / SCENE_ID

CONFIG_PATH = BASE / "stair_triangle_selection.review.json"
NAV_PATH = (
    ROOT / "data/scene_datasets/hm3d/minival"
    / f"00802-{SCENE_ID}" / f"{SCENE_ID}.basis.navmesh"
)
OUTPUT_PATH = BASE / "stair_selection.overlay.html"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    if config["navmesh_sha256"] != digest(NAV_PATH.read_bytes()):
        raise RuntimeError("선택 기록과 원본 NavMesh가 다릅니다.")

    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(NAV_PATH)):
        raise RuntimeError("NavMesh 로딩 실패")

    island = config["island_id"]
    native = np.asarray(
        pf.build_navmesh_vertices(island), dtype=float
    )
    vertices = np.column_stack((
        native[:, 0], -native[:, 2], native[:, 1]
    ))
    faces = np.asarray(
        pf.build_navmesh_vertex_indices(island), dtype=np.int64
    ).reshape(-1, 3)

    geometry_hash = digest(
        vertices.astype("<f8").tobytes()
        + faces.astype("<i8").tobytes()
    )
    if geometry_hash != config["geometry_sha256"]:
        raise RuntimeError("선택 기록과 현재 삼각형 순서가 다릅니다.")

    keep = config["retained_triangle_ids"]
    remove = config["excluded_triangle_ids"]
    all_ids = keep + remove

    if (
        config["triangle_count"] != len(faces)
        or any(type(i) is not int for i in all_ids)
        or len(all_ids) != len(set(all_ids))
        or set(all_ids) != set(range(len(faces)))
    ):
        raise ValueError("유지·제외 삼각형 목록에 중복 또는 누락이 있습니다.")

    points = load_scene_points()
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=1.5, color="#666666", opacity=0.25),
        name="원본 건물 형상",
        hoverinfo="skip",
    ))

    for ids, label, color in [
        (keep, "유지", "#0072B2"),
        (remove, "제외 대상", "#D55E00"),
    ]:
        if not ids:
            continue

        subset = faces[np.asarray(ids, dtype=int)]
        triangles = vertices[subset]
        centers = triangles.mean(axis=1)

        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=subset[:, 0], j=subset[:, 1], k=subset[:, 2],
            color=color,
            opacity=0.55,
            flatshading=True,
            name=f"{label} 면",
            showlegend=True,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter3d(
            x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
            mode="markers",
            marker=dict(size=3, color=color),
            customdata=np.column_stack((
                ids,
                triangles[:, :, 2].min(axis=1),
                triangles[:, :, 2].max(axis=1),
            )),
            name=f"{label} 삼각형 번호",
            hovertemplate=(
                "삼각형 %{customdata[0]:.0f}<br>"
                f"상태: {label}<br>"
                "최소 높이: %{customdata[1]:.3f} m<br>"
                "최대 높이: %{customdata[2]:.3f} m"
                "<extra></extra>"
            ),
        ))

        z = triangles[:, :, 2]
        print(
            f"{label}: {len(ids)}개, "
            f"높이 {z.min():.3f} ~ {z.max():.3f} m"
        )

    # 제외 대상 주변으로 화면만 확대합니다.
    # 원본 정점이나 선택 목록은 자르거나 수정하지 않습니다.
    if remove:
        focus = vertices[faces[np.asarray(remove)]].reshape(-1, 3)
    else:
        focus = vertices

    low = focus.min(axis=0) - np.array([1.0, 1.0, 0.5])
    high = focus.max(axis=0) + np.array([1.0, 1.0, 0.5])

    full_low = points.min(axis=0) - 0.3
    full_high = points.max(axis=0) + 0.3

    def ranges(lo, hi):
        return {
            "scene.xaxis.range": [float(lo[0]), float(hi[0])],
            "scene.yaxis.range": [float(lo[1]), float(hi[1])],
            "scene.zaxis.range": [float(lo[2]), float(hi[2])],
        }

    fig.update_layout(
        title=(
            "계단 제외 선택과 원본 건물 형상 비교"
            "<br><sup>파랑: 유지 / 주황: 제외 대상 / 회색: 원본 메시"
            " — 선택은 아직 검토 상태입니다.</sup>"
        ),
        scene=dict(
            xaxis=dict(title="X (m)", range=[low[0], high[0]]),
            yaxis=dict(title="Y (m)", range=[low[1], high[1]]),
            zaxis=dict(title="높이 Z (m)", range=[low[2], high[2]]),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
                up=dict(x=0, y=0, z=1),
                projection=dict(type="orthographic"),
            ),
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0, y=1.08,
            buttons=[
                dict(
                    label="계단 주변",
                    method="relayout",
                    args=[ranges(low, high)],
                ),
                dict(
                    label="전체 건물",
                    method="relayout",
                    args=[ranges(full_low, full_high)],
                ),
            ],
        )],
        margin=dict(l=0, r=20, t=140, b=0),
        height=900,
    )

    fig.write_html(
        str(OUTPUT_PATH),
        include_plotlyjs=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
    print("검토 화면:", OUTPUT_PATH)
    print("선택 설정과 원본 NavMesh는 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
