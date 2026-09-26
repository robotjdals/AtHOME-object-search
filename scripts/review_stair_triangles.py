import hashlib
import json
from pathlib import Path

import habitat_sim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Button


ROOT = Path(__file__).resolve().parents[1]
SCENE_ID = "wcojb4TFT35"
BASE = ROOT / "outputs/hm3d" / SCENE_ID
NAV_PATH = (
    ROOT / "data/scene_datasets/hm3d/minival"
    / f"00802-{SCENE_ID}" / f"{SCENE_ID}.basis.navmesh"
)
CONFIG_PATH = BASE / "stair_triangle_selection.review.json"
GRAPH_PATH = (
    ROOT / "outputs/hm3d" / f"{SCENE_ID}.workspace_graph.v2.json"
)
ISLAND_ID = 0


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(NAV_PATH)):
        raise RuntimeError("NavMesh 로딩 실패")

    native = np.asarray(
        pf.build_navmesh_vertices(ISLAND_ID), dtype=float
    )
    vertices = np.column_stack((
        native[:, 0], -native[:, 2], native[:, 1]
    ))
    faces = np.asarray(
        pf.build_navmesh_vertex_indices(ISLAND_ID), dtype=np.int64
    ).reshape(-1, 3)

    if (
        not len(faces)
        or not np.isfinite(vertices).all()
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("NavMesh 좌표 또는 삼각형 참조 오류")

    triangles = vertices[faces]
    centers = triangles.mean(axis=1)
    selected = np.zeros(len(faces), dtype=bool)

    nav_hash = sha256(NAV_PATH.read_bytes())
    geometry_hash = sha256(
        vertices.astype("<f8").tobytes()
        + faces.astype("<i8").tobytes()
    )

    # 이전 선택은 원본 파일과 삼각형 순서가 같을 때만 복원합니다.
    if CONFIG_PATH.exists():
        previous = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if (
            previous.get("navmesh_sha256") != nav_hash
            or previous.get("geometry_sha256") != geometry_hash
            or previous.get("island_id") != ISLAND_ID
        ):
            raise RuntimeError(
                "저장된 설정과 현재 NavMesh가 다릅니다. "
                "기존 설정을 덮어쓰지 않습니다."
            )
        ids = previous["excluded_triangle_ids"]
        if any(
            not isinstance(i, int) or i < 0 or i >= len(faces)
            for i in ids
        ):
            raise ValueError("저장된 삼각형 번호가 유효하지 않습니다.")
        selected[ids] = True
        print("기존 검토 선택을 복원했습니다.")

    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    # 계단 객체 상자는 참고용입니다. 자동 제외에 사용하지 않습니다.
    stair_boxes = [
        obj["bbox"] for obj in graph["objects"]
        if str(obj.get("semantic_tag", "")).strip().lower()
        in {"stair", "stairs"}
        and obj.get("bbox")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.subplots_adjust(bottom=0.22, top=0.85, wspace=0.3)

    # 글꼴 의존성을 피하기 위해 화면 축에는 좌표 기호를 사용합니다.
    projections = [
        (0, 1, "XY", "X (m)", "Y (m)"),
        (1, 2, "YZ", "Y (m)", "Z (m)"),
        (0, 2, "XZ", "X (m)", "Z (m)"),
    ]

    collections = []
    scatter_to_axes = {}
    scatters = []
    highlights = []
    texts = []

    for ax, (a, b, title, xlabel, ylabel) in zip(axes, projections):
        collection = PolyCollection(
            triangles[:, :, [a, b]],
            linewidths=0.5,
            edgecolors="#555555",
        )
        ax.add_collection(collection)
        collections.append(collection)

        scatter = ax.scatter(
            centers[:, a], centers[:, b],
            s=16, picker=4, zorder=4,
        )
        scatters.append(scatter)
        scatter_to_axes[scatter] = (ax, a, b)

        highlight = PolyCollection(
            [], facecolors="none",
            edgecolors="#00A000", linewidths=2.5, zorder=5,
        )
        ax.add_collection(highlight)
        highlights.append(highlight)
        texts.append(ax.text(
            0.02, 0.98, "", transform=ax.transAxes,
            va="top", fontsize=10,
        ))

        for bbox in stair_boxes:
            low = np.asarray(bbox["min"])
            high = np.asarray(bbox["max"])
            ax.plot(
                [low[a], high[a], high[a], low[a], low[a]],
                [low[b], low[b], high[b], high[b], low[b]],
                color="#E69F00", linestyle="--", linewidth=1,
            )

        # 坐標範囲は領域0を基準に固定します。
        # 표시 범위는 영역 0의 정점으로 결정합니다.
        lo = vertices[:, [a, b]].min(axis=0)
        hi = vertices[:, [a, b]].max(axis=0)
        margin = np.maximum((hi - lo) * 0.06, 0.15)
        ax.set_xlim(lo[0] - margin[0], hi[0] + margin[0])
        ax.set_ylim(lo[1] - margin[1], hi[1] + margin[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    title = fig.suptitle("")
    history = []

    def redraw():
        facecolors = [
            "#D55E0055" if value else "#0072B233"
            for value in selected
        ]
        pointcolors = [
            "#D55E00" if value else "#0072B2"
            for value in selected
        ]
        for collection, scatter in zip(collections, scatters):
            collection.set_facecolors(facecolors)
            scatter.set_color(pointcolors)

        title.set_text(
            f"Island {ISLAND_ID} | "
            f"Keep: {int((~selected).sum())} | "
            f"Exclude: {int(selected.sum())}\n"
            "Blue: keep / Red: exclude / Green outline: last clicked"
        )
        fig.canvas.draw_idle()

    def on_pick(event):
        if event.artist not in scatter_to_axes:
            return
        toolbar = getattr(fig.canvas.manager, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return

        indices = np.asarray(event.ind, dtype=int)
        if len(indices) != 1:
            print(
                "여러 삼각형의 중심점이 겹칩니다. "
                "확대하거나 다른 방향의 화면에서 선택하세요."
            )
            return

        index = int(indices[0])
        history.append(index)
        selected[index] = not selected[index]

        for highlight, text, (a, b, *_rest) in zip(
            highlights, texts, projections
        ):
            highlight.set_verts([triangles[index][:, [a, b]]])
            text.set_text(f"Triangle {index}")

        low = triangles[index, :, 2].min()
        high = triangles[index, :, 2].max()
        state = "제외 대상" if selected[index] else "유지"
        print(
            f"삼각형 {index}: {state}, "
            f"높이 {low:.3f} ~ {high:.3f} m"
        )
        redraw()

    def undo(_event):
        if history:
            index = history.pop()
            selected[index] = not selected[index]
            redraw()

    def save(_event):
        if not selected.any():
            print("제외 대상으로 선택한 삼각형이 없습니다.")
            return
        if selected.all():
            print("전체 영역을 제외할 수 없습니다.")
            return

        record = {
            "schema_version": "0.1",
            "scene_id": SCENE_ID,
            "status": "manual_selection_pending_validation",
            "coordinate_frame": "athome_z_up",
            "navmesh_path": str(NAV_PATH.relative_to(ROOT)),
            "navmesh_sha256": nav_hash,
            "geometry_sha256": geometry_hash,
            "island_id": ISLAND_ID,
            "triangle_count": len(faces),
            "excluded_triangle_ids": np.flatnonzero(selected).tolist(),
            "retained_triangle_ids": np.flatnonzero(~selected).tolist(),
            "selection_method": "manual_triangle_review",
            "floor_assignment_verified": False,
        }

        BASE.mkdir(parents=True, exist_ok=True)
        temporary = CONFIG_PATH.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(CONFIG_PATH)

        print("\n=== 계단 제외 선택 저장 ===")
        print("유지 삼각형:", int((~selected).sum()))
        print("제외 삼각형:", int(selected.sum()))
        print("저장 위치:", CONFIG_PATH)
        print("원본 NavMesh는 변경하지 않았습니다.")
        print("선택 후 연결성과 위층 바닥 보존 검증이 필요합니다.")

    save_ax = fig.add_axes([0.65, 0.06, 0.14, 0.06])
    undo_ax = fig.add_axes([0.81, 0.06, 0.14, 0.06])
    save_button = Button(save_ax, "Save selection")
    undo_button = Button(undo_ax, "Undo")
    save_button.on_clicked(save)
    undo_button.on_clicked(undo)

    fig.canvas.mpl_connect("pick_event", on_pick)
    redraw()

    print("=== 계단 삼각형 검토 ===")
    print("파란 점 클릭: 제외 대상으로 선택")
    print("빨간 점 클릭: 유지로 복원")
    print("초록 테두리: 마지막으로 클릭한 삼각형")
    print("주황 점선: 계단 객체 상자이며 참고용입니다.")
    print("Save selection: 선택 저장 / Undo: 직전 선택 취소")
    print("확대·이동 도구를 해제한 뒤 점을 선택하세요.")
    print("창을 닫는 것만으로는 선택이 저장되지 않습니다.")
    plt.show()


if __name__ == "__main__":
    main()
