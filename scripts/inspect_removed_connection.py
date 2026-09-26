import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import habitat_sim
import numpy as np
import plotly.graph_objects as go

from inspect_navmesh_structure import load_scene_points


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/hm3d/wcojb4TFT35"
CONFIG_PATH = BASE / "stair_triangle_selection.review.json"
OUTPUT_PATH = BASE / "removed_connection.review.html"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    nav_path = ROOT / config["navmesh_path"]

    if digest(nav_path.read_bytes()) != config["navmesh_sha256"]:
        raise RuntimeError("원본 NavMesh가 선택 기록과 다릅니다.")

    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(nav_path)):
        raise RuntimeError("NavMesh 로딩 실패")

    island = config["island_id"]
    native = np.asarray(pf.build_navmesh_vertices(island), dtype=float)
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
    if signature != config["geometry_sha256"]:
        raise RuntimeError("삼각형 순서 또는 좌표가 변경됐습니다.")

    keep_list = config["retained_triangle_ids"]
    remove_list = config["excluded_triangle_ids"]
    ids = keep_list + remove_list
    if (
        any(type(i) is not int for i in ids)
        or len(ids) != len(set(ids))
        or set(ids) != set(range(len(faces)))
    ):
        raise ValueError("선택 목록에 중복 또는 누락이 있습니다.")

    keep = set(keep_list)
    removed = set(remove_list)
    triangles = vertices[faces]
    centers = triangles.mean(axis=1)
    areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    ) / 2

    # 완전히 같은 좌표만 합칩니다.
    _, remap = np.unique(vertices, axis=0, return_inverse=True)
    canonical = remap[faces]
    owners = defaultdict(list)

    for i, face in enumerate(canonical):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(face[a]), int(face[b]))))
            owners[edge].append(i)

    adjacency = [set() for _ in faces]
    for group in owners.values():
        if len(group) > 2:
            raise RuntimeError("세 면 이상이 공유하는 변이 있습니다.")
        if len(group) == 2:
            a, b = group
            adjacency[a].add(b)
            adjacency[b].add(a)

    def components(active):
        remaining = set(active)
        result = []
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            stack = [seed]
            group = []
            while stack:
                node = stack.pop()
                group.append(node)
                neighbors = adjacency[node] & remaining
                remaining.difference_update(neighbors)
                stack.extend(sorted(neighbors))
            result.append(group)
        return sorted(
            result, key=lambda g: (-float(areas[g].sum()), min(g))
        )

    if len(components(range(len(faces)))) != 1:
        raise RuntimeError(
            "제거 전 연결성을 재현하지 못했습니다. "
            "현재 방식으로 연결 후보를 확정하지 않습니다."
        )

    groups = components(keep)
    if len(groups) < 2:
        print("유지 영역이 하나입니다. 분리된 두 영역이 없습니다.")
        return

    # 현재 결과에서 면적이 큰 두 영역을 비교 대상으로 사용합니다.
    first, second = set(groups[0]), set(groups[1])

    start_boundary = sorted({
        neighbor
        for node in first
        for neighbor in adjacency[node]
        if neighbor in removed
    })
    end_boundary = {
        neighbor
        for node in second
        for neighbor in adjacency[node]
        if neighbor in removed
    }

    # 제외된 삼각형만 통과하며 두 영역 사이 연결 후보를 찾습니다.
    # 삼각형 개수가 가장 적은 후보 하나이며 실제 이동 최단경로는 아닙니다.
    queue = deque(start_boundary)
    previous = {node: None for node in start_boundary}
    endpoint = None

    while queue:
        node = queue.popleft()
        if node in end_boundary:
            endpoint = node
            break
        for neighbor in sorted(adjacency[node] & removed):
            if neighbor not in previous:
                previous[neighbor] = node
                queue.append(neighbor)

    if endpoint is None:
        raise RuntimeError(
            "제외 삼각형만으로 두 영역을 잇는 후보를 찾지 못했습니다."
        )

    path = []
    node = endpoint
    while node is not None:
        path.append(node)
        node = previous[node]
    path.reverse()

    print("=== 비교하는 유지 영역 ===")
    for name, group in (("A", groups[0]), ("B", groups[1])):
        z = triangles[group, :, 2]
        print(
            f"영역 {name}: 삼각형 {len(group)}개, "
            f"면적 {areas[group].sum():.3f} m², "
            f"높이 {z.min():.3f} ~ {z.max():.3f} m"
        )

    print("\n=== 제외된 연결 후보 ===")
    print("영역 A에 접한 제외 삼각형:", start_boundary)
    print("영역 B에 접한 제외 삼각형:", sorted(end_boundary))
    print("두 영역을 잇는 후보 순서:", path)

    for i in path:
        z = triangles[i, :, 2]
        print(
            f"삼각형 {i}: "
            f"높이 {z.min():.3f} ~ {z.max():.3f} m, "
            f"면적 {areas[i]:.4f} m², "
            f"중심 {np.round(centers[i], 3).tolist()}"
        )

    fig = go.Figure()
    points = load_scene_points()
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=1, color="#777777", opacity=0.15),
        name="원본 형상",
        hoverinfo="skip",
    ))

    def add_mesh(triangle_ids, name, color, opacity):
        triangle_ids = sorted(triangle_ids)
        if not triangle_ids:
            return
        subset = faces[triangle_ids]
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=subset[:, 0], j=subset[:, 1], k=subset[:, 2],
            color=color, opacity=opacity,
            name=name, showlegend=True,
            hoverinfo="skip",
        ))

    add_mesh(first, "유지 영역 A", "#0072B2", 0.55)
    add_mesh(second, "유지 영역 B", "#009E73", 0.55)
    add_mesh(keep - first - second, "나머지 유지 조각", "#777777", 0.8)
    add_mesh(removed - set(path), "나머지 제외 대상", "#E69F00", 0.25)
    add_mesh(path, "검토할 연결 후보", "#CC0077", 0.9)

    c = centers[path]
    fig.add_trace(go.Scatter3d(
        x=c[:, 0], y=c[:, 1], z=c[:, 2],
        mode="markers+text",
        marker=dict(size=4, color="#CC0077"),
        text=[str(i) for i in path],
        textposition="top center",
        name="연결 후보 번호",
        hovertemplate="삼각형 %{text}<br>높이 %{z:.3f} m<extra></extra>",
    ))

    focus = triangles[path].reshape(-1, 3)
    low = focus.min(axis=0) - 1.0
    high = focus.max(axis=0) + 1.0

    fig.update_layout(
        title=(
            "분리된 두 영역 사이의 연결 후보"
            "<br><sup>진분홍색은 검토 대상이며 자동 복원 대상이 아닙니다.</sup>"
        ),
        scene=dict(
            xaxis=dict(title="X (m)", range=[low[0], high[0]]),
            yaxis=dict(title="Y (m)", range=[low[1], high[1]]),
            zaxis=dict(title="높이 Z (m)", range=[low[2], high[2]]),
            aspectmode="data",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                projection=dict(type="orthographic"),
            ),
        ),
        height=850,
        margin=dict(l=0, r=20, t=90, b=0),
    )
    fig.write_html(
        str(OUTPUT_PATH),
        include_plotlyjs=True,
        config={"scrollZoom": True},
    )
    print("\n검토 화면:", OUTPUT_PATH)
    print("원본 NavMesh와 선택 파일은 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
