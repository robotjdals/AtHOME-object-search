import itertools
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/hm3d/wcojb4TFT35"
MESH = (
    ROOT / "data/scene_datasets/hm3d/minival"
    / "00802-wcojb4TFT35/wcojb4TFT35.semantic.glb"
)
GRAPH = ROOT / "outputs/hm3d/wcojb4TFT35.workspace_graph.v2.json"

if not MESH.is_file():
    raise SystemExit(f"원본 Semantic Mesh가 없습니다: {MESH}")

scene = trimesh.load_scene(str(MESH), process=False)

# GLB 내부 노드 변환을 적용한 장면 좌표를 사용합니다.
parts = []
for node in scene.graph.nodes_geometry:
    transform, geometry_name = scene.graph[node]
    geometry = scene.geometry[geometry_name]
    if isinstance(geometry, trimesh.Trimesh) and len(geometry.vertices):
        parts.append(trimesh.transform_points(
            geometry.vertices, transform
        ))

if not parts:
    raise SystemExit("표시할 Mesh 정점이 없습니다.")

native = np.concatenate(parts, axis=0)
if not np.isfinite(native).all():
    raise SystemExit("Mesh에 유효하지 않은 좌표가 있습니다.")

points = native.copy()
print("=== 원본 Mesh 좌표 범위 ===")
print("최소:", points.min(axis=0).round(3).tolist())
print("최대:", points.max(axis=0).round(3).tolist())

# 화면 표시만 경량화합니다. 원본 파일과 주석은 변경하지 않습니다.
rng = np.random.default_rng(0)
if len(points) > 100000:
    points = points[rng.choice(len(points), 100000, replace=False)]

fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=points[:, 0], y=points[:, 1], z=points[:, 2],
    mode="markers",
    name="원본 형상 정점",
    marker={
        "size": 1,
        "color": points[:, 2],
        "colorscale": "Viridis",
        "opacity": 0.25,
        "colorbar": {"title": "높이(m)"},
    },
    hoverinfo="skip",
))

graph = json.loads(GRAPH.read_text(encoding="utf-8"))
bits = list(itertools.product((0, 1), repeat=3))
edges = [
    (i, j)
    for i in range(8) for j in range(i + 1, 8)
    if sum(a != b for a, b in zip(bits[i], bits[j])) == 1
]

for obj in graph["objects"]:
    tag = obj["semantic_tag"].strip().casefold()
    if tag not in {"floor", "stairs", "stair"}:
        continue

    bbox = obj["bbox"]
    lo, hi = np.array(bbox["min"]), np.array(bbox["max"])
    corners = np.array([
        np.where(np.array(bit, dtype=bool), hi, lo)
        for bit in bits
    ])

    xyz = [[], [], []]
    for a, b in edges:
        for axis in range(3):
            xyz[axis].extend([
                float(corners[a, axis]), float(corners[b, axis]), None
            ])

    is_floor = tag == "floor"
    name = f"{obj['room_id']} / {obj['object_id']}"
    color = "crimson" if is_floor else "darkorange"

    fig.add_trace(go.Scatter3d(
        x=xyz[0], y=xyz[1], z=xyz[2],
        mode="lines",
        name=name,
        legendgroup=obj["room_id"],
        line={"color": color, "width": 4 if is_floor else 7},
        hovertemplate=name + "<extra></extra>",
    ))

    center = bbox["center"]
    fig.add_trace(go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]],
        mode="markers+text",
        text=[name],
        textposition="top center",
        marker={"size": 3, "color": color},
        legendgroup=obj["room_id"],
        showlegend=False,
        hovertemplate=name + "<extra></extra>",
    ))

views = [
    ("입체", {"x": 1.6, "y": 1.6, "z": 1.0}, {"x": 0, "y": 0, "z": 1}),
    ("위에서", {"x": 0, "y": 0, "z": 2.5}, {"x": 0, "y": 1, "z": 0}),
    ("X방향 측면", {"x": 2.5, "y": 0, "z": 0}, {"x": 0, "y": 0, "z": 1}),
    ("Y방향 측면", {"x": 0, "y": 2.5, "z": 0}, {"x": 0, "y": 0, "z": 1}),
]
buttons = [
    {
        "label": label,
        "method": "relayout",
        "args": [{
            "scene.camera": {
                "eye": eye,
                "up": up,
                "projection": {"type": "orthographic"},
            }
        }],
    }
    for label, eye, up in views
]

fig.update_layout(
    title="원본 장면 검토 — 층 분리 미적용",
    scene={
        "aspectmode": "data",
        "xaxis_title": "AtHOME X(m)",
        "yaxis_title": "AtHOME Y(m)",
        "zaxis_title": "높이 Z(m)",
    },
    updatemenus=[{
        "type": "buttons", "direction": "right",
        "buttons": buttons, "x": 0, "y": 1.08,
    }],
    height=900,
    margin={"l": 0, "r": 0, "b": 0, "t": 110},
)

output = BASE / "scene_structure.review.html"
fig.write_html(str(output), include_plotlyjs=True)
print("검토 화면 저장:", output)
print("표시한 정점 수:", len(points))
print("원본 데이터와 층 배정 상태는 변경하지 않았습니다.")
