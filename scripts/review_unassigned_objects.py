import hashlib
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]
SCENE = "wcojb4TFT35"
BASE = ROOT / "outputs" / "hm3d" / SCENE
GRID_DIR = BASE / "component_grids" / "f440d0f045c4_5cm"

GRAPH_PATH = ROOT / "outputs" / "hm3d" / f"{SCENE}.workspace_graph.v2.json"
MEMBERSHIP_PATH = BASE / "component_membership.review.json"
REPORT_PATH = BASE / "component_room_candidates.review.json"
OUTPUT_PATH = BASE / "unassigned_objects.review.html"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def bbox_trace(obj, color, width=5, visible=True):
    lower = np.asarray(obj["bbox"]["min"], dtype=float)
    upper = np.asarray(obj["bbox"]["max"], dtype=float)

    require(
        lower.shape == (3,)
        and upper.shape == (3,)
        and np.isfinite([lower, upper]).all()
        and np.all(upper >= lower),
        f"Invalid bounding box: {obj['object_id']}",
    )

    corners = np.array([
        [lower[0], lower[1], lower[2]],
        [upper[0], lower[1], lower[2]],
        [upper[0], upper[1], lower[2]],
        [lower[0], upper[1], lower[2]],
        [lower[0], lower[1], upper[2]],
        [upper[0], lower[1], upper[2]],
        [upper[0], upper[1], upper[2]],
        [lower[0], upper[1], upper[2]],
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    coordinates = [[], [], []]
    for first, second in edges:
        for axis in range(3):
            coordinates[axis].extend([
                float(corners[first, axis]),
                float(corners[second, axis]),
                None,
            ])

    return go.Scatter3d(
        x=coordinates[0],
        y=coordinates[1],
        z=coordinates[2],
        mode="lines",
        line={"color": color, "width": width},
        name=obj["object_id"],
        visible=visible,
        hovertemplate=obj["object_id"] + "<extra></extra>",
    )


graph = read_json(GRAPH_PATH)
membership = read_json(MEMBERSHIP_PATH)
report = read_json(REPORT_PATH)

require(
    membership["source_graph_sha256"] == sha256(GRAPH_PATH),
    "Graph changed after membership review.",
)
require(
    membership["source_component_report_sha256"] == sha256(REPORT_PATH),
    "Component report changed after membership review.",
)

objects = {obj["object_id"]: obj for obj in graph["objects"]}
require(
    len(objects) == len(graph["objects"]),
    "Duplicate object IDs.",
)

pending = membership["unresolved_objects"]
require(bool(pending), "No unresolved objects.")

fig = go.Figure()
colors = {"A": "#168aad", "B": "#2a9d8f"}

for component in report["components"]:
    name = component["component"]
    mesh_path = GRID_DIR / f"{name}.mesh.npz"
    metadata = read_json(GRID_DIR / f"{name}.metadata.json")

    require(
        metadata["source_report_sha256"] == sha256(REPORT_PATH),
        f"Mesh metadata refers to another component report: {name}",
    )

    with np.load(mesh_path, allow_pickle=False) as data:
        vertices = data["vertices"].astype(float)
        faces = data["faces"].astype(int)
        triangle_ids = data["original_triangle_ids"].astype(int)

    require(
        sorted(triangle_ids.tolist()) == sorted(component["triangle_ids"]),
        f"Triangle selection mismatch: {name}",
    )

    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=colors[name],
        opacity=0.55,
        name=f"Component {name}",
        showlegend=True,
        hovertemplate=f"Component {name}<extra></extra>",
    ))

# The floor boxes are reference geometry, not room boundaries.
floor_ids = set()
for component in report["components"]:
    name = component["component"]
    for candidate in component["room_candidates"]:
        floor_id = candidate["floor_object_id"]
        if floor_id in floor_ids:
            continue
        require(floor_id in objects, f"Missing floor: {floor_id}")
        floor_ids.add(floor_id)
        trace = bbox_trace(objects[floor_id], colors[name], width=2)
        trace.showlegend = False
        fig.add_trace(trace)

reference_count = len(fig.data)

for index, entry in enumerate(pending):
    object_id = entry["object_id"]
    require(object_id in objects, f"Missing object: {object_id}")
    fig.add_trace(bbox_trace(
        objects[object_id],
        color="#e63946",
        width=7,
        visible=(index == 0),
    ))


def title_for(entry):
    obj = objects[entry["object_id"]]
    low = obj["bbox"]["min"][2]
    high = obj["bbox"]["max"][2]
    return (
        f"{obj['object_id']} | Room {obj['room_id']} | "
        f"Z: {low:.3f} to {high:.3f} m"
        "<br><sup>Red: selected object bbox | "
        "Blue: A | Green: B | Thin lines: floor bboxes</sup>"
    )


buttons = []
for index, entry in enumerate(pending):
    visibility = (
        [True] * reference_count
        + [number == index for number in range(len(pending))]
    )
    buttons.append({
        "label": entry["object_id"],
        "method": "update",
        "args": [
            {"visible": visibility},
            {"title": {"text": title_for(entry)}},
        ],
    })

fig.update_layout(
    title={"text": title_for(pending[0])},
    scene={
        "xaxis_title": "AtHOME X (m)",
        "yaxis_title": "AtHOME Y (m)",
        "zaxis_title": "Height Z (m)",
        "aspectmode": "data",
        "uirevision": "keep-camera",
    },
    updatemenus=[{
        "buttons": buttons,
        "direction": "down",
        "x": 0,
        "y": 1.10,
        "xanchor": "left",
        "yanchor": "top",
    }],
    margin={"l": 10, "r": 10, "t": 150, "b": 10},
    height=850,
    legend={"x": 1.0, "y": 1.0},
)

fig.write_html(
    OUTPUT_PATH,
    include_plotlyjs=True,
    full_html=True,
)

print("Unresolved objects:", len(pending))
print("Coordinate frame: athome_z_up")
print("Saved:", OUTPUT_PATH)
print("Assignments and source files were not modified.")
print("Bounding boxes are spatial evidence, not exact object surfaces.")
