import runpy
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh


ROOT = Path(__file__).resolve().parents[1]

# Reuse the existing review figure and validation checks.
state = runpy.run_path(
    str(ROOT / "scripts" / "review_unassigned_objects.py"),
    run_name="context_review",
)
fig = state["fig"]
objects = state["objects"]
pending = state["pending"]
reference_count = state["reference_count"]

mesh_path = (
    ROOT / "data" / "scene_datasets" / "hm3d" / "minival"
    / "00802-wcojb4TFT35" / "wcojb4TFT35.semantic.glb"
)

scene = trimesh.load(str(mesh_path), force="scene", process=False)

# Apply GLB node transforms once.
# For this inspected scene, the transformed coordinates already match AtHOME.
parts = []
for node in scene.graph.nodes_geometry:
    transform, geometry_name = scene.graph[node]
    geometry = scene.geometry[geometry_name]
    vertices = trimesh.transform_points(geometry.vertices, transform)
    parts.append(np.asarray(vertices, dtype=float))

if not parts:
    raise RuntimeError("No mesh vertices found.")

vertices = np.concatenate(parts, axis=0)
if not np.isfinite(vertices).all():
    raise RuntimeError("Mesh contains non-finite coordinates.")

# Check the previously verified scene bounds before overlaying geometry.
expected_min = np.array([-3.662, -1.730, -2.605])
expected_max = np.array([7.075, 19.042, 5.133])

if not (
    np.allclose(vertices.min(axis=0), expected_min, atol=0.03, rtol=0)
    and np.allclose(vertices.max(axis=0), expected_max, atol=0.03, rtol=0)
):
    raise RuntimeError(
        "Scene bounds differ from the verified coordinate frame. "
        "Stop and inspect the mesh transform."
    )

# Restrict display to the unresolved room and its immediate surroundings.
lower = np.min(
    [objects[item["object_id"]]["bbox"]["min"] for item in pending],
    axis=0,
) - 0.5
upper = np.max(
    [objects[item["object_id"]]["bbox"]["max"] for item in pending],
    axis=0,
) + 0.5

inside = np.all((vertices >= lower) & (vertices <= upper), axis=1)
context = vertices[inside]

if not len(context):
    raise RuntimeError("No context vertices inside the review bounds.")

# Sampling affects visualization only.
limit = 150000
if len(context) > limit:
    rng = np.random.default_rng(0)
    context = context[rng.choice(len(context), limit, replace=False)]

fig.add_trace(go.Scatter3d(
    x=context[:, 0],
    y=context[:, 1],
    z=context[:, 2],
    mode="markers",
    marker={"size": 1, "color": "#777777", "opacity": 0.20},
    name="Semantic mesh context",
    hoverinfo="skip",
))

# Keep the context visible when switching objects.
for index, button in enumerate(fig.layout.updatemenus[0].buttons):
    visibility = (
        [True] * reference_count
        + [number == index for number in range(len(pending))]
        + [True]
    )
    button.args = (
        {"visible": visibility},
        {"title": {"text": state["title_for"](pending[index])}},
    )

# Start with the object currently under review.
selected = next(
    index for index, item in enumerate(pending)
    if item["object_id"] == "lamp_314"
)
for index in range(len(pending)):
    fig.data[reference_count + index].visible = (index == selected)

fig.layout.updatemenus[0].active = selected
fig.update_layout(
    title={"text": state["title_for"](pending[selected])},
)

output = state["BASE"] / "unassigned_objects.context.review.html"
fig.write_html(output, include_plotlyjs=True, full_html=True)

print("Context vertices displayed:", len(context))
print("Saved:", output)
print("Object assignments and source geometry were not modified.")
