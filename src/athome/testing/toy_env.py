"""Two-room toy environment for tests and demos.

    y
    6 +--------------------+--------------------+
      | fridge     counter |  sofa        lamp  |
      |                    |                    |
    3 |  kitchen (R_A)    door   living (R_B)    |
      |                    |                    |
      |    table           |  coffee table shelf|
    0 +--------------------+--------------------+ x
      0                    5                    10
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from athome.navigation import GridMap, traversable_from_occupancy
from athome.scene_graph.query import normalize_category

RESOLUTION = 0.05
SIZE = (10.0, 6.0)
DOOR_Y = (2.5, 3.5)

# object_id: (category, room, bbox min, bbox max, parent source or None,
#             is obstacle on the floor)
OBJECTS: Dict[str, tuple] = {
    "table_1": ("table", "R_A", (1.0, 1.0, 0.0), (2.0, 1.8, 0.75), None, True),
    "cup_2": ("cup", "R_A", (1.2, 1.2, 0.75), (1.3, 1.3, 0.85), "table_1", False),
    "plate_3": ("plate", "R_A", (1.6, 1.3, 0.75), (1.8, 1.5, 0.77), "table_1", False),
    "counter_4": ("counter", "R_A", (3.6, 4.8, 0.0), (4.8, 5.8, 0.9), None, True),
    "kettle_5": ("kettle", "R_A", (4.0, 5.2, 0.9), (4.2, 5.4, 1.1), "counter_4", False),
    "fridge_6": ("fridge", "R_A", (0.2, 4.8, 0.0), (1.0, 5.8, 1.8), None, True),
    "coffee table_7": ("coffee table", "R_B", (6.5, 2.0, 0.0), (7.5, 2.8, 0.45), None, True),
    "remote_8": ("remote", "R_B", (6.8, 2.3, 0.45), (6.9, 2.4, 0.48), "coffee table_7", False),
    "shelf_9": ("shelf", "R_B", (9.3, 0.5, 0.0), (9.8, 2.5, 1.8), None, True),
    "book_10": ("book", "R_B", (9.4, 1.0, 1.0), (9.6, 1.2, 1.25), "shelf_9", False),
    "lamp_11": ("lamp", "R_B", (8.5, 5.0, 0.0), (8.9, 5.4, 1.5), None, True),
    "sofa_12": ("sofa", "R_B", (6.2, 4.8, 0.0), (8.0, 5.8, 0.9), None, True),
    "wall_13": ("wall", "R_B", (5.0, 0.0, 0.0), (5.1, 2.5, 2.5), None, False),
}
WORKSPACE_SOURCES = {
    "table_1": "dining_surface",
    "counter_4": "food_preparation",
    "coffee table_7": "living_surface",
    "shelf_9": "storage",
}
ROOM_LABELS = {"R_A": "kitchen", "R_B": "living room"}


def toy_occupancy() -> np.ndarray:
    cols, rows = int(SIZE[0] / RESOLUTION), int(SIZE[1] / RESOLUTION)
    occ = np.zeros((rows, cols), dtype=np.int16)

    def block(x0, y0, x1, y1):
        c0, c1 = int(round(x0 / RESOLUTION)), int(round(x1 / RESOLUTION))
        r0, r1 = int(round(y0 / RESOLUTION)), int(round(y1 / RESOLUTION))
        occ[max(r0, 0):r1, max(c0, 0):c1] = 100

    t = 0.1
    block(0, 0, SIZE[0], t)
    block(0, SIZE[1] - t, SIZE[0], SIZE[1])
    block(0, 0, t, SIZE[1])
    block(SIZE[0] - t, 0, SIZE[0], SIZE[1])
    block(5.0, 0, 5.1, DOOR_Y[0])
    block(5.0, DOOR_Y[1], 5.1, SIZE[1])
    for _, _, lo, hi, _, obstacle in OBJECTS.values():
        if obstacle:
            block(lo[0], lo[1], hi[0], hi[1])
    return occ


def toy_grid(inflation_radius: float = 0.25) -> GridMap:
    free = traversable_from_occupancy(toy_occupancy(), RESOLUTION, inflation_radius)
    return GridMap(free, (0.0, 0.0), RESOLUTION)


def toy_graph(hidden_categories: Iterable[str] = ()) -> dict:
    """Workspace graph in the ``build_workspace_graph`` format.

    ``hidden_categories`` are removed, as by target masking, to create
    unknown targets.
    """
    hidden = {normalize_category(c) for c in hidden_categories}
    kept = {k: v for k, v in OBJECTS.items() if v[0] not in hidden}

    workspaces, objects = [], []
    for sid, function in WORKSPACE_SOURCES.items():
        if sid not in kept:
            continue
        workspaces.append({
            "workspace_id": f"workspace:{kept[sid][1]}:{sid}",
            "room_id": kept[sid][1],
            "source_object_id": sid,
            "function_label": function,
            "child_object_ids": sorted(
                k for k, v in kept.items() if v[4] == sid),
        })
    for oid, (category, room, lo, hi, parent, _) in kept.items():
        if oid in WORKSPACE_SOURCES:
            role = "source"
        elif parent in kept:
            role = "child"
        else:
            role = "standalone"
        objects.append({
            "object_id": oid,
            "room_id": room,
            "semantic_tag": category,
            "bbox": {"min": list(lo), "max": list(hi)},
            "role": role,
        })

    return {
        "schema_version": "0.1",
        "stage": "workspace_graph",
        "coordinate_frame": "athome_z_up",
        "rooms": [
            {"room_id": r, "room_label": label} for r, label in ROOM_LABELS.items()
        ],
        "workspaces": workspaces,
        "objects": objects,
    }


def write_toy_static_features(path, hidden_categories=(), rng_seed=0) -> None:
    """Static Features of the toy objects in the perception module format
    (one JSON list). Walls are structure, not objects; ``hidden`` categories
    are left out. Includes a fake CLIP feature and sampled surface points."""
    import json
    from pathlib import Path

    rng = np.random.default_rng(rng_seed)
    hidden = {normalize_category(c) for c in hidden_categories}
    objects = []
    for i, (oid, (category, _, lo, hi, _, _)) in enumerate(OBJECTS.items()):
        if category == "wall" or category in hidden:
            continue
        center = [(a + b) / 2 for a, b in zip(lo, hi)]
        points = rng.uniform(lo, hi, size=(20, 3))
        feature = rng.normal(size=8)
        objects.append({
            "object_id": i,
            "semantic_label": category,
            "clip_feature": (feature / np.linalg.norm(feature)).round(4).tolist(),
            "point_cloud": points.round(3).tolist(),
            "bbox_3d": {"center": center, "extent": [b - a for a, b in zip(lo, hi)], "yaw": 0.0},
            "centroid": points.mean(axis=0).round(3).tolist(),
        })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(objects, indent=1))


_TOY_SOURCES = {"table": "dining", "counter": "food_preparation",
                "coffee table": "living_surface", "shelf": "general_storage"}


def toy_labeler(messages, schema) -> dict:
    """Rule-based stand-in for the labeling LLM (same I/O as OpenAIChat)."""
    import json

    room = json.loads(messages[-1]["content"])
    categories = set(room["object_counts"])
    label = "kitchen" if {"fridge", "counter"} & categories else (
        "living_room" if "sofa" in categories else "unknown")
    return {
        "room_id": room["room_id"],
        "room_label": label,
        "workspace_sources": [
            {"source_object_id": o["id"], "function_label": _TOY_SOURCES[o["category"]]}
            for o in room["objects"] if o["category"] in _TOY_SOURCES
        ],
    }


def toy_world(moved: Dict[str, Tuple[float, float, float]] = None) -> List[tuple]:
    """Ground-truth objects as (object_id, category, centroid) for the
    simulator. ``moved`` overrides centroids to model objects that are no
    longer where the scene graph says."""
    moved = moved or {}
    out = []
    for oid, (category, _, lo, hi, _, _) in OBJECTS.items():
        center = tuple((a + b) / 2 for a, b in zip(lo, hi))
        out.append((oid, category, moved.get(oid, center)))
    return out
