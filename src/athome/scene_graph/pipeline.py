"""Offline scene graph construction for the real environment (proposal 4).

Static Feature + 2D occupancy map
  -> room segmentation -> object-room association          (this module)
  -> geometric relations -> LLM labeling -> workspace graph (shared with HM3D)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from athome.scene_graph.geometric_relations import build_relations
from athome.scene_graph.object_rooms import RoomAssignment, assign_rooms
from athome.scene_graph.room_segmentation import (
    RoomSegmentation,
    SegmentationConfig,
    segment_rooms,
)
from athome.scene_graph.semantic_inputs import prepare_inputs
from athome.scene_graph.semantic_labeling import Complete, label_rooms
from athome.scene_graph.static_features import StaticFeatures
from athome.scene_graph.workspace_builder import build_workspace_graph


@dataclass(frozen=True)
class BuildConfig:
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    assign_radius: float = 0.5
    # Same thresholds as the HM3D graphs.
    distance_xy: float = 0.5
    delta_z: float = 0.1
    overlap_xy: float = 0.5


@dataclass
class BuildResult:
    graph: dict
    segmentation: RoomSegmentation
    assignment: RoomAssignment
    labels: dict
    empty_rooms: list


def build_scene(features: StaticFeatures, assignment: RoomAssignment) -> Tuple[dict, list]:
    """Scene in the ``room_object_map`` format. Rooms without objects are
    left out (nothing to search there). Returns (scene, empty room IDs)."""
    objects, by_room = [], {}
    for obj in features.objects:
        rid = assignment.rooms.get(obj.object_id)
        if rid is None:
            continue
        lo, hi = list(obj.bbox_min), list(obj.bbox_max)
        objects.append({
            "object_id": obj.object_id,
            "semantic_tag": obj.label,
            "room_id": rid,
            "bbox": {
                "min": lo,
                "max": hi,
                "size": [b - a for a, b in zip(lo, hi)],
                "center": [(a + b) / 2 for a, b in zip(lo, hi)],
            },
        })
        by_room.setdefault(rid, []).append(obj.object_id)
    scene = {
        "schema_version": "0.2",
        "stage": "room_object_map",
        "coordinate_frame": "athome_z_up",
        "up_axis": "z",
        "length_unit": "meter",
        "bbox_method": "aabb_from_obb",
        "rooms": [
            {"room_id": rid, "object_ids": sorted(ids)}
            for rid, ids in sorted(by_room.items())
        ],
        "objects": objects,
    }
    return scene, by_room


def build_scene_graph(
    features: StaticFeatures,
    free: np.ndarray,
    origin: Tuple[float, float],
    resolution: float,
    complete: Complete,
    config: BuildConfig = BuildConfig(),
) -> BuildResult:
    """``free``: raw free space of the occupancy map (no robot inflation)."""
    seg = segment_rooms(free, origin, resolution, config.segmentation)
    assignment = assign_rooms(
        features.objects, seg, features.points, config.assign_radius)
    scene, by_room = build_scene(features, assignment)
    if not scene["objects"]:
        raise ValueError("Room에 배정된 객체 없음")

    geometry = build_relations(
        scene, config.distance_xy, config.delta_z, config.overlap_xy)
    inputs = prepare_inputs(scene, geometry)
    labels = label_rooms(inputs, complete)
    graph = build_workspace_graph(inputs, labels)
    graph["provenance"] = {
        "map_version": features.map_version,
        "clip_model": features.clip_model,
        "room_count": len(seg.room_ids),
        "unassigned_objects": assignment.unassigned,
    }
    empty = [r for r in seg.room_ids if r not in by_room]
    return BuildResult(graph, seg, assignment, labels, empty)
