"""Offline scene graph construction for the real environment (proposal 4).

Static Feature + 2D occupancy map
  -> room segmentation -> object-room association          (this module)
  -> geometric relations -> LLM labeling -> workspace graph (shared with HM3D)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from athome.data.hm3d.scope import apply_storage_scope

from athome.scene_graph.geometric_relations import build_relations
from athome.scene_graph.object_rooms import RoomAssignment, assign_rooms
from athome.scene_graph.room_segmentation import (
    RoomSegmentation,
    SegmentationConfig,
    segment_rooms,
)
from athome.scene_graph.semantic_inputs import prepare_inputs
from athome.scene_graph.semantic_labeling import PROTOCOL, PROTOCOLS, MODEL, Sample, label_rooms
from athome.scene_graph.source_review import (
    ASSOCIATION_POLICY, apply_source_policy, validate_workspace_graph)
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
    # Storage scope rule of the training graphs (configs/data/scene_scope.json):
    # objects inside storage furniture are removed before labeling.
    scene_scope: Optional[dict] = None
    # Floor height in the map frame, written to every room (room.floor_z_m)
    # so the search-height scope (query.MAX_BASE_ABOVE_FLOOR_M) applies.
    floor_z_m: Optional[float] = None


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
    sample: Sample,
    config: BuildConfig = BuildConfig(),
) -> BuildResult:
    """``free``: raw free space of the occupancy map (no robot inflation)."""
    seg = segment_rooms(free, origin, resolution, config.segmentation)
    assignment = assign_rooms(
        features.objects, seg, features.points, config.assign_radius)
    scene, by_room = build_scene(features, assignment)
    contained = {}
    if config.scene_scope is not None:
        # Same rule and stage as the HM3D maps (scripts/apply_scene_scope.py).
        scene, contained = apply_storage_scope(scene, config.scene_scope)
        scene["rooms"] = [r for r in scene["rooms"] if r["object_ids"]]
        by_room = {r["room_id"]: r["object_ids"] for r in scene["rooms"]}
    if not scene["objects"]:
        raise ValueError("Room에 배정된 객체 없음")

    geometry = build_relations(
        scene, config.distance_xy, config.delta_z, config.overlap_xy)
    inputs = prepare_inputs(scene, geometry)
    # Same labeling protocol and source policy as the HM3D training graphs
    # (scripts/build_hm3d_workspace_graph.py), so real and training graphs match.
    labels = label_rooms(inputs, sample)
    objects = {o["object_id"]: o for room in inputs["rooms"] for o in room["objects"]}
    reviewed = apply_source_policy(labels, objects, "real_environment_source_policy")
    graph = build_workspace_graph(inputs, reviewed)
    validate_workspace_graph(graph, objects, "real_environment")
    if config.floor_z_m is not None:
        for room in graph["rooms"]:
            room["floor_z_m"] = config.floor_z_m
    graph["association_policy"] = dict(ASSOCIATION_POLICY)
    graph["provenance"] = {
        "map_version": features.map_version,
        "clip_model": features.clip_model,
        "room_count": len(seg.room_ids),
        "unassigned_objects": assignment.unassigned,
        "scope_excluded_objects": contained,
        "floor_z_m": config.floor_z_m,
        "labeling_protocol": {"name": labels.get("protocol", PROTOCOL),
                              **PROTOCOLS[labels.get("protocol", PROTOCOL)], "model": MODEL},
        "source_review": reviewed["review"],
    }
    empty = [r for r in seg.room_ids if r not in by_room]
    return BuildResult(graph, seg, assignment, reviewed, empty)
