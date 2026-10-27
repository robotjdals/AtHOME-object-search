"""Static Feature export from the perception module (proposal 3-1, F_j).

Perception format (one object; a file holds a list of these, or a directory
holds one file per object):

    {"object_id": 0, "semantic_label": "lamp",
     "clip_feature": [...], "point_cloud": [[x, y, z], ...],
     "bbox_3d": {"center": [x, y, z], "extent": [l, w, h], "yaw": 1.52},
     "centroid": [x, y, z]}

``extent`` is the full box size along its axes (Open3D convention), before
the ``yaw`` rotation about z. Coordinates are in the map frame of the 2D map
used for the scene graph.

Perception handoff v0.8 (its objects.schema.json): {"frame_id": "map",
"objects": [{"object_id", "semantic_tag", "confidence", "centroid": {x, y, z},
"bbox_3d": {"center", "size", "yaw"}, "clip_feature", "num_observations"}]}
with ``points/obj_<id:04d>.npy``.

Also accepted: ``<dir>/objects.json`` with {"map_version", "clip_model",
"objects": [{"id", "label", "confidence", "n_observations", "centroid",
"bbox": {"center", "size", "yaw"}}]} plus optional ``features.npy`` (N, D)
and ``points/<id>.npy``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class StaticObject:
    object_id: str             # "<label>_<id>", unique in the scene
    source_id: int             # perception object ID
    label: str
    confidence: float
    n_observations: int
    centroid: Vec3
    bbox_min: Vec3             # axis-aligned, from the oriented box
    bbox_max: Vec3
    row: int                   # row in ``features``


@dataclass
class StaticFeatures:
    map_version: Optional[str]
    clip_model: Optional[str]
    objects: List[StaticObject]
    features: Optional[np.ndarray] = None
    root: Optional[Path] = None
    inline_points: Dict[int, np.ndarray] = field(default_factory=dict)
    # sha256 of the map_server files the objects were built on, from the
    # handoff's metadata.json (occupancy.map_version.sha256); None if absent.
    map_file_sha256: Optional[Tuple[str, ...]] = None

    def built_on_map(self, map_yaml) -> Optional[bool]:
        """Whether the objects were built on this map_server map (same yaml
        and image file contents, whatever the names); None without a record."""
        if self.map_file_sha256 is None:
            return None
        import hashlib
        import yaml

        map_yaml = Path(map_yaml)
        image = map_yaml.parent / yaml.safe_load(map_yaml.read_text())["image"]
        ours = {hashlib.sha256(f.read_bytes()).hexdigest() for f in (map_yaml, image)}
        return ours == set(self.map_file_sha256)

    def points(self, obj: StaticObject) -> Optional[np.ndarray]:
        if obj.source_id in self.inline_points:
            return self.inline_points[obj.source_id]
        if self.root is None:
            return None
        for name in (f"{obj.source_id}.npy", f"obj_{obj.source_id:04d}.npy"):   # v0.8 name
            path = self.root / "points" / name
            if path.exists():
                return np.load(path)
        return None


def oriented_to_aabb(center, size, yaw) -> Tuple[Vec3, Vec3]:
    hx, hy, hz = (s / 2 for s in size)
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    ex, ey = hx * c + hy * s, hx * s + hy * c
    cx, cy, cz = center
    return (cx - ex, cy - ey, cz - hz), (cx + ex, cy + ey, cz + hz)


def _normalize(o: dict) -> dict:
    """Any input format -> common fields."""
    if "semantic_tag" in o:        # perception handoff v0.8 (objects.schema.json)
        box, c = o["bbox_3d"], o["centroid"]
        return {
            "id": o["object_id"],
            "label": o["semantic_tag"],
            "confidence": o["confidence"],
            "n_observations": o["num_observations"],
            "centroid": [c["x"], c["y"], c["z"]],
            "center": box["center"],
            "size": box["size"],
            "yaw": box["yaw"],
            "feature": o.get("clip_feature"),
            "points": None,              # points/obj_<id:04d>.npy
        }
    if "semantic_label" in o:      # perception format
        box = o["bbox_3d"]
        return {
            "id": o["object_id"],
            "label": o["semantic_label"],
            "confidence": o.get("confidence", 1.0),
            "n_observations": o.get("n_observations", 1),
            "centroid": o["centroid"],
            "center": box["center"],
            "size": box["extent"],
            "yaw": box.get("yaw", 0.0),
            "feature": o.get("clip_feature"),
            "points": o.get("point_cloud"),
        }
    box = o["bbox"]
    return {
        "id": o["id"],
        "label": o["label"],
        "confidence": o.get("confidence", 1.0),
        "n_observations": o.get("n_observations", 1),
        "centroid": o["centroid"],
        "center": box["center"],
        "size": box["size"],
        "yaw": box.get("yaw", 0.0),
        "feature": None,
        "points": None,
    }


def _read(path: Path) -> Tuple[dict, List[dict], Optional[Path]]:
    """(metadata, raw objects, directory with objects.json-side files)."""
    if path.is_dir():
        if (path / "objects.json").exists():
            raw = json.loads((path / "objects.json").read_text(encoding="utf-8"))
            return raw, raw["objects"], path
        files = sorted(path.glob("*.json"))
        if not files:
            raise ValueError(f"{path}: JSON 파일 없음")
        return {}, [json.loads(f.read_text(encoding="utf-8")) for f in files], None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {}, raw, None
    return raw, raw["objects"], path.parent


def load_static_features(path, min_observations: int = 1) -> StaticFeatures:
    meta, raw_objects, root = _read(Path(path))
    if meta.get("frame_id", "map") != "map":
        raise ValueError(f"{path}: frame_id {meta['frame_id']!r} (map 좌표계여야 함)")

    objects, seen, inline_points, inline_features = [], set(), {}, []
    for row, raw in enumerate(raw_objects):
        o = _normalize(raw)
        sid = int(o["id"])
        if sid in seen:
            raise ValueError(f"중복 객체 ID: {sid}")
        seen.add(sid)
        values = list(o["center"]) + list(o["size"]) + list(o["centroid"])
        if (
            len(o["center"]) != 3 or len(o["size"]) != 3 or len(o["centroid"]) != 3
            or not all(math.isfinite(v) for v in values) or min(o["size"]) < 0
        ):
            raise ValueError(f"{sid}: 잘못된 bbox/centroid")
        inline_features.append(o["feature"])
        if o["points"]:
            inline_points[sid] = np.asarray(o["points"], dtype=float).reshape(-1, 3)
        if int(o["n_observations"]) < min_observations:
            continue
        lo, hi = oriented_to_aabb(o["center"], o["size"], float(o["yaw"]))
        label = " ".join(str(o["label"]).split())
        objects.append(StaticObject(
            object_id=f"{label}_{sid}",
            source_id=sid,
            label=label,
            confidence=float(o["confidence"]),
            n_observations=int(o["n_observations"]),
            centroid=tuple(o["centroid"]),
            bbox_min=lo,
            bbox_max=hi,
            row=row,
        ))

    features = None
    if all(f is not None for f in inline_features) and inline_features:
        if len({len(f) for f in inline_features}) != 1:
            raise ValueError("clip_feature 차원이 객체마다 다름")
        features = np.asarray(inline_features, dtype=np.float32)
    elif root is not None and (root / "features.npy").exists():
        features = np.load(root / "features.npy")
        if features.shape[0] != len(raw_objects):
            raise ValueError("features.npy 행 수와 객체 수 불일치")

    record = root / "metadata.json" if root is not None else None
    map_files = None
    if record is not None and record.exists():
        version = (json.loads(record.read_text(encoding="utf-8")).get("occupancy") or {}).get("map_version")
        if version and version.get("sha256"):
            map_files = tuple(version["sha256"].values())

    return StaticFeatures(
        map_file_sha256=map_files,
        map_version=meta.get("map_version"),
        clip_model=meta.get("clip_model"),
        objects=objects,
        features=features,
        root=root,
        inline_points=inline_points,
    )
