"""Types received from the perception module. Frame: map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class ObservedObject:
    # Persistent map object ID for static objects, -1 for dynamic ones.
    object_id: int
    label: str
    confidence: float
    centroid: Vec3
    is_static: bool = True
    bbox_center: Optional[Vec3] = None
    bbox_size: Optional[Vec3] = None
    bbox_yaw: float = 0.0
    # L2-normalized CLIP image feature.
    clip_feature: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class ObservationFrame:
    # Sensor capture time, not processing time.
    stamp: float
    objects: Tuple[ObservedObject, ...] = ()
