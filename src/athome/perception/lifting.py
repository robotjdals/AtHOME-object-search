"""3D boxes for 2D detections from LiDAR points (real-time perception).

The robot's camera has no depth (D435 over USB 2.0), so a detection's mask
is lifted with the Mid-360 points around the image's capture time, moved to
the map frame with each scan's own pose. Chosen on the 412 run5 frames
(scripts/evaluate_detectors_412.py, real scans): small objects come within
~1 cm in bottom height, large ones within ~2 cm in top height.
"""
from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

# Points behind the nearest one of their image cell by more than this are
# hidden (the scans come from several poses).
CELL_PX = 16
HIDDEN_M = 0.1
# Depth jump that separates surfaces, and the share of the mask's points the
# object's group must hold (nearer, smaller groups are occluding fragments).
GAP_M = 0.15
MIN_SHARE = 0.1
MIN_POINTS = 3


@dataclass(frozen=True)
class Camera:
    K: np.ndarray            # 3x3 intrinsics
    width: int
    height: int
    T_cam_body: np.ndarray   # 4x4, body (LiDAR/IMU) -> camera


@dataclass(frozen=True)
class Box:
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    points: int


class ScanWindow:
    """Recent scans in the map frame, kept for ``keep_s`` seconds."""

    def __init__(self, keep_s: float = 3.0):
        self.keep_s = keep_s
        self._scans: deque = deque()

    def add(self, stamp: float, points_map: np.ndarray) -> None:
        self._scans.append((stamp, np.asarray(points_map, dtype=np.float32)))
        while self._scans and self._scans[0][0] < stamp - self.keep_s:
            self._scans.popleft()

    @property
    def latest(self) -> Optional[float]:
        return self._scans[-1][0] if self._scans else None

    def points(self, start: float, end: float) -> np.ndarray:
        parts = [p for t, p in self._scans if start <= t <= end]
        return np.concatenate(parts) if parts else np.zeros((0, 3), dtype=np.float32)


def transform(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


class FrameView:
    """Points of one image: those in view and in front in their cell."""

    def __init__(self, camera: Camera, T_map_cam: np.ndarray, points_map: np.ndarray,
                 max_range: float = 8.0):
        Pc = transform(np.linalg.inv(T_map_cam), np.asarray(points_map, float))
        z = Pc[:, 2]
        ok = (z > 0.3) & (z < max_range)
        uv = np.full((len(Pc), 2), -1.0)
        uv[ok] = (Pc[ok] @ camera.K.T)[:, :2] / z[ok, None]
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < camera.width) & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
        uv, z, P = uv[ok], z[ok], np.asarray(points_map, float)[ok]
        zb = np.full((camera.height // CELL_PX + 1, camera.width // CELL_PX + 1), np.inf)
        cu, cv = (uv[:, 0] // CELL_PX).astype(int), (uv[:, 1] // CELL_PX).astype(int)
        np.minimum.at(zb, (cv, cu), z)
        front = z <= zb[cv, cu] + HIDDEN_M
        self.uv, self.z, self.xyz = uv[front], z[front], P[front]

    def lift(self, region: np.ndarray) -> Optional[Box]:
        """Box of the object in ``region`` (bool image): the nearest depth
        group holding at least MIN_SHARE of the region's points. The
        open-source mappers keep the largest DBSCAN cluster instead
        (ConceptGraphs, VLFM); on the 412 scans that matched this rule for
        small objects but kept a fragment of large ones (top height off by
        9 cm vs 2 cm), as their parameters are set for dense depth images."""
        uv = self.uv.astype(int)
        keep = region[uv[:, 1], uv[:, 0]]
        if keep.sum() < MIN_POINTS:
            return None
        z, P = self.z[keep], self.xyz[keep]
        order = np.argsort(z)
        groups = np.split(order, np.nonzero(np.diff(z[order]) > GAP_M)[0] + 1)
        need = max(MIN_POINTS, int(np.ceil(MIN_SHARE * len(z))))
        group = next((g for g in groups if len(g) >= need), None)
        if group is None:
            return None
        lo, hi = P[group].min(0), P[group].max(0)
        return Box(tuple(float(v) for v in (lo + hi) / 2), tuple(float(v) for v in hi - lo), len(group))


def box_region(box: Sequence[float], width: int, height: int) -> np.ndarray:
    region = np.zeros((height, width), dtype=bool)
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(width, int(np.ceil(box[2]))), min(height, int(np.ceil(box[3])))
    region[y0:y1, x0:x1] = True
    return region


def pack_mask(m: np.ndarray) -> dict:
    """Mask cropped to its pixels' box, bits packed row-major."""
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return {"x0": 0, "y0": 0, "w": 0, "h": 0, "bits": ""}
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    bits = np.packbits(m[y0:y1, x0:x1].ravel())
    return {"x0": int(x0), "y0": int(y0), "w": int(x1 - x0), "h": int(y1 - y0),
            "bits": base64.b64encode(bits.tobytes()).decode("ascii")}


def unpack_mask(m: dict, width: int, height: int) -> np.ndarray:
    full = np.zeros((height, width), dtype=bool)
    if m["w"]:
        bits = np.unpackbits(np.frombuffer(base64.b64decode(m["bits"]), dtype=np.uint8))[:m["w"] * m["h"]]
        full[m["y0"]:m["y0"] + m["h"], m["x0"]:m["x0"] + m["w"]] = bits.reshape(m["h"], m["w"]).astype(bool)
    return full
