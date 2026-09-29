"""Floor height of a NavMesh component at an XY point (athome_z_up)."""
from __future__ import annotations

import numpy as np


class NavmeshSurface:
    def __init__(self, vertices, faces, ambiguity_tol_m: float = 0.05):
        self._tri = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=int)]
        if self._tri.ndim != 3 or self._tri.shape[1:] != (3, 3) or not len(self._tri):
            raise ValueError("잘못된 NavMesh 삼각형")
        if not np.isfinite(self._tri).all():
            raise ValueError("NavMesh에 비유한 값이 있습니다.")
        self._tol = ambiguity_tol_m

    def height_at(self, x: float, y: float) -> float:
        a, b, c = self._tri[:, 0], self._tri[:, 1], self._tri[:, 2]
        v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
        v2 = np.array([x, y]) - a[:, :2]
        det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        valid = np.abs(det) > 1e-12  # vertical triangles have no XY footprint
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]) / det
            v = (v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0]) / det
        eps = 1e-9
        inside = valid & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps)
        if not inside.any():
            raise ValueError(f"NavMesh 밖의 XY: ({x:.3f}, {y:.3f})")
        z = (a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2]))[inside]
        if z.max() - z.min() > self._tol:
            raise ValueError(f"겹친 NavMesh 층: ({x:.3f}, {y:.3f}) Z={z.min():.3f}..{z.max():.3f}")
        return float(z.mean())
