"""2D line-of-sight observation against walls (symbolic environment).

The symbolic environment does not simulate the robot's sensor (proposal
6.2). Detection follows the sensor model of multi-object search (Wandzel et
al., ICRA 2019, pomdp_py): static structures such as walls block the line of
sight and the sensor has a limited range. The robot turns to six headings at
a goal, so the sensing region is a disk.

Occluders are opaque structures in their scanned state -- walls and door
leaves -- the same state the NavMesh was built from (a door captured closed
also separates the NavMesh). Glass (windows, shower doors) does not block.
They are kept as exact 2D geometry instead of grid cells: semantic-mesh
triangles of ``occluder_categories`` clipped to a height band above the floor
(excludes other storeys) and projected to XY. Rasterizing them made objects
standing against the far side of a wall visible, because their bbox shares
cells with the wall.

Standard ray casting: an instance is observed if its footprint is within
``range_m`` (nearest point, like a depth sensor's closest surface) and, for
some sample point of the footprint, the first occluder hit along the segment
is no closer than ``contact_tolerance_m`` before the point (objects flush with a
wall stay visible from the room side). Sample points are the 3x3 interior
grid at 1/4, 1/2, 3/4 of the footprint, not its edges, because annotated
bbox edges often coincide with wall surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class WallLosSpec:
    range_m: float
    occluder_categories: Tuple[str, ...]
    band_above_floor_m: Tuple[float, float]
    contact_tolerance_m: float

    def __post_init__(self):
        lo, hi = self.band_above_floor_m
        if not (self.range_m > 0 and hi > lo and self.contact_tolerance_m >= 0):
            raise ValueError("잘못된 벽 가림 설정")


def clip_to_z_band(triangle, z0, z1):
    """Polygon (k, 3) of a triangle clipped to z0 <= z <= z1 (may be empty)."""
    poly = [np.asarray(p, dtype=float) for p in triangle]
    for bound, keep_above in ((z0, True), (z1, False)):
        out = []
        for i, cur in enumerate(poly):
            prev = poly[i - 1]
            cur_in = cur[2] >= bound if keep_above else cur[2] <= bound
            prev_in = prev[2] >= bound if keep_above else prev[2] <= bound
            if cur_in != prev_in:
                t = (bound - prev[2]) / (cur[2] - prev[2])
                out.append(prev + t * (cur - prev))
            if cur_in:
                out.append(cur)
        poly = out
        if not poly:
            return np.zeros((0, 3))
    return np.array(poly)


class WallGeometry:
    """Projected occluder pieces in one height band, indexed for segment queries."""

    def __init__(self, wall_triangles: Sequence[np.ndarray], z_band):
        import shapely
        from shapely.geometry import MultiPoint

        self._shapely = shapely
        self.geoms = []
        for tri in wall_triangles:
            pts = clip_to_z_band(tri, *z_band)
            if len(pts):
                self.geoms.append(MultiPoint(pts[:, :2]).convex_hull)
        self.tree = shapely.STRtree(self.geoms) if self.geoms else None

    def first_hit(self, p0, p1):
        """Distance from p0 to the first occluder on segment p0-p1, or None."""
        if self.tree is None:
            return None
        from shapely.geometry import LineString, Point
        segment, start = LineString([p0, p1]), Point(p0)
        hits = self.tree.query(segment, predicate="intersects")
        if not len(hits):
            return None
        return min(start.distance(segment.intersection(self.geoms[i])) for i in hits)


def footprint_samples(lo, hi):
    """3x3 interior points at 1/4, 1/2, 3/4 of the XY footprint."""
    (x0, y0), (x1, y1) = lo[:2], hi[:2]
    fractions = (0.25, 0.5, 0.75)
    return [(x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)) for fy in fractions for fx in fractions]


class WallLosObserver:
    """Observer for SymbolicEnvironment. Isotropic: observes once per visit."""

    per_heading = False

    def __init__(self, walls: WallGeometry,
                 instances: Mapping[int, Tuple[Sequence[float], Sequence[float]]],
                 spec: WallLosSpec):
        self.walls, self.spec = walls, spec
        self.instances = {int(k): (tuple(v[0]), tuple(v[1])) for k, v in instances.items()}

    def visible(self, x, y, lo, hi):
        """Distance to the footprint if in range and some interior sample is
        unoccluded, else None. Range is measured to the nearest footprint
        point (the closest surface, as a depth sensor does)."""
        near = math.hypot(max(lo[0] - x, 0.0, x - hi[0]), max(lo[1] - y, 0.0, y - hi[1]))
        if near > self.spec.range_m:
            return None
        for px, py in footprint_samples(lo, hi):
            d = math.hypot(px - x, py - y)
            hit = self.walls.first_hit((x, y), (px, py))
            if hit is None or hit >= d - self.spec.contact_tolerance_m:
                return near
        return None

    def observe(self, x: float, y: float, floor_z: float, yaw: float) -> Dict[int, float]:
        """semantic_id -> distance (m) of observed instances."""
        out = {}
        for sid, (lo, hi) in self.instances.items():
            d = self.visible(x, y, lo, hi)
            if d is not None:
                out[sid] = round(d, 4)
        return out
