"""Split a NavMesh island into floor levels and stairs (replaces manual review).

Level segmentation of navigable surfaces:
1. Flat triangles (slope <= ``flat_slope_deg``) sharing vertices form flat
   regions; regions of at least ``min_platform_area_m2`` are floor platforms
   (small flat pieces are stair treads or landings).
2. Stairs start as the triangles that leave every platform's height band
   (platform height +- the NavMesh step height ``agent_max_climb``) and grow
   through adjacent sloped triangles: the first step of a staircase is inside
   the band but belongs to the staircase.
3. A triangle belongs to a platform's level if it lies within the band, is
   not stairs, and connects to the platform through such triangles.
Connected kept triangles form the navigable components of the island.

The step height comes from the NavMesh settings, the same traversability
model used for the grids and paths. Both ends of a staircase are treated
alike (the manual review had kept the first sloped step on one side only).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from typing import Dict, List

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


DEGENERATE_AREA_M2 = 1e-9


@dataclass(frozen=True)
class LevelRule:
    version: str = "navmesh_level_segmentation_v1"
    flat_slope_deg: float = 10.0
    min_platform_area_m2: float = 1.0


def island_geometry(pathfinder, island):
    """Vertices (athome_z_up), faces and geometry hash of one NavMesh island."""
    native = np.asarray(pathfinder.build_navmesh_vertices(island), dtype=float)
    vertices = np.column_stack((native[:, 0], -native[:, 2], native[:, 1]))  # habitat -> athome_z_up
    faces = np.asarray(pathfinder.build_navmesh_vertex_indices(island), dtype=np.int64).reshape(-1, 3)
    signature = hashlib.sha256(vertices.astype("<f8").tobytes()
                               + faces.astype("<i8").tobytes()).hexdigest()
    return vertices, faces, signature


def selection_islands(selection: Dict) -> List[Dict]:
    """Island records of a selection file: the multi-island schema lists them
    under ``islands``; the single-island schema is one record itself."""
    return selection["islands"] if "islands" in selection else [selection]


def triangle_geometry(vertices, faces):
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normal, axis=1)
    area = norm / 2
    slope = np.degrees(np.arccos(np.clip(np.abs(normal[:, 2]) / np.maximum(norm, 1e-12), 0, 1)))
    return tri, area, slope


def _components(faces_vid, subset):
    """Connected components (shared vertex) of the triangles in ``subset``."""
    subset = np.asarray(sorted(subset), dtype=int)
    if not len(subset):
        return subset, np.zeros(0, dtype=int)
    owners = defaultdict(list)
    for t in subset:
        for v in faces_vid[t]:
            owners[v].append(t)
    pos = {t: k for k, t in enumerate(subset)}
    rows, cols = list(range(len(subset))), list(range(len(subset)))
    for ts in owners.values():
        for a in ts[1:]:
            rows.append(pos[a])
            cols.append(pos[ts[0]])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(subset),) * 2)
    return subset, connected_components(graph, directed=False)[1]


def segment_levels(vertices, faces, max_climb, rule: LevelRule = LevelRule()) -> Dict:
    """vertices (athome_z_up), faces of one island -> components and stairs."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    tri, area, slope = triangle_geometry(vertices, faces)
    # Zero-area (collinear) triangles, which Recast can emit, carry no
    # navigable surface and have no defined slope: excluded from every level
    # (standard degenerate-face removal).
    valid = area > DEGENERATE_AREA_M2
    # Duplicate vertices at identical positions still connect triangles.
    _, vid = np.unique(np.round(vertices, 6), axis=0, return_inverse=True)
    faces_vid = np.asarray(vid).reshape(-1)[faces]

    flat = np.flatnonzero((slope <= rule.flat_slope_deg) & valid)
    ids, labels = _components(faces_vid, flat)
    platforms = []
    for label in np.unique(labels):
        members = ids[labels == label]
        if area[members].sum() >= rule.min_platform_area_m2:
            platforms.append({"triangles": members,
                              "height_m": float(np.median(tri[members][:, :, 2]))})

    zmin, zmax = tri[:, :, 2].min(axis=1), tri[:, :, 2].max(axis=1)
    in_band = np.zeros(len(faces), dtype=bool)
    for platform in platforms:
        z = platform["height_m"]
        in_band |= (zmin >= z - max_climb - 1e-6) & (zmax <= z + max_climb + 1e-6)
    neighbours = defaultdict(set)
    owners = defaultdict(list)
    for t, verts in enumerate(faces_vid):
        for v in verts:
            owners[v].append(t)
    for ts in owners.values():
        for a in ts:
            neighbours[a].update(ts)
    sloped = (slope > rule.flat_slope_deg) & valid
    stairs = set(np.flatnonzero(~in_band & valid).tolist())
    frontier = list(stairs)
    while frontier:
        t = frontier.pop()
        for u in neighbours[t]:
            if u not in stairs and sloped[u]:
                stairs.add(u)
                frontier.append(u)

    kept = set()
    for platform in platforms:
        z = platform["height_m"]
        band = np.flatnonzero((zmin >= z - max_climb - 1e-6) & (zmax <= z + max_climb + 1e-6))
        band = np.array([t for t in band if t not in stairs and valid[t]], dtype=int)
        b_ids, b_labels = _components(faces_vid, band)
        home = {b_labels[k] for k, t in enumerate(b_ids) if t in set(platform["triangles"])}
        kept |= {int(t) for k, t in enumerate(b_ids) if b_labels[k] in home}

    c_ids, c_labels = _components(faces_vid, kept)
    components: List[Dict] = []
    for label in np.unique(c_labels):
        members = sorted(int(t) for t in c_ids[c_labels == label])
        member_set = set(members)
        heights = sorted(p["height_m"] for p in platforms
                         if set(int(t) for t in p["triangles"]) <= member_set)
        components.append({"triangle_ids": members, "platform_heights_m": heights,
                           "area_m2": float(area[members].sum()),
                           "z_range_m": [float(zmin[members].min()), float(zmax[members].max())]})
    components.sort(key=lambda c: -c["area_m2"])
    excluded = sorted(set(range(len(faces))) - kept)
    extra = {"degenerate_triangle_ids": np.flatnonzero(~valid).tolist()} if not valid.all() else {}
    return {**extra,"components": components, "excluded_triangle_ids": excluded,
            "retained_triangle_ids": sorted(kept),
            "platforms": [{"height_m": p["height_m"], "triangles": len(p["triangles"]),
                           "area_m2": float(area[p["triangles"]].sum())} for p in platforms]}


def navmesh_components(vertices, faces) -> Dict:
    """Components of a NavMesh built for the robot's own step and slope limits.

    Such a NavMesh already leaves out stairs and steps the robot cannot take
    (Recast agent_max_climb / agent_max_slope), so no level rule is applied:
    every non-degenerate triangle is kept and connected triangles form a
    component. Same result schema as ``segment_levels``.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    tri, area, _ = triangle_geometry(vertices, faces)
    valid = area > DEGENERATE_AREA_M2
    _, vid = np.unique(np.round(vertices, 6), axis=0, return_inverse=True)
    faces_vid = np.asarray(vid).reshape(-1)[faces]
    kept = set(np.flatnonzero(valid).tolist())
    ids, labels = _components(faces_vid, kept)
    zmin, zmax = tri[:, :, 2].min(axis=1), tri[:, :, 2].max(axis=1)
    components = []
    for label in np.unique(labels):
        members = sorted(int(t) for t in ids[labels == label])
        components.append({"triangle_ids": members, "platform_heights_m": [],
                           "area_m2": float(area[members].sum()),
                           "z_range_m": [float(zmin[members].min()), float(zmax[members].max())]})
    components.sort(key=lambda c: -c["area_m2"])
    out = {"components": components, "excluded_triangle_ids": np.flatnonzero(~valid).tolist(),
           "retained_triangle_ids": sorted(kept), "platforms": []}
    if not valid.all():
        out["degenerate_triangle_ids"] = np.flatnonzero(~valid).tolist()
    return out
