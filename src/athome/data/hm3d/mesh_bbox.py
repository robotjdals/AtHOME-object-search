"""Object AABBs from HM3DSem semantic mesh triangles (athome_z_up).

Habitat's OBB-derived AABBs cover a single mesh piece and miss the rest for
~10% of objects (e.g. a chair's back). This rule uses the annotated geometry:

1. Split an instance's triangles into components: triangles whose vertices
   are within ``gap_m`` of each other are connected.
2. Drop components below ``min_area_share`` of the instance surface area
   (annotation specks).
3. Structural surfaces (wall/floor/ceiling) keep every remaining component:
   furniture occludes them, so one surface is legitimately split by large gaps.
4. Other objects are rigid: remaining components within ``object_gap_m`` of
   each other are one object (pieces split by reconstruction holes, e.g. seat
   and back). Only the largest-area group is kept; farther pieces are paint
   bleed or a neighbouring instance (e.g. part of the next window).
5. The AABB of the kept triangles is the object bbox.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class MeshBboxRule:
    version: str = "semantic_mesh_aabb_v2"
    gap_m: float = 0.10
    min_area_share: float = 0.05
    object_gap_m: float = 0.30
    # Same set as the workspace association policy's structural exclusions.
    structural_categories: Tuple[str, ...] = ("wall", "floor", "ceiling", "staircase wall")


def _areas(triangles):
    return 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)


def _components(triangles, gap_m):
    n = len(triangles)
    vertices = triangles.reshape(-1, 3)
    pairs = cKDTree(vertices).query_pairs(gap_m, output_type="ndarray")
    owner = np.repeat(np.arange(n), 3)
    rows = np.concatenate([owner[pairs[:, 0]], np.arange(n)]) if len(pairs) else np.arange(n)
    cols = np.concatenate([owner[pairs[:, 1]], np.arange(n)]) if len(pairs) else np.arange(n)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def _box(points):
    lo, hi = points.min(axis=0), points.max(axis=0)
    return {"center": ((lo + hi) / 2).tolist(), "size": (hi - lo).tolist(),
            "min": lo.tolist(), "max": hi.tolist()}


def is_structural(category: str, rule: MeshBboxRule) -> bool:
    return category.strip().casefold() in rule.structural_categories


def split_components(triangles, category: str, rule: MeshBboxRule = MeshBboxRule()):
    """Return (labels per triangle, area share per component, keep mask)."""
    triangles = np.asarray(triangles, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError("잘못된 삼각형 배열")
    areas = _areas(triangles)
    total = float(areas.sum())
    labels = _components(triangles, rule.gap_m)
    comp_area = np.bincount(labels, weights=areas)
    # Degenerate (zero-area) instances: equal shares.
    share = comp_area / total if total > 0 else np.ones(len(comp_area)) / len(comp_area)
    keep = share >= rule.min_area_share
    if not keep.any():
        keep[np.argmax(share)] = True
    if not is_structural(category, rule) and keep.sum() > 1:
        kept_ids = np.flatnonzero(keep)
        mask = keep[labels]
        groups = _components(triangles[mask], rule.object_gap_m)
        # Component -> group (components never straddle groups: gap_m < object_gap_m).
        group_of = {c: groups[np.flatnonzero(labels[mask] == c)[0]] for c in kept_ids}
        group_share = {}
        for c in kept_ids:
            group_share[group_of[c]] = group_share.get(group_of[c], 0.0) + share[c]
        best = max(group_share, key=lambda g: (group_share[g], -g))
        keep = np.zeros_like(keep)
        keep[[c for c in kept_ids if group_of[c] == best]] = True
    return labels, share, keep


def mesh_bbox(triangles, category: str, rule: MeshBboxRule = MeshBboxRule()):
    """Return (bbox dict, audit dict) for one instance's (N, 3, 3) triangles."""
    triangles = np.asarray(triangles, dtype=float)
    labels, share, keep = split_components(triangles, category, rule)
    main = _box(triangles[labels == np.argmax(share)].reshape(-1, 3))
    bbox = _box(triangles[keep[labels]].reshape(-1, 3))
    speck = share < rule.min_area_share
    audit = {
        "triangles": int(len(triangles)), "surface_area_m2": float(_areas(triangles).sum()),
        "structural": is_structural(category, rule),
        "components": int(len(share)), "kept_components": int(keep.sum()),
        "dropped_area_share": float(share[~keep].sum()),
        "dropped_speck_area_share": float(share[~keep & speck].sum()),
        # Substantial pieces dropped as detached from the rigid object.
        "dropped_detached_area_share": float(share[~keep & ~speck].sum()),
        "dropped_triangles": int((~keep[labels]).sum()),
        "extent_beyond_main_component_m": max(
            float(np.max(np.subtract(main["min"], bbox["min"]))),
            float(np.max(np.subtract(bbox["max"], main["max"])))),
    }
    return bbox, audit


def rule_dict(rule: MeshBboxRule):
    return asdict(rule)
