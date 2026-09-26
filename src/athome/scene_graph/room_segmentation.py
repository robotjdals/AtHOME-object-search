"""Room segmentation of a 2D occupancy map (proposal 4-1).

Distance transform of free space -> local maxima as room markers ->
marker-controlled watershed on -D -> merge over-segmented regions.

Rooms have large clearance inside and meet through narrow openings
(doors), so two regions are merged when the clearance where they touch is
close to the clearance inside them.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class SegmentationConfig:
    # Markers: local maxima of clearance at least this far apart [m] ...
    marker_separation: float = 1.0
    # ... and with at least this clearance [m]. Low enough for narrow rooms
    # (closets); spurious markers in large rooms are merged afterwards.
    min_marker_clearance: float = 0.4
    # Merge when passage clearance / min(inner clearance) exceeds this.
    passage_ratio: float = 0.7
    # Regions smaller than this [m^2] are merged into a neighbor.
    min_room_area: float = 1.0


@dataclass
class RoomSegmentation:
    labels: np.ndarray                 # int32, 0 = no room
    origin: Tuple[float, float]
    resolution: float

    @property
    def room_ids(self):
        return [f"room_{i}" for i in np.unique(self.labels) if i > 0]

    def area(self, index: int) -> float:
        return float((self.labels == index).sum()) * self.resolution ** 2


def segment_rooms(
    free: np.ndarray,
    origin: Tuple[float, float],
    resolution: float,
    config: SegmentationConfig = SegmentationConfig(),
) -> RoomSegmentation:
    """``free`` is raw free space (not inflated by the robot radius)."""
    free = free.astype(bool)
    clearance = ndimage.distance_transform_edt(free) * resolution

    size = max(3, int(round(config.marker_separation / resolution)) | 1)
    peaks = (
        free
        & (clearance >= config.min_marker_clearance)
        & (clearance == ndimage.maximum_filter(clearance, size=size))
    )
    markers, n = ndimage.label(peaks, structure=np.ones((3, 3)))

    # Free components without a marker (narrow rooms): one marker each if
    # large enough, otherwise left out.
    components, count = ndimage.label(free)
    min_cells = config.min_room_area / resolution ** 2
    for comp in range(1, count + 1):
        mask = components == comp
        if markers[mask].any() or mask.sum() < min_cells:
            continue
        n += 1
        r, c = np.unravel_index(np.argmax(np.where(mask, clearance, -1)), free.shape)
        markers[r, c] = n

    labels = _flood(free, clearance, markers)
    labels = _merge(labels, clearance, resolution, config)
    return RoomSegmentation(_relabel(labels), origin, resolution)


def _flood(free, clearance, markers) -> np.ndarray:
    """Watershed on -clearance: grow from markers, highest clearance first."""
    labels = markers.astype(np.int32).copy()
    h, w = free.shape
    queue = [(-clearance[r, c], r, c) for r, c in zip(*np.nonzero(labels))]
    heapq.heapify(queue)
    while queue:
        _, r, c = heapq.heappop(queue)
        lab = labels[r, c]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and labels[nr, nc] == 0:
                labels[nr, nc] = lab
                heapq.heappush(queue, (-clearance[nr, nc], nr, nc))
    return labels


def _boundaries(labels, clearance) -> Dict[Tuple[int, int], Tuple[float, int]]:
    """(a, b) -> (passage clearance, boundary length in cells)."""
    out: Dict[Tuple[int, int], list] = {}
    for a_sl, b_sl in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        la, lb = labels[a_sl], labels[b_sl]
        mask = (la > 0) & (lb > 0) & (la != lb)
        if not mask.any():
            continue
        pair_clear = np.minimum(clearance[a_sl], clearance[b_sl])[mask]
        for x, y, v in zip(la[mask], lb[mask], pair_clear):
            key = (min(x, y), max(x, y))
            entry = out.setdefault(key, [0.0, 0])
            entry[0] = max(entry[0], float(v))
            entry[1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def _merge(labels, clearance, resolution, config) -> np.ndarray:
    labels = labels.copy()
    while True:
        ids = [i for i in np.unique(labels) if i > 0]
        if len(ids) <= 1:
            return labels
        peak = dict(zip(ids, ndimage.maximum(clearance, labels, ids)))
        area = dict(zip(ids, ndimage.sum(np.ones_like(clearance), labels, ids)))
        bounds = _boundaries(labels, clearance)

        best = None
        for (a, b), (passage, length) in bounds.items():
            small = min(area[a], area[b]) * resolution ** 2 < config.min_room_area
            ratio = passage / max(min(peak[a], peak[b]), 1e-9)
            if small or ratio > config.passage_ratio:
                # Small regions first, then the widest passage.
                score = (small, length if small else ratio)
                if best is None or score > best[0]:
                    best = (score, a, b)
        if best is None:
            # Small regions with no neighbor to merge into are not rooms.
            for i in ids:
                if area[i] * resolution ** 2 < config.min_room_area:
                    labels[labels == i] = 0
            return labels
        _, a, b = best
        keep, drop = (a, b) if area[a] >= area[b] else (b, a)
        labels[labels == drop] = keep


def _relabel(labels) -> np.ndarray:
    """Consecutive room indices 1..N, ordered by position (row, col)."""
    out = np.zeros_like(labels)
    ids = [i for i in np.unique(labels) if i > 0]
    first = {i: np.argwhere(labels == i)[0].tolist() for i in ids}
    for n, i in enumerate(sorted(ids, key=lambda i: first[i]), 1):
        out[labels == i] = n
    return out
