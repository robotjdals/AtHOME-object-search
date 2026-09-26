"""Assign objects to segmented rooms (proposal 4-1).

Order: room label at the XY centroid -> nearest room label within
``search_radius`` beyond the object footprint (centroid on an occupied or
unknown cell, e.g. furniture) -> majority label of the projected object
points -> unassigned.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from athome.scene_graph.room_segmentation import RoomSegmentation
from athome.scene_graph.static_features import StaticObject


@dataclass
class RoomAssignment:
    rooms: Dict[str, str]              # object_id -> room_id
    methods: Dict[str, str]            # object_id -> centroid / nearest / points
    unassigned: List[str]


def _cell(seg: RoomSegmentation, x: float, y: float):
    return (
        int(np.floor((y - seg.origin[1]) / seg.resolution)),
        int(np.floor((x - seg.origin[0]) / seg.resolution)),
    )


def _label_at(seg: RoomSegmentation, x: float, y: float) -> int:
    r, c = _cell(seg, x, y)
    h, w = seg.labels.shape
    return int(seg.labels[r, c]) if 0 <= r < h and 0 <= c < w else 0


def _nearest_label(seg: RoomSegmentation, x: float, y: float, radius: float) -> int:
    r0, c0 = _cell(seg, x, y)
    k = int(np.ceil(radius / seg.resolution))
    h, w = seg.labels.shape
    rs, cs = slice(max(r0 - k, 0), min(r0 + k + 1, h)), slice(max(c0 - k, 0), min(c0 + k + 1, w))
    window = seg.labels[rs, cs]
    rows, cols = np.nonzero(window)
    if rows.size == 0:
        return 0
    d = np.hypot(rows + rs.start - r0, cols + cs.start - c0) * seg.resolution
    i = int(np.argmin(d))
    return int(window[rows[i], cols[i]]) if d[i] <= radius else 0


def assign_rooms(
    objects: Iterable[StaticObject],
    seg: RoomSegmentation,
    points: Callable[[StaticObject], Optional[np.ndarray]] = lambda o: None,
    search_radius: float = 0.5,
) -> RoomAssignment:
    rooms, methods, unassigned = {}, {}, []
    for obj in objects:
        x, y = obj.centroid[0], obj.centroid[1]
        label, method = _label_at(seg, x, y), "centroid"
        if label == 0:
            # Large furniture is itself occupied: search beyond its footprint.
            half = max(obj.bbox_max[0] - obj.bbox_min[0],
                       obj.bbox_max[1] - obj.bbox_min[1]) / 2
            label, method = _nearest_label(seg, x, y, search_radius + half), "nearest"
        if label == 0:
            pts = points(obj)
            if pts is not None and len(pts):
                votes = Counter(_label_at(seg, px, py) for px, py in pts[:, :2])
                votes.pop(0, None)
                if votes:
                    label, method = votes.most_common(1)[0][0], "points"
        if label == 0:
            unassigned.append(obj.object_id)
            continue
        rooms[obj.object_id] = f"room_{label}"
        methods[obj.object_id] = method
    return RoomAssignment(rooms, methods, unassigned)
