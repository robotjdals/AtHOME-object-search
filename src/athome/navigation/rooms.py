"""Which room a map point is in, from the room segmentation of the graph
(<graph>.rooms.npz of scripts/build_scene_graph.py)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class RoomMap:
    labels: np.ndarray                      # 0: no room; row = +y, col = +x
    origin: Tuple[float, float]
    resolution: float
    room_ids: Optional[Sequence[str]] = None    # room_ids[i - 1] names label i; None: room_<i>

    @classmethod
    def load(cls, path) -> "RoomMap":
        with np.load(Path(path), allow_pickle=False) as saved:
            return cls(saved["labels"], tuple(float(v) for v in saved["origin_xy_m"]),
                       float(saved["resolution_m"].item()),
                       [str(v) for v in saved["room_ids"]] if "room_ids" in saved.files else None)

    def room_at(self, x: float, y: float) -> Optional[str]:
        row = int(np.floor((y - self.origin[1]) / self.resolution))
        col = int(np.floor((x - self.origin[0]) / self.resolution))
        if not (0 <= row < self.labels.shape[0] and 0 <= col < self.labels.shape[1]):
            return None
        label = int(self.labels[row, col])
        if label <= 0:
            return None
        return f"room_{label}" if self.room_ids is None else self.room_ids[label - 1]
