"""Live detection test of one object: the operator puts it down and drives the
robot around it; every perception frame is judged with the search's found
check, per distance and viewing angle.

A frame that misses the object has no position of it, so the object's place is
taken from the frames that pass (their median, the anchor): from then on every
frame with the anchor inside the camera image counts, seen or missed. Until
the first pass there is nothing to measure by distance; the frame counts still
show that it was never found.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

DISTANCE_BINS = (0.0, 0.75, 1.0, 1.5, 2.0, 3.0, 99.0)     # camera to object [m]
ANGLE_BINS = (0.0, 10.0, 20.0, 30.0, 180.0)               # off the camera axis [deg]
SAME_PLACE_M = 0.5                                        # a detection this close is at the anchor
MIN_DEPTH_M = 0.3
RECENT_FRAMES = 30
CONFIRM_WINDOW = 10


def _bin_names(edges: Sequence[float], unit: str) -> List[str]:
    return [f"{a:g}~{b:g}{unit}" if b < 90 else f"{a:g}{unit}~" for a, b in zip(edges, edges[1:])]


class DetectionWatch:
    def __init__(self, target: str, K, image_size: Tuple[int, int], confirmations: int = 2):
        self.target = target
        self._K = np.asarray(K, float)
        self._size = image_size
        self._confirmations = confirmations
        self.frames = 0
        self.passed = 0
        self.elsewhere = 0
        self.best = 0.0
        self.current = None
        self._places = []
        self._recent = deque(maxlen=RECENT_FRAMES)
        self._labels = Counter()
        self._by_distance = [[0, 0] for _ in DISTANCE_BINS[1:]]
        self._by_angle = [[0, 0] for _ in ANGLE_BINS[1:]]

    def anchor(self) -> Optional[np.ndarray]:
        return np.median(self._places, axis=0) if self._places else None

    def _view(self, point, T_cam_map) -> Optional[Tuple[float, float]]:
        """(distance, angle off the camera axis) of a map point inside the
        image, else None."""
        p = T_cam_map[:3, :3] @ np.asarray(point, float) + T_cam_map[:3, 3]
        if p[2] < MIN_DEPTH_M:
            return None
        u, v, _ = self._K @ (p / p[2])
        if not (0 <= u < self._size[0] and 0 <= v < self._size[1]):
            return None
        return float(np.linalg.norm(p)), float(np.degrees(np.arctan2(np.hypot(p[0], p[1]), p[2])))

    def observe(self, objects: Sequence, explain: Callable, T_cam_map=None) -> None:
        """``objects`` of one perception frame; ``explain(target, obj)`` is
        athome.search.matching.explain with the search's matcher."""
        self.frames += 1
        here = self.anchor()
        hits, weak = [], []
        for o in objects:
            if not o.is_static:
                continue
            reason = explain(self.target, o)
            self.best = max(self.best, o.confidence if reason != "label" else 0.0)
            if reason == "ok":
                hits.append(o)
            elif reason != "label":
                weak.append((o, reason))
            if here is not None and np.linalg.norm(np.asarray(o.centroid[:2]) - here[:2]) < SAME_PLACE_M:
                self._labels[o.label] += 1
        on_object = [o for o in hits if here is None
                     or np.linalg.norm(np.asarray(o.centroid[:2]) - here[:2]) < SAME_PLACE_M]
        self.elsewhere += bool(hits) and not on_object
        self.passed += bool(on_object)
        self._recent.append(bool(on_object))
        best = max(on_object, key=lambda o: o.confidence, default=None)
        if best is not None:
            self._places.append(tuple(best.centroid[:3]))
            del self._places[:-200]
        self._count(on_object, T_cam_map)
        top = best or max((o for o, _ in weak), key=lambda o: o.confidence, default=None)
        reason = "ok" if best is not None else (next(r for o, r in weak if o is top) if top is not None else "none")
        view = None
        if T_cam_map is not None and (top is not None or here is not None):
            view = self._view(top.centroid if top is not None else here, T_cam_map)
        self.current = {"reason": reason, "label": top.label if top is not None else None,
                        "confidence": round(top.confidence, 3) if top is not None else None,
                        "distance_m": round(view[0], 2) if view else None,
                        "angle_deg": round(view[1], 1) if view else None}

    def _count(self, on_object, T_cam_map) -> None:
        here = self.anchor()
        if here is None or T_cam_map is None:
            return
        view = self._view(here, T_cam_map)
        if view is None:
            return
        for value, edges, table in ((view[0], DISTANCE_BINS, self._by_distance),
                                    (view[1], ANGLE_BINS, self._by_angle)):
            i = min(int(np.searchsorted(edges, value, side="right")) - 1, len(table) - 1)
            table[i][0] += 1
            table[i][1] += bool(on_object)

    def summary(self) -> dict:
        recent = list(self._recent)
        table = lambda names, rows: [{"bin": n, "frames": a, "passed": b} for n, (a, b) in zip(names, rows)]
        here = self.anchor()
        return {
            "target": self.target, "frames": self.frames, "passed": self.passed, "elsewhere": self.elsewhere,
            "best_confidence": round(self.best, 3),
            "recent_rate": round(sum(recent) / len(recent), 2) if recent else None,
            # What the found check needs: the same place in enough frames.
            "confirmed": sum(recent[-CONFIRM_WINDOW:]) >= self._confirmations,
            "current": self.current,
            "anchor": [round(float(v), 2) for v in here] if here is not None else None,
            "labels_at_object": self._labels.most_common(5),
            "by_distance": table(_bin_names(DISTANCE_BINS, " m"), self._by_distance),
            "by_angle": table(_bin_names(ANGLE_BINS, "°"), self._by_angle),
        }
