"""Online object layer of a pre-built scene graph.

Rooms and the Workspaces of the pre-map stay as built offline (proposal 4).
Objects are assumed to stay where they are; what changes is what the robot
has seen: objects the pre-map does not have are added from live
observations, with the rules of the offline build so the planner sees the
same kind of graph it was trained on: room label map (proposal 4-1), the
storage scope of the training graphs, Workspace child geometry and
tie-breaks (proposal 4-3). Room segmentation and the LLM labeling are not
rerun, so existing room and Workspace IDs never change; a new object keeps
its ID for the whole run.

New furniture becomes a Workspace when the training labels made its
category a Workspace source in most of its occurrences
(configs/data/workspace_source_priors.json, scripts/build_workspace_source_priors.py),
with that category's most frequent function label. Objects of its room that
now lie on it move under it by the same child rule.

Association of a static observation, in order:
1. the pre-map object perception linked it to (ObservedObject.object_id,
   perception's static association to the pre-mapped objects.json), if it is
   within ``link_radius_m`` of that object;
2. the nearest graph object of the same target category whose bbox center
   is within ``associate_radius_m``, or whose box contains the observation's
   center or is centered inside the observation's box (Hydra merges object
   nodes on this containment test; it keeps a partly seen piece of furniture
   from becoming a second object);
3. a candidate, added to the graph once seen in ``min_frames`` frames, so
   one false detection does not enter the map.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from athome.data.hm3d.scope import find_contained
from athome.scene_graph.geometric_relations import pair_geometry
from athome.scene_graph.object_rooms import assign_rooms
from athome.scene_graph.query import normalize_category
from athome.scene_graph.room_segmentation import RoomSegmentation
from athome.scene_graph.static_features import StaticObject, oriented_to_aabb
from athome.schemas import ObservationFrame, ObservedObject

# Never Workspace children (workspace_builder: architectural surfaces stay
# under the room).
STRUCTURAL = {"wall", "floor", "ceiling", "staircase wall"}
# Pre-map object IDs are "<label>_<perception id>" (static_features).
_PRIOR_ID = re.compile(r"(.+)_(\d+)")

Box = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


@dataclass(frozen=True)
class OnlineConfig:
    associate_radius_m: float = 0.5
    link_radius_m: float = 1.0
    min_frames: int = 2
    # Candidates not confirmed within this time are dropped.
    candidate_timeout_s: float = 60.0
    # New furniture is a Workspace when the training labels chose its
    # category as a source more often than not, in at least this many cases.
    source_min_rate: float = 0.5
    source_min_count: int = 20
    # Online objects: a box update that moves the center more than this is
    # reported (and the graph rebuilt).
    move_report_m: float = 0.2
    # Same Workspace child thresholds as the offline build (BuildConfig).
    distance_xy: float = 0.5
    delta_z: float = 0.1
    overlap_xy: float = 0.5

    def __post_init__(self):
        if self.min_frames < 1:
            raise ValueError("min_frames는 1 이상")
        if not 0.0 <= self.source_min_rate < 1.0:
            raise ValueError("source_min_rate는 0 이상 1 미만")

    @classmethod
    def from_dict(cls, raw: dict) -> "OnlineConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known - {"enabled", "output", "source_priors"}
        if unknown:
            raise ValueError(f"scene_graph.online_update에 모르는 키: {sorted(unknown)}")
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass(eq=False)
class _Candidate:
    label: str
    key: str
    lo: np.ndarray
    hi: np.ndarray
    stamps: set = field(default_factory=set)
    last: float = 0.0


def observation_box(obj: ObservedObject) -> Box:
    """Axis-aligned bbox of an observation (its centroid when it has no box)."""
    if obj.bbox_center is None or obj.bbox_size is None:
        c = tuple(float(v) for v in obj.centroid)
        return c, c
    return oriented_to_aabb(obj.bbox_center, obj.bbox_size, obj.bbox_yaw)


def _box_dict(lo, hi) -> dict:
    lo, hi = [float(v) for v in lo], [float(v) for v in hi]
    return {"min": lo, "max": hi, "size": [b - a for a, b in zip(lo, hi)],
            "center": [(a + b) / 2 for a, b in zip(lo, hi)]}


def _center(box: dict) -> np.ndarray:
    return (np.asarray(box["min"], float) + np.asarray(box["max"], float)) / 2


class OnlineSceneGraph:
    """``graph``: workspace graph JSON (build_scene_graph / toy_graph).
    ``rooms``: its room segmentation (label i = ``room_<i>``, or
    ``room_names[i - 1]``). ``outside``: cells beyond the mapped rooms on the
    same grid (objects there are not assigned). ``category_key``: target
    category of a label (Vocabulary.canonical). ``scope``: storage scope rule
    of the training graphs (None: not applied). ``source_priors``:
    workspace_source_priors.json (None: new furniture stays standalone)."""

    def __init__(self, graph: dict, rooms: RoomSegmentation,
                 config: OnlineConfig = OnlineConfig(),
                 category_key: Callable[[str], str] = normalize_category,
                 scope: Optional[dict] = None, outside: Optional[np.ndarray] = None,
                 room_names: Optional[Sequence[str]] = None,
                 source_priors: Optional[dict] = None):
        self.config = config
        self._graph = copy.deepcopy(graph)
        self._rooms = rooms
        self._room_names = list(room_names) if room_names is not None else None
        self._outside = outside
        self._key = category_key
        self._scope = scope
        priors = (source_priors or {}).get("categories", {})
        self._sources = {c: v["function_label"] for c, v in priors.items()
                         if v["present"] >= config.source_min_count and v["rate"] > config.source_min_rate}
        policy = self._graph.get("association_policy") or {}
        self._no_child = STRUCTURAL | {normalize_category(c) for c in policy.get("child_excluded_categories", ())}

        self._objects: Dict[str, dict] = {o["object_id"]: o for o in self._graph["objects"]}
        self._workspaces: Dict[str, dict] = {w["workspace_id"]: w for w in self._graph["workspaces"]}
        self._room_records: Dict[str, dict] = {r["room_id"]: r for r in self._graph["rooms"]}
        self._prior_ids: Dict[int, str] = {}
        for oid in self._objects:
            m = _PRIOR_ID.fullmatch(oid)
            if m:
                self._prior_ids[int(m.group(2))] = oid
        self._online: Dict[str, int] = {}        # online object ID -> frames merged into its box
        self._candidates: List[_Candidate] = []
        # Rejected objects (outside the rooms of the graph, inside storage):
        # later observations there are ignored instead of rejected again.
        self._ignored: List[Tuple[str, np.ndarray]] = []
        self._next = 0
        self.added: List[str] = []

    # --- public ---------------------------------------------------------

    def graph(self) -> dict:
        """Current graph (pre-map rooms and Workspaces, objects added online)."""
        out = copy.deepcopy(self._graph)
        out.setdefault("provenance", {})["online_update"] = {
            "added_objects": list(self.added),
            "added_workspaces": [w for w, r in self._workspaces.items() if r.get("online")]}
        return out

    def observe(self, frame: ObservationFrame) -> List[dict]:
        """Apply one perception frame. Returns the changes of the graph."""
        changes: List[dict] = []
        for obj in frame.objects:
            if not obj.is_static:
                continue
            lo, hi = (np.asarray(v, float) for v in observation_box(obj))
            oid = self._associate(obj, lo, hi)
            if oid is not None:
                if oid in self._online:
                    changes += self._refine(oid, lo, hi)
                continue
            if self._is_ignored(obj, lo, hi):
                continue
            new = self._candidate(obj, lo, hi, frame.stamp)
            if new is not None:
                changes += self._add(new)
        self._candidates = [c for c in self._candidates
                            if frame.stamp - c.last <= self.config.candidate_timeout_s]
        return changes

    # --- association ------------------------------------------------------

    def _associate(self, obj: ObservedObject, lo, hi) -> Optional[str]:
        center = (lo + hi) / 2
        if obj.object_id >= 0:
            oid = self._prior_ids.get(obj.object_id)
            if oid is not None and oid in self._objects and float(
                    np.linalg.norm(_center(self._objects[oid]["bbox"]) - center)) <= self.config.link_radius_m:
                return oid
        key = self._key(obj.label)
        best, best_d = None, math.inf
        for oid, o in self._objects.items():
            if self._key(o["semantic_tag"]) != key:
                continue
            lo2, hi2 = np.asarray(o["bbox"]["min"], float), np.asarray(o["bbox"]["max"], float)
            d = self._same(center, lo, hi, lo2, hi2)
            if d is not None and d < best_d:
                best, best_d = oid, d
        return best

    def _same(self, center, lo, hi, lo2, hi2) -> Optional[float]:
        """Center distance if two boxes are the same object (centers within the
        association radius, or one box contains the other's center), else None."""
        center2 = (lo2 + hi2) / 2
        d = float(np.linalg.norm(center2 - center))
        inside = bool(np.all(lo2 <= center) and np.all(center <= hi2)
                      or np.all(lo <= center2) and np.all(center2 <= hi))
        return d if d <= self.config.associate_radius_m or inside else None

    def _is_ignored(self, obj, lo, hi) -> bool:
        key, center = self._key(obj.label), (lo + hi) / 2
        return any(k == key and float(np.linalg.norm(c - center)) <= self.config.associate_radius_m
                   for k, c in self._ignored)

    def _candidate(self, obj, lo, hi, stamp) -> Optional[_Candidate]:
        key = self._key(obj.label)
        center = (lo + hi) / 2
        match = None
        for c in self._candidates:
            if c.key == key and self._same(center, lo, hi, c.lo, c.hi) is not None:
                match = c
                break
        if match is None:
            match = _Candidate(" ".join(obj.label.split()), key, lo.copy(), hi.copy())
            self._candidates.append(match)
        elif stamp not in match.stamps:
            n = len(match.stamps)
            match.lo = (match.lo * n + lo) / (n + 1)
            match.hi = (match.hi * n + hi) / (n + 1)
        match.stamps.add(stamp)
        match.last = max(match.last, stamp)
        if len(match.stamps) < self.config.min_frames:
            return None
        self._candidates = [c for c in self._candidates if c is not match]
        return match

    # --- graph edits ------------------------------------------------------

    def _add(self, c: _Candidate) -> List[dict]:
        self._next += 1
        oid = f"{c.label}_online{self._next}"
        record = {"object_id": oid, "semantic_tag": c.label, "bbox": _box_dict(c.lo, c.hi),
                  "online": True}
        rid, reason = self._room_of(record)
        if reason is None:
            record["room_id"] = rid
            reason = self._out_of_scope(record)
        if reason is not None:
            self._ignored.append((c.key, (c.lo + c.hi) / 2))
            return [{"event": "object_rejected", "object_id": oid, "label": c.label, "reason": reason,
                     "center": [round(float(v), 3) for v in record["bbox"]["center"]]}]
        self._graph["objects"].append(record)
        self._objects[oid] = record
        self._online[oid] = len(c.stamps)
        self.added.append(oid)
        function = self._sources.get(normalize_category(c.label))
        if function is not None:
            return [self._make_workspace(record, function), *self._rehome(rid)]
        self._link(record)
        return [self._event("object_added", oid)]

    def _refine(self, oid: str, lo, hi) -> List[dict]:
        """Running mean of an online object's box (at most the last ~10 frames
        weigh in). A Workspace source keeps its room; its room's objects are
        re-homed."""
        o = self._objects[oid]
        n = min(self._online[oid], 9)
        old_center = _center(o["bbox"])
        new_box = _box_dict((np.asarray(o["bbox"]["min"]) * n + lo) / (n + 1),
                            (np.asarray(o["bbox"]["max"]) * n + hi) / (n + 1))
        self._online[oid] += 1
        moved = float(np.linalg.norm(_center(new_box) - old_center)) > self.config.move_report_m
        if o["role"] == "source":
            o["bbox"] = new_box
            w = self._workspaces[o["workspace_id"]]
            w["position"] = [new_box["center"][0], new_box["center"][1], new_box["max"][2]]
            return ([self._event("object_moved", oid)] if moved else []) + self._rehome(o["room_id"])
        before = (o["room_id"], o.get("parent_id"))
        self._unlink(o)
        o["bbox"] = new_box
        rid, reason = self._room_of(o)
        if reason is None:
            o["room_id"] = rid
            reason = self._out_of_scope(o)
        if reason is not None:            # the better estimate is out of the graph's rooms
            self._drop(oid)
            return [{"event": "object_rejected", "object_id": oid, "label": o["semantic_tag"],
                     "reason": reason, "center": [round(float(v), 3) for v in new_box["center"]]}]
        self._link(o)
        if (o["room_id"], o.get("parent_id")) != before or moved:
            return [self._event("object_moved", oid)]
        return []

    def _room_of(self, record: dict) -> Tuple[Optional[str], Optional[str]]:
        lo, hi = record["bbox"]["min"], record["bbox"]["max"]
        static = StaticObject(object_id=record["object_id"], source_id=-1, label=record["semantic_tag"],
                              confidence=1.0, n_observations=1,
                              centroid=tuple(record["bbox"]["center"]),
                              bbox_min=tuple(lo), bbox_max=tuple(hi), row=-1)
        rid = assign_rooms([static], self._rooms, outside=self._outside).rooms.get(record["object_id"])
        if rid is None:
            return None, "outside_rooms"
        if self._room_names is not None:
            rid = self._room_names[int(rid.rsplit("_", 1)[1]) - 1]
        if rid not in self._room_records:
            # Rooms without objects were left out of the graph and have no
            # label (labeling is not rerun online).
            return None, "room_not_in_graph"
        return rid, None

    def _out_of_scope(self, record: dict) -> Optional[str]:
        if self._scope is None:
            return None
        same_room = [o for o in self._objects.values()
                     if o["room_id"] == record["room_id"] and o["object_id"] != record["object_id"]]
        if record["object_id"] in find_contained(same_room + [record], self._scope):
            return "inside_storage_container"
        return None

    def _best_workspace(self, record: dict) -> Optional[str]:
        """Workspace an object lies on (offline child rule and tie-break)."""
        if normalize_category(record["semantic_tag"]) in self._no_child:
            return None
        best = None
        for wid, w in self._workspaces.items():
            if w["room_id"] != record["room_id"] or w["source_object_id"] == record["object_id"]:
                continue
            g = pair_geometry(record["bbox"], self._objects[w["source_object_id"]]["bbox"])
            if (g["overlap_xy"] is not None and g["distance_xy"] <= self.config.distance_xy
                    and g["delta_z"] < self.config.delta_z and g["overlap_xy"] > self.config.overlap_xy):
                key = (-g["overlap_xy"], g["delta_z"], w["source_object_id"])
                if best is None or key < best[0]:
                    best = (key, wid)
        return None if best is None else best[1]

    def _link(self, record: dict) -> None:
        """Child of the Workspace it lies on, else a standalone object of its room."""
        rid, wid = record["room_id"], self._best_workspace(record)
        if wid is not None:
            record.update(role="child", workspace_id=wid, parent_id=wid, parent_type="workspace")
            children = self._workspaces[wid]["child_object_ids"]
            children.append(record["object_id"])
            children.sort()
            return
        record.update(role="standalone", workspace_id=None, parent_id=rid, parent_type="room")
        room = self._room_records[rid]
        for name in ("standalone_object_ids", "direct_object_ids"):
            if name in room:
                room[name] = sorted(room[name] + [record["object_id"]])

    def _make_workspace(self, record: dict, function: str) -> dict:
        rid, oid = record["room_id"], record["object_id"]
        wid = f"workspace:{rid}:{oid}"
        box = record["bbox"]
        self._workspaces[wid] = workspace = {
            "workspace_id": wid, "room_id": rid, "source_object_id": oid, "function_label": function,
            "position": [box["center"][0], box["center"][1], box["max"][2]], "child_object_ids": [],
            "online": True}
        self._graph["workspaces"].append(workspace)
        record.update(role="source", workspace_id=wid, parent_id=rid, parent_type="room")
        room = self._room_records[rid]
        for name in ("workspace_ids", "source_object_ids", "direct_object_ids"):
            if name in room:
                room[name] = sorted(room[name] + [wid if name == "workspace_ids" else oid])
        event = self._event("workspace_added", oid)
        event.update(workspace_id=wid, function_label=function)
        return event

    def _rehome(self, room_id: str) -> List[dict]:
        """After a Workspace was added or its box moved: standalone objects and
        children of online Workspaces in its room go under the one they now
        lie on. Children of pre-map Workspaces keep the offline decision."""
        changes = []
        for o in [o for o in self._objects.values()
                  if o["room_id"] == room_id and o.get("role") in ("child", "standalone")]:
            current = next((wid for wid, w in self._workspaces.items()
                            if o["object_id"] in w["child_object_ids"]), None)
            if current is not None and not self._workspaces[current].get("online"):
                continue
            if self._best_workspace(o) == current:
                continue
            self._unlink(o)
            self._link(o)
            changes.append(self._event("object_moved", o["object_id"]))
        return changes

    def _unlink(self, o: dict) -> None:
        """Remove an object's links (Workspace child list, room lists)."""
        oid = o["object_id"]
        if o.get("role") == "child":
            for w in self._workspaces.values():
                if oid in w["child_object_ids"]:
                    w["child_object_ids"].remove(oid)
        room = self._room_records.get(o.get("room_id"))
        if room is not None:
            for name in ("standalone_object_ids", "direct_object_ids"):
                if name in room and oid in room[name]:
                    room[name] = [i for i in room[name] if i != oid]

    def _drop(self, oid: str) -> None:
        self._objects.pop(oid)
        self._graph["objects"] = [x for x in self._graph["objects"] if x["object_id"] != oid]
        self._online.pop(oid, None)
        self.added.remove(oid)

    # --- events -------------------------------------------------------------

    def _event(self, name: str, oid: str) -> dict:
        o = self._objects[oid]
        return {"event": name, "object_id": oid, "label": o["semantic_tag"], "room_id": o["room_id"],
                "role": o["role"], "parent_id": o.get("parent_id"),
                "center": [round(float(v), 3) for v in _center(o["bbox"])]}
