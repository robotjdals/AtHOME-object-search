"""Read-only access to a workspace graph for search planning.

Input is the ``workspace_graph`` JSON produced by ``build_workspace_graph``.
Search code depends on this module only, not on the storage format, so a
Spark-DSG backend can replace it later.
"""

from __future__ import annotations

import json
import math
from statistics import median
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# Not useful as places to search: structure and fixtures.
DEFAULT_EXCLUDED_CATEGORIES = frozenset({
    "wall", "floor", "ceiling", "staircase wall", "window", "door",
    "door frame", "light switch", "fire sprinkler", "air vent", "air duct",
    "ceiling light", "smoke detector", "outlet", "stairs",
})

WORKSPACE = "workspace"
STANDALONE = "standalone"
# Goal at a Known target object that is not itself a Search Location (e.g. a
# lamp: lighting is not a place to search). Used for Known targets only,
# never a candidate of the unknown search, so planner inputs are unchanged.
OBJECT = "object"


def normalize_category(tag: str) -> str:
    return " ".join(tag.casefold().split())


@dataclass(frozen=True)
class Room:
    room_id: str
    label: str


@dataclass(frozen=True)
class SearchLocation:
    location_id: str
    kind: str                       # WORKSPACE or STANDALONE
    room_id: str
    # Object whose bbox defines where the robot approaches.
    object_id: str
    category: str
    bbox_min: Tuple[float, float, float]
    bbox_max: Tuple[float, float, float]
    function_label: Optional[str] = None
    child_categories: Tuple[str, ...] = ()


# Objects whose bottom is higher than this above the room floor (ceiling
# lights, fans, wall-mounted fixtures, items on top of tall units) are out of
# the search scope: not standalone search locations, not Known-target matches,
# and not targets in HM3D GT (scripts/build_target_catalog.py).
MAX_BASE_ABOVE_FLOOR_M = 1.5


def above_search_height(obj: dict, floor_z: Dict[str, float],
                        limit: float = MAX_BASE_ABOVE_FLOOR_M) -> bool:
    """True if the object's bottom is more than ``limit`` above its room floor.
    Rooms without floor evidence are not height-filtered."""
    ref = floor_z.get(obj["room_id"])
    return ref is not None and obj["bbox"]["min"][2] - ref > limit


class SceneGraph:
    def __init__(
        self,
        graph: dict,
        excluded_categories: Iterable[str] = DEFAULT_EXCLUDED_CATEGORIES,
        # See MAX_BASE_ABOVE_FLOOR_M.
        max_standalone_base_height: float = MAX_BASE_ABOVE_FLOOR_M,
        room_floor_z: Optional[Dict[str, float]] = None,
        # False: Workspaces are the only Search Locations (proposal 4:
        # "Workspace 단위로 탐색 위치 판단"); standalone objects are seen from them.
        standalone_locations: bool = True,
        # Known-target lookup key of a category (athome.scene_graph.vocabulary:
        # raw names grouped into target categories as in the training GT).
        category_key: Callable[[str], str] = normalize_category,
        # Categories that count as a target (Vocabulary.target_keys: the target
        # and its subtypes, as in the training GT). None: the target's own key.
        target_keys: Optional[Callable[[str], Iterable[str]]] = None,
    ):
        if graph.get("coordinate_frame") != "athome_z_up":
            raise ValueError("athome_z_up 좌표계 그래프만 지원")
        excluded = {normalize_category(c) for c in excluded_categories}
        objects = {o["object_id"]: o for o in graph["objects"]}
        if len(objects) != len(graph["objects"]):
            raise ValueError("객체 ID 중복")

        # Explicit, fixed floor references are preferred for masked graphs.
        # Otherwise use semantic floor objects / room metadata, never the
        # lowest arbitrary object. No evidence means no height-based exclusion.
        if not math.isfinite(max_standalone_base_height) or max_standalone_base_height < 0:
            raise ValueError("Invalid standalone height limit")
        floor_z = dict(room_floor_z) if room_floor_z is not None else floor_references(graph)
        room_ids = {r["room_id"] for r in graph["rooms"]}
        if set(floor_z) - room_ids or any(not math.isfinite(z) for z in floor_z.values()):
            raise ValueError("Invalid room floor references")
        self.room_floor_z = floor_z

        self.rooms: Dict[str, Room] = {
            r["room_id"]: Room(r["room_id"], r["room_label"])
            for r in graph["rooms"]
        }
        self.locations: Dict[str, SearchLocation] = {}
        # Object ID -> search location that covers it.
        self._object_location: Dict[str, str] = {}
        self._objects_by_category: Dict[str, List[str]] = {}

        for ws in graph["workspaces"]:
            source = objects[ws["source_object_id"]]
            children = [objects[c] for c in ws["child_object_ids"]]
            loc = SearchLocation(
                location_id=ws["workspace_id"],
                kind=WORKSPACE,
                room_id=ws["room_id"],
                object_id=source["object_id"],
                category=normalize_category(source["semantic_tag"]),
                bbox_min=tuple(source["bbox"]["min"]),
                bbox_max=tuple(source["bbox"]["max"]),
                function_label=ws.get("function_label"),
                # One entry per child object (repeats kept): the prompt shows
                # counts, as MoMa-LLM's "2 balls".
                child_categories=tuple(sorted(
                    normalize_category(c["semantic_tag"]) for c in children
                )),
            )
            self.locations[loc.location_id] = loc
            self._object_location[source["object_id"]] = loc.location_id
            for c in children:
                self._object_location[c["object_id"]] = loc.location_id

        self._category_key = category_key
        self._target_keys = target_keys
        for obj in graph["objects"]:
            category = normalize_category(obj["semantic_tag"])
            high = above_search_height(obj, floor_z, max_standalone_base_height)
            if not high:  # out-of-scope objects never make a target Known
                self._objects_by_category.setdefault(
                    category_key(obj["semantic_tag"]), []).append(obj["object_id"])
            if (not standalone_locations or obj["role"] != "standalone"
                    or category in excluded or high):
                continue
            loc = SearchLocation(
                location_id=f"standalone:{obj['object_id']}",
                kind=STANDALONE,
                room_id=obj["room_id"],
                object_id=obj["object_id"],
                category=category,
                bbox_min=tuple(obj["bbox"]["min"]),
                bbox_max=tuple(obj["bbox"]["max"]),
            )
            self.locations[loc.location_id] = loc
            self._object_location[obj["object_id"]] = loc.location_id

        # Known targets go to their object: its Search Location if it has one,
        # otherwise an object goal (not part of the unknown search).
        self.object_goals: Dict[str, SearchLocation] = {}
        for oid in (o for ids in self._objects_by_category.values() for o in ids):
            if oid in self._object_location:
                continue
            obj = objects[oid]
            goal = SearchLocation(
                location_id=f"object:{oid}", kind=OBJECT, room_id=obj["room_id"], object_id=oid,
                category=normalize_category(obj["semantic_tag"]),
                bbox_min=tuple(obj["bbox"]["min"]), bbox_max=tuple(obj["bbox"]["max"]))
            self.object_goals[goal.location_id] = goal

    def goal(self, location_id: str) -> SearchLocation:
        """Search Location or Known object goal."""
        return self.locations.get(location_id) or self.object_goals[location_id]

    @classmethod
    def load(cls, path, **kwargs) -> "SceneGraph":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), **kwargs)

    def room_locations(self, room_id: str, kind: Optional[str] = None):
        return [
            l for l in self.locations.values()
            if l.room_id == room_id and (kind is None or l.kind == kind)
        ]

    def locations_defined_by(self, object_id: Optional[str]) -> List[str]:
        """Search Locations approached through this object (Workspace source
        or standalone object)."""
        return [lid for lid, loc in self.locations.items() if loc.object_id == object_id]

    def known_locations(self, category: str) -> List[str]:
        """Search locations covering objects of ``category`` or its subtypes
        (deduplicated)."""
        keys = (self._target_keys(category) if self._target_keys is not None
                else (self._category_key(category),))
        ids = [oid for key in keys for oid in self._objects_by_category.get(key, [])]
        out = []
        for oid in ids:
            lid = self._object_location.get(oid, f"object:{oid}")
            if lid not in out:
                out.append(lid)
        return out


def floor_references(graph: dict) -> Dict[str, float]:
    """Room reference = median semantic floor AABB center Z.

    These are reference heights, not exact support surfaces. A reviewed
    room.floor_z_m overrides this reduction for multi-level rooms.
    """
    values = {}
    for obj in graph["objects"]:
        if normalize_category(obj["semantic_tag"]) == "floor":
            box = obj["bbox"]
            z = (float(box["min"][2]) + float(box["max"][2])) / 2
            if not math.isfinite(z):
                raise ValueError("Nonfinite floor height")
            values.setdefault(obj["room_id"], []).append(z)
    result = {rid: float(median(zs)) for rid, zs in values.items()}
    for room in graph["rooms"]:
        if "floor_z_m" in room:
            z = float(room["floor_z_m"])
            if not math.isfinite(z):
                raise ValueError("Nonfinite floor height")
            result[room["room_id"]] = z
    return result
