"""Read-only access to a workspace graph for search planning.

Input is the ``workspace_graph`` JSON produced by ``build_workspace_graph``.
Search code depends on this module only, not on the storage format, so a
Spark-DSG backend can replace it later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Not useful as places to search: structure and fixtures.
DEFAULT_EXCLUDED_CATEGORIES = frozenset({
    "wall", "floor", "ceiling", "staircase wall", "window", "door",
    "door frame", "light switch", "fire sprinkler", "air vent", "air duct",
    "ceiling light", "smoke detector", "outlet", "stairs",
})

WORKSPACE = "workspace"
STANDALONE = "standalone"


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


class SceneGraph:
    def __init__(
        self,
        graph: dict,
        excluded_categories: Iterable[str] = DEFAULT_EXCLUDED_CATEGORIES,
        # Standalone objects whose bottom is higher than this above the room
        # floor (ceiling fans, wall-mounted fixtures) are not search locations.
        max_standalone_base_height: float = 1.5,
    ):
        if graph.get("coordinate_frame") != "athome_z_up":
            raise ValueError("athome_z_up 좌표계 그래프만 지원")
        excluded = {normalize_category(c) for c in excluded_categories}
        objects = {o["object_id"]: o for o in graph["objects"]}
        if len(objects) != len(graph["objects"]):
            raise ValueError("객체 ID 중복")

        # Room floor height: lowest object bottom in the room.
        floor_z: Dict[str, float] = {}
        for o in graph["objects"]:
            z = o["bbox"]["min"][2]
            floor_z[o["room_id"]] = min(z, floor_z.get(o["room_id"], z))

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
                child_categories=tuple(sorted({
                    normalize_category(c["semantic_tag"]) for c in children
                })),
            )
            self.locations[loc.location_id] = loc
            self._object_location[source["object_id"]] = loc.location_id
            for c in children:
                self._object_location[c["object_id"]] = loc.location_id

        for obj in graph["objects"]:
            category = normalize_category(obj["semantic_tag"])
            self._objects_by_category.setdefault(category, []).append(
                obj["object_id"])
            if obj["role"] != "standalone" or category in excluded:
                continue
            base = obj["bbox"]["min"][2] - floor_z[obj["room_id"]]
            if base > max_standalone_base_height:
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

    @classmethod
    def load(cls, path, **kwargs) -> "SceneGraph":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), **kwargs)

    def room_locations(self, room_id: str, kind: Optional[str] = None):
        return [
            l for l in self.locations.values()
            if l.room_id == room_id and (kind is None or l.kind == kind)
        ]

    def known_locations(self, category: str) -> List[str]:
        """Search locations covering objects of ``category`` (deduplicated)."""
        ids = self._objects_by_category.get(normalize_category(category), [])
        out = []
        for oid in ids:
            lid = self._object_location.get(oid)
            if lid is not None and lid not in out:
                out.append(lid)
        return out
