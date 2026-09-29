"""Floor levels of a building (floor_environments.json).

Levels are found first and rooms are placed on levels, as in hierarchical
scene graphs (Hydra; HOV-SG: floors -> rooms -> objects). A region annotated
across several levels (stairwell halls, duplex rooms) is split per level:
each object belongs to the level of the highest floor surface at or below
its bottom (objects rest on the floor below them), within
``BELOW_FLOOR_TOLERANCE_M`` (half the NavMesh step height, as in the
membership rule). Such rooms are listed in every level they occupy, with the
object IDs of that level under ``partial_room_object_ids``.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

BELOW_FLOOR_TOLERANCE_M = 0.1


def level_of(bottom_z: float, levels: Sequence[tuple]) -> str:
    """levels: (floor_id, lowest floor-centre height), any order.
    Highest level whose floor is at or below the bottom; else the lowest."""
    ordered = sorted(levels, key=lambda item: item[1])
    chosen = ordered[0][0]
    for floor_id, height in ordered:
        if height <= bottom_z + BELOW_FLOOR_TOLERANCE_M:
            chosen = floor_id
    return chosen


def split_room(objects: Iterable[dict], levels: Sequence[tuple],
               floor_level: Dict[str, str]) -> Dict[str, List[str]]:
    """Object IDs per level; floor objects keep the level of their group."""
    out: Dict[str, List[str]] = {}
    for obj in objects:
        oid = obj["object_id"]
        level = floor_level.get(oid) or level_of(obj["bbox"]["min"][2], levels)
        out.setdefault(level, []).append(oid)
    return {k: sorted(v) for k, v in out.items()}


def on_floor(env: dict, obj: dict) -> bool:
    """Whether a graph object lies on this floor environment."""
    if obj["room_id"] not in env["room_ids"]:
        return False
    partial = env.get("partial_room_object_ids", {})
    return obj["room_id"] not in partial or obj["object_id"] in partial[obj["room_id"]]


def environment(floor_plan: dict, floor_id: str) -> dict:
    envs = [e for e in floor_plan["environments"] if e["floor_id"] == floor_id]
    if len(envs) != 1:
        raise ValueError(f"층 환경을 찾을 수 없습니다: {floor_id}")
    return envs[0]


def component_environment(floor_plan: dict, component: dict, room_ids: Iterable[str]) -> dict:
    """Floor environment of a navigable component: the recorded ``floor_id``
    (multi-island reports) or the one environment holding all its rooms."""
    if "floor_id" in component:
        return environment(floor_plan, component["floor_id"])
    rooms = set(room_ids)
    envs = [e for e in floor_plan["environments"] if rooms <= set(e["room_ids"])]
    if len(envs) != 1:
        raise ValueError(f"{component['component']}: 영역이 한 층에 속하지 않습니다.")
    return envs[0]


FLOOR_GAP_M = 1.0  # default level separation of prepare_floor_environments.py


def multi_level_floor_heights(graph: dict, gap: float = FLOOR_GAP_M) -> Dict[str, List[float]]:
    """Rooms whose floor objects are more than ``gap`` apart in height ->
    their floor-centre heights (single-level rooms are not listed)."""
    heights: Dict[str, List[float]] = {}
    for obj in graph["objects"]:
        if " ".join(obj["semantic_tag"].casefold().split()) == "floor" and obj.get("bbox"):
            heights.setdefault(obj["room_id"], []).append(float(obj["bbox"]["center"][2]))
    return {rid: sorted(zs) for rid, zs in heights.items() if max(zs) - min(zs) > gap}


def level_floor_height(bottom_z: float, heights: Sequence[float]) -> float:
    """Floor of the object's level: highest floor at or below its bottom."""
    return float(level_of(bottom_z, [(z, z) for z in heights]))
