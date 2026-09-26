"""Robot configuration shared by the search system and the motion module.

A* and MPPI must use the same map and inflation, so both read this file.
Relative paths are resolved against the config file directory.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RobotConfig:
    map_yaml: Path
    map_version: str
    inflation_radius: float
    unknown_as_occupied: bool
    goal_offset: float
    # Robot center clearance is ``inflation_radius`` (Nav2 robot_radius);
    # costs decay out to ``costmap_inflation_radius`` (Nav2 inflation layer).
    costmap_inflation_radius: float
    # nav2: path from Nav2 planner_server, candidate costs on the Nav2
    # global costmap. none: goal only, internal grid (tests without Nav2).
    path_planner: str
    graph_path: Optional[Path]
    map_frame: str
    base_frame: str
    # Raw sections, built by athome.inference.factory.
    planner: dict = field(default_factory=dict)
    matcher: dict = field(default_factory=dict)
    command_parser: dict = field(default_factory=dict)


def map_version_of(map_yaml) -> str:
    """Hash of the map yaml and image, so every module can check it uses
    the same map."""
    import yaml

    map_yaml = Path(map_yaml)
    meta = yaml.safe_load(map_yaml.read_text())
    h = hashlib.sha256(map_yaml.read_bytes())
    h.update((map_yaml.parent / meta["image"]).read_bytes())
    return "sha256:" + h.hexdigest()[:16]


def load_robot_config(path) -> RobotConfig:
    import yaml

    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    base = path.parent

    def resolve(p):
        return None if p is None else (base / p).resolve()

    map_yaml = resolve(raw["map"]["yaml"])
    frames = raw.get("frames", {})
    nav = raw.get("navigation", {})
    if nav.get("path_planner", "nav2") not in ("nav2", "none"):
        raise ValueError(f"알 수 없는 path_planner: {nav.get('path_planner')}")
    return RobotConfig(
        map_yaml=map_yaml,
        map_version=map_version_of(map_yaml),
        inflation_radius=float(raw["robot"]["inflation_radius"]),
        unknown_as_occupied=bool(raw["map"].get("unknown_as_occupied", True)),
        goal_offset=float(nav.get("goal_offset", 0.45)),
        costmap_inflation_radius=float(nav.get(
            "costmap_inflation_radius", float(raw["robot"]["inflation_radius"]) + 0.25)),
        path_planner=nav.get("path_planner", "nav2"),
        graph_path=resolve(raw.get("scene_graph", {}).get("path")),
        map_frame=frames.get("map", "map"),
        base_frame=frames.get("base", "base_link"),
        planner=raw.get("planner") or {"type": "min_cost"},
        matcher=raw.get("matcher") or {"type": "label"},
        command_parser=raw.get("command_parser") or {},
    )
