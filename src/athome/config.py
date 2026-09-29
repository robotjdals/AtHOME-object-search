"""Robot configuration shared by the search system and the motion module.

A* and MPPI must use the same map and inflation, so both read this file.
Relative paths are resolved against the config file directory.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class RobotGeometry:
    """Rectangular footprint -> the radii Nav2 derives from a footprint.

    inscribed: center to the nearest edge; a 2D planner treats cells closer
    than this to an obstacle as certain collision (Nav2 INSCRIBED cost).
    circumscribed: center to a corner; clearance for any heading, needed where
    the robot rotates in place (360-degree observation at every goal).
    """
    footprint: Tuple[float, float]      # [m] length x width

    @property
    def inscribed_radius(self) -> float:
        return min(self.footprint) / 2

    @property
    def circumscribed_radius(self) -> float:
        return math.hypot(*self.footprint) / 2

    @property
    def rotation_clearance(self) -> float:
        return self.circumscribed_radius - self.inscribed_radius

    def polygon(self) -> List[List[float]]:
        """Nav2 ``footprint`` parameter (base_link centered)."""
        hx, hy = self.footprint[0] / 2, self.footprint[1] / 2
        return [[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]]


def robot_geometry(robot: dict) -> Tuple[RobotGeometry, float]:
    """``robot`` section of a robot config -> (geometry, inflation radius).

    ``inflation_radius`` (Nav2 robot_radius, robot center clearance) is the
    inscribed radius of ``footprint_m``; a config may state it, but it must
    agree with the footprint."""
    if "footprint_m" not in robot:
        return None, float(robot["inflation_radius"])
    geometry = RobotGeometry(tuple(float(v) for v in robot["footprint_m"]))
    if len(geometry.footprint) != 2 or min(geometry.footprint) <= 0:
        raise ValueError("footprint_m은 [길이, 폭] 양수여야 합니다.")
    radius = geometry.inscribed_radius
    if "inflation_radius" in robot and not math.isclose(float(robot["inflation_radius"]), radius,
                                                        abs_tol=1e-9):
        raise ValueError("inflation_radius가 footprint의 내접 반경과 다릅니다.")
    return geometry, radius


def traversability(robot: dict) -> Optional[Tuple[float, float]]:
    """(max step [m], max slope [deg]) of robot.max_step_m / max_slope_deg,
    or None when the config does not state them."""
    if "max_step_m" not in robot and "max_slope_deg" not in robot:
        return None
    step, slope = float(robot["max_step_m"]), float(robot["max_slope_deg"])
    if step <= 0 or not 0 < slope < 90:
        raise ValueError("max_step_m, max_slope_deg 값이 잘못됐습니다.")
    return step, slope


def goal_geometry(raw: dict) -> Tuple[float, float, Optional[float]]:
    """Robot config -> (goal offset, goal clearance, max goal offset), proposal 5-2.

    d_off = r_robot + d_safe with r_robot the circumscribed radius (the robot
    turns in place at the goal, so any side may face the object) when the
    config gives ``navigation.goal_safety_margin``; otherwise the stated
    ``navigation.goal_offset``. ``navigation.goal_max_offset`` lets the goal
    move out to that distance where the robot cannot stand at the offset
    (see athome.navigation.goal_poses.goal_candidates)."""
    geometry, _ = robot_geometry(raw["robot"])
    nav = raw.get("navigation", {})
    if "goal_safety_margin" in nav:
        if geometry is None:
            raise ValueError("goal_safety_margin에는 robot.footprint_m이 필요합니다.")
        if "goal_offset" in nav:
            raise ValueError("goal_offset과 goal_safety_margin 중 하나만 지정하세요.")
        offset = geometry.circumscribed_radius + float(nav["goal_safety_margin"])
    else:
        offset = float(nav.get("goal_offset", 0.45))
    max_offset = nav.get("goal_max_offset")
    if max_offset is not None and float(max_offset) <= offset:
        raise ValueError("goal_max_offset은 goal offset보다 커야 합니다.")
    return (offset, geometry.rotation_clearance if geometry else 0.0,
            None if max_offset is None else float(max_offset))


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
    # Rectangular footprint (None: circular robot of ``inflation_radius``).
    geometry: Optional[RobotGeometry] = None
    goal_max_offset: Optional[float] = None
    # search.time_budget_s (None: no time budget).
    time_budget_s: Optional[float] = None
    # search.observation_range_m: range for the observed part of each room
    # (None: not tracked; the planner then sees visit counts only).
    observation_range_m: Optional[float] = None
    # scene_graph.search_locations: the Search Location policy the planner was
    # trained with (None: DEFAULT_EXCLUDED_CATEGORIES).
    search_locations: Optional[Path] = None
    # scene_graph.scene_scope: storage scope rule of the training graphs
    # (objects inside storage furniture removed; configs/data/scene_scope.json).
    scene_scope: Optional[Path] = None
    # scene_graph.target_categories: grouping of raw labels into target
    # categories used by the training GT (athome.scene_graph.vocabulary).
    target_categories: Optional[Path] = None
    # map.floor_z_m: floor height in the map frame. Written to every room of
    # the scene graph so the 1.5 m search-height scope applies as in training.
    floor_z_m: Optional[float] = None
    # sensors.camera_hfov_deg: horizontal FOV of the camera used for detection;
    # sets the observation rotation yaw tolerance (observation_yaw_tolerance).
    camera_hfov_deg: Optional[float] = None

    @property
    def goal_clearance(self) -> float:
        """Extra goal-cell clearance for in-place rotation."""
        return self.geometry.rotation_clearance if self.geometry else 0.0


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
    geometry, inflation_radius = robot_geometry(raw["robot"])
    return RobotConfig(
        map_yaml=map_yaml,
        map_version=map_version_of(map_yaml),
        inflation_radius=inflation_radius,
        geometry=geometry,
        unknown_as_occupied=bool(raw["map"].get("unknown_as_occupied", True)),
        goal_offset=goal_geometry(raw)[0],
        goal_max_offset=goal_geometry(raw)[2],
        time_budget_s=(raw.get("search") or {}).get("time_budget_s"),
        observation_range_m=(raw.get("search") or {}).get("observation_range_m"),
        search_locations=resolve(raw.get("scene_graph", {}).get("search_locations")),
        scene_scope=resolve(raw.get("scene_graph", {}).get("scene_scope")),
        target_categories=resolve(raw.get("scene_graph", {}).get("target_categories")),
        floor_z_m=(None if raw["map"].get("floor_z_m") is None
                   else float(raw["map"]["floor_z_m"])),
        camera_hfov_deg=(None if (raw.get("sensors") or {}).get("camera_hfov_deg") is None
                         else float(raw["sensors"]["camera_hfov_deg"])),
        costmap_inflation_radius=float(nav.get(
            "costmap_inflation_radius", inflation_radius + 0.25)),
        path_planner=nav.get("path_planner", "nav2"),
        graph_path=resolve(raw.get("scene_graph", {}).get("path")),
        map_frame=frames.get("map", "map"),
        base_frame=frames.get("base", "base_link"),
        planner=raw.get("planner") or {"type": "min_cost"},
        matcher=raw.get("matcher") or {"type": "label"},
        command_parser=raw.get("command_parser") or {},
    )
