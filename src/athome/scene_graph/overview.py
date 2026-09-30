"""A scene graph drawn on its map, to check by eye (one PNG).

What a person checks on it and tests cannot: rooms and Search Locations sit
on the right furniture of the real map (frames, floor height), and the goal
poses are free spots in front of the furniture, facing it.

Goal poses come from the search server's own code
(``NavigationPlanner.add_location``) on the map inflated by the robot radius.
On the robot with ``path_planner: nav2`` the server runs the same code on the
Nav2 global costmap, whose inscribed area has the same radius.

Locations the robot cannot visit are flagged: no goal pose (e.g. furniture
boxed in), or goal poses only outside the largest free area (e.g. a doorway
narrower than the robot).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from athome.navigation import (
    GridMap,
    NavigationConfig,
    NavigationPlanner,
    OccupancyMap,
    load_map_server,
    load_occupancy,
)
from athome.schemas import Pose2D
from athome.scene_graph.location_policy import excluded_categories
from athome.scene_graph.query import WORKSPACE, SceneGraph

NO_GOAL = "no_goal"
DISCONNECTED = "disconnected"
PROBLEM_TEXT = {
    NO_GOAL: "목표 자세 없음 (주변에 로봇이 설 빈 공간 없음)",
    DISCONNECTED: "목표 자세가 가장 넓은 주행 영역과 이어지지 않음 (좁은 문, 지도 구멍 확인)",
}

# Map cells (RGB).
FREE = (1.0, 1.0, 1.0)
NEAR_OBSTACLE = (1.0, 0.86, 0.86)     # free on the map, too close for the robot center
UNKNOWN = (0.78, 0.78, 0.78)
OCCUPIED = (0.0, 0.0, 0.0)
WORKSPACE_COLOR = "tab:blue"
STANDALONE_COLOR = "tab:green"
GOAL_COLOR = "tab:orange"
PROBLEM_COLOR = "red"
# Distinct from each other and from the goal/problem/map colors.
ROOM_COLORS = ("tab:blue", "tab:green", "tab:purple", "tab:cyan", "tab:olive",
               "tab:pink", "tab:brown")
ARROW_LENGTH_M = 0.3
EXAMPLE_GOALS = 3                     # arrows per location, spread around it


@dataclass(frozen=True)
class LocationCheck:
    location_id: str
    category: str
    room_id: str
    goals: Tuple[Pose2D, ...]         # every goal candidate
    problem: Optional[str] = None     # None, NO_GOAL or DISCONNECTED


@dataclass(frozen=True)
class RoomMap:
    """Room segmentation saved by build_scene_graph.py (``room_<i>`` =
    label i, as in athome.search.coverage), row 0 = lowest y."""
    labels: np.ndarray
    origin: Tuple[float, float]
    resolution: float

    @classmethod
    def load(cls, path) -> "RoomMap":
        with np.load(path, allow_pickle=False) as saved:
            origin = tuple(float(v) for v in saved["origin_xy_m"])
            return cls(saved["labels"], origin, float(saved["resolution_m"].item()))


def check_locations(graph: SceneGraph, grid: GridMap,
                    config: NavigationConfig) -> Dict[str, LocationCheck]:
    """Goal candidates of every Search Location and why it cannot be visited."""
    planner = NavigationPlanner(grid, config)
    # The planner's 8-connected moves without corner cutting
    # (athome.navigation.path_cost) connect exactly the 4-connected cells.
    components, count = ndimage.label(grid.free)
    largest = int(np.argmax(np.bincount(components.ravel())[1:])) + 1 if count else 0
    out = {}
    for lid, loc in sorted(graph.locations.items()):
        planner.add_location(lid, loc.bbox_min, loc.bbox_max)
        candidates = planner.candidate_goals(lid)
        problem = None
        if not candidates:
            problem = NO_GOAL
        elif all(components[cell] != largest for cell, _ in candidates):
            problem = DISCONNECTED
        out[lid] = LocationCheck(lid, loc.category, loc.room_id,
                                 tuple(pose for _, pose in candidates), problem)
    return out


def describe_problems(checks: Dict[str, LocationCheck]) -> List[str]:
    bad = [c for c in checks.values() if c.problem]
    if not bad:
        return [f"탐색 위치 {len(checks)}개 모두 목표 자세 있음"]
    lines = [f"방문할 수 없는 탐색 위치 {len(bad)}개 (그림의 빨간 X):"]
    lines += [f"  {c.location_id} ({c.category}, {c.room_id}): {PROBLEM_TEXT[c.problem]}"
              for c in bad]
    return lines


def example_goals(goals: Sequence[Pose2D], center, k: int = EXAMPLE_GOALS) -> List[Pose2D]:
    """Up to ``k`` goals spread around the location (by bearing from its center)."""
    if len(goals) <= k:
        return list(goals)
    ordered = sorted(goals, key=lambda g: math.atan2(g.y - center[1], g.x - center[0]))
    return [ordered[int(i * len(ordered) / k)] for i in range(k)]


def _extent(shape, origin, resolution):
    rows, cols = shape
    return [origin[0], origin[0] + cols * resolution,
            origin[1], origin[1] + rows * resolution]


def map_image(occupancy: OccupancyMap, grid: GridMap) -> np.ndarray:
    """RGB image of the map, row 0 = lowest y (draw with origin="lower")."""
    occ = occupancy.occupancy
    if (occ.shape != grid.free.shape or occupancy.origin != grid.origin
            or occupancy.resolution != grid.resolution):
        raise ValueError("지도와 주행 격자가 같은 지도에서 온 것이 아님")
    image = np.empty(occ.shape + (3,))
    image[:] = FREE
    image[(occ >= 0) & (occ < 50) & ~grid.free] = NEAR_OBSTACLE
    image[occ < 0] = UNKNOWN
    image[occ >= 50] = OCCUPIED
    return image


def _room_anchor(mask: np.ndarray, rooms: RoomMap) -> Tuple[float, float]:
    """Cell deepest inside the room (a label spot inside non-convex rooms)."""
    depth = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    row, col = np.unravel_index(int(np.argmax(depth)), mask.shape)
    return (rooms.origin[0] + (col + 0.5) * rooms.resolution,
            rooms.origin[1] + (row + 0.5) * rooms.resolution)




# --- drawing blocks, shared with athome.run_report --------------------------

def map_axes(occupancy: OccupancyMap, points: Sequence[Tuple[float, float]] = ()):
    """Figure and axes sized to the known map area and ``points``, and their
    bounds (x0, x1, y0, y1)."""
    from matplotlib.figure import Figure

    rows, cols = np.nonzero(occupancy.occupancy >= 0)
    if rows.size == 0:
        raise ValueError("지도에 알려진 칸이 없음")
    res, (ox, oy) = occupancy.resolution, occupancy.origin
    xs = [ox + cols.min() * res, ox + (cols.max() + 1) * res] + [p[0] for p in points]
    ys = [oy + rows.min() * res, oy + (rows.max() + 1) * res] + [p[1] for p in points]
    bounds = (min(xs) - 0.5, max(xs) + 0.5, min(ys) - 0.5, max(ys) + 0.5)
    x0, x1, y0, y1 = bounds
    scale = min(1.2, 11.0 / max(x1 - x0, y1 - y0))          # inches per meter
    fig = Figure(figsize=((x1 - x0) * scale + 4.0, (y1 - y0) * scale + 1.5))
    return fig, fig.add_subplot(), bounds


def location_corners(graph: SceneGraph) -> List[Tuple[float, float]]:
    return [tuple(p[:2]) for loc in graph.locations.values()
            for p in (loc.bbox_min, loc.bbox_max)]


def draw_map(ax, occupancy: OccupancyMap, grid: GridMap) -> None:
    ax.imshow(map_image(occupancy, grid), origin="lower", interpolation="nearest",
              extent=_extent(occupancy.occupancy.shape, occupancy.origin, occupancy.resolution))


def draw_rooms(ax, graph: SceneGraph, rooms: Optional[RoomMap] = None) -> None:
    """Room regions (with a room map) and room names."""
    from matplotlib.colors import to_rgba

    anchors = {}
    if rooms is not None:
        overlay = np.zeros(rooms.labels.shape + (4,))
        for label in np.unique(rooms.labels[rooms.labels > 0]):
            mask = rooms.labels == label
            overlay[mask] = to_rgba(ROOM_COLORS[(int(label) - 1) % len(ROOM_COLORS)], 0.25)
            anchors[f"room_{int(label)}"] = _room_anchor(mask, rooms)
        ax.imshow(overlay, origin="lower", interpolation="nearest",
                  extent=_extent(rooms.labels.shape, rooms.origin, rooms.resolution))
    for rid, room in graph.rooms.items():
        if rid not in anchors:            # no room map: middle of its locations
            centers = [((l.bbox_min[0] + l.bbox_max[0]) / 2, (l.bbox_min[1] + l.bbox_max[1]) / 2)
                       for l in graph.room_locations(rid)]
            if not centers:
                continue
            anchors[rid] = tuple(np.mean(centers, axis=0))
        ax.text(*anchors[rid], f"{room.label}\n{rid}", ha="center", va="center",
                fontsize=10, color="navy", zorder=5,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))


def draw_locations(ax, graph: SceneGraph,
                   checks: Optional[Dict[str, LocationCheck]] = None) -> None:
    """Search Locations; with ``checks`` also their goal poses and problems."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle

    for lid, loc in sorted(graph.locations.items()):
        check = checks[lid] if checks is not None else None
        workspace = loc.kind == WORKSPACE
        color = WORKSPACE_COLOR if workspace else STANDALONE_COLOR
        (lx, ly), (hx, hy) = loc.bbox_min[:2], loc.bbox_max[:2]
        center = ((lx + hx) / 2, (ly + hy) / 2)
        problem = check.problem if check is not None else None
        ax.add_patch(Rectangle(
            (lx, ly), hx - lx, hy - ly, facecolor=to_rgba(color, 0.18), zorder=3,
            edgecolor=PROBLEM_COLOR if problem else color,
            linewidth=1.5 if workspace else 0.8, linestyle="-" if workspace else "--"))
        ax.text(*center, loc.category, ha="center", va="center", zorder=6,
                fontsize=7 if workspace else 6, color="black",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.8))
        if check is None:
            continue
        if check.goals:
            ax.scatter([g.x for g in check.goals], [g.y for g in check.goals], s=2,
                       color=GOAL_COLOR, alpha=0.35, linewidths=0, zorder=4)
        for g in example_goals(check.goals, center):
            ax.plot(g.x, g.y, "o", color=GOAL_COLOR, markersize=3.5, zorder=4)
            ax.annotate("", xytext=(g.x, g.y), zorder=4,
                        xy=(g.x + ARROW_LENGTH_M * math.cos(g.yaw),
                            g.y + ARROW_LENGTH_M * math.sin(g.yaw)),
                        arrowprops=dict(arrowstyle="-|>", color=GOAL_COLOR, lw=1.2,
                                        shrinkA=0, shrinkB=0))
        if problem:
            ax.plot(*center, marker="x", color=PROBLEM_COLOR, markersize=12, mew=2.5, zorder=7)
            ax.text(center[0], ly - 0.08, problem.replace("_", " "), ha="center",
                    va="top", fontsize=7, color=PROBLEM_COLOR, zorder=7)


def base_legend(rooms: bool) -> list:
    """Legend entries of the map, rooms and Search Locations."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=FREE, edgecolor="0.6", label="free"),
        Patch(facecolor=NEAR_OBSTACLE, label="too close to obstacles for the robot center"),
        Patch(facecolor=OCCUPIED, label="occupied"),
        Patch(facecolor=UNKNOWN, label="unknown"),
    ]
    if rooms:
        handles.append(Patch(facecolor=to_rgba(ROOM_COLORS[0], 0.25),
                             label="room region (one color per room)"))
    return handles + [
        Patch(facecolor=to_rgba(WORKSPACE_COLOR, 0.18), edgecolor=WORKSPACE_COLOR,
              label="workspace (search location)"),
        Patch(facecolor=to_rgba(STANDALONE_COLOR, 0.18), edgecolor=STANDALONE_COLOR,
              linestyle="--", label="standalone object (search location)"),
    ]


def finish_axes(ax, bounds, title: str, handles: list) -> None:
    """Map origin, 1 m grid, title and legend."""
    from matplotlib.ticker import MultipleLocator

    x0, x1, y0, y1 = bounds
    if x0 <= 0 <= x1 and y0 <= 0 <= y1:
        ax.plot(0, 0, marker="+", color="purple", markersize=14, mew=2, zorder=7)
        ax.text(0.08, 0.08, "map (0, 0)", color="purple", fontsize=8, zorder=7)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if max(x1 - x0, y1 - y0) <= 40:
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.grid(color="0.5", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9, loc="left")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, fontsize=8)


def render_overview(path, graph: SceneGraph, occupancy: OccupancyMap, grid: GridMap,
                    checks: Dict[str, LocationCheck], rooms: Optional[RoomMap] = None,
                    title: str = "") -> None:
    from matplotlib.lines import Line2D

    fig, ax, bounds = map_axes(occupancy, location_corners(graph))
    draw_map(ax, occupancy, grid)
    draw_rooms(ax, graph, rooms)
    draw_locations(ax, graph, checks)
    workspaces = sum(l.kind == WORKSPACE for l in graph.locations.values())
    problems = [c.problem for c in checks.values()]
    summary = (f"rooms {len(graph.rooms)}, search locations {len(graph.locations)} "
               f"(workspace {workspaces}, standalone {len(graph.locations) - workspaces}); "
               f"no goal pose {problems.count(NO_GOAL)}, "
               f"not connected {problems.count(DISCONNECTED)}")
    handles = base_legend(rooms is not None) + [
        Line2D([], [], marker="o", linestyle="", color=GOAL_COLOR, alpha=0.35, markersize=3,
               label="goal pose candidates (robot center)"),
        Line2D([], [], marker="o", color=GOAL_COLOR, markersize=4,
               label="goal pose examples (arrow = facing)"),
        Line2D([], [], marker="x", linestyle="", color=PROBLEM_COLOR, mew=2.5, markersize=9,
               label="cannot be visited (reason below it)"),
    ]
    finish_axes(ax, bounds, "\n".join(s for s in (title, summary) if s), handles)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def load_robot_graph(config, graph_path: Path) -> Tuple[dict, SceneGraph]:
    """Scene graph as the search server loads it (same Search Locations);
    refuses a graph built on another map than the robot config's."""
    raw = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    built_on = (raw.get("provenance") or {}).get("map_version")
    if built_on is not None and built_on != config.map_version:
        raise ValueError(f"그래프는 지도 {built_on}에서 만들어짐, 현재 지도는 {config.map_version}")
    return raw, SceneGraph(raw, excluded_categories=excluded_categories(config.search_locations))


def load_room_map(graph_path: Path) -> Optional[RoomMap]:
    """<graph>.rooms.npz of build_scene_graph.py, if present."""
    path = Path(graph_path).with_name(Path(graph_path).stem + ".rooms.npz")
    return RoomMap.load(path) if path.is_file() else None


def write_overview(config, graph_path: Path, out_path: Path) -> Dict[str, LocationCheck]:
    """Overview of ``graph_path`` as the robot of ``config`` (athome.config.
    RobotConfig) would search it: same Search Locations, map inflation and
    goal geometry as the search server. Room colors from <graph>.rooms.npz."""
    raw, graph = load_robot_graph(config, graph_path)
    grid = load_map_server(config.map_yaml, config.inflation_radius, config.unknown_as_occupied)
    checks = check_locations(graph, grid, NavigationConfig(
        goal_offset=config.goal_offset, goal_clearance=config.goal_clearance,
        goal_max_offset=config.goal_max_offset))
    built_on = (raw.get("provenance") or {}).get("map_version")
    offset = f"goal offset {config.goal_offset:.2f} m" + (
        f" (up to {config.goal_max_offset:.2f} m)" if config.goal_max_offset else "")
    render_overview(
        out_path, graph, load_occupancy(config.map_yaml), grid, checks, load_room_map(graph_path),
        title=(f"{Path(graph_path).name} on {config.map_yaml.name} ({config.map_version}"
               f"{'' if built_on else ', graph has no map version'})\n"
               f"robot radius {config.inflation_radius:.2f} m, {offset}"))
    return checks
