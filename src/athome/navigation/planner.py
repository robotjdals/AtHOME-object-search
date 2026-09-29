"""Path cost and goal poses for search locations from the current pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from athome.navigation.goal_poses import goal_candidates, spread_goals
from athome.navigation.grid import Cell, GridMap
from athome.navigation.path_cost import shortest_paths
from athome.schemas import Pose2D


@dataclass(frozen=True)
class NavigationConfig:
    # Distance from the object footprint to the robot center.
    goal_offset: float = 0.45
    # Extra clearance of a goal cell for in-place rotation (see goal_candidates).
    goal_clearance: float = 0.0
    # Farthest goal distance when the ring at goal_offset is blocked (None: only goal_offset).
    goal_max_offset: Optional[float] = None
    max_goals_per_location: int = 3
    goal_separation: float = 0.4
    # Snap a start pose that falls in the inflated area to free space.
    start_snap_distance: float = 0.3


@dataclass(frozen=True)
class LocationCost:
    location_id: str
    cost: float                  # minimum A* path cost [m]
    goals: Tuple[Pose2D, ...]    # cheapest first, spread apart


class StartNotFree(RuntimeError):
    pass


class NavigationPlanner:
    def __init__(self, grid: GridMap, config: NavigationConfig = NavigationConfig()):
        self.grid = grid
        self.config = config
        self._candidates: Dict[str, List[Tuple[Cell, Pose2D]]] = {}

    def add_location(
        self,
        location_id: str,
        bbox_min_xy: Sequence[float],
        bbox_max_xy: Sequence[float],
    ) -> int:
        """Precompute goal candidates (static map). Returns the count.
        Cached, so sessions sharing a planner compute them once."""
        if location_id in self._candidates:
            return len(self._candidates[location_id])
        candidates = goal_candidates(
            self.grid, bbox_min_xy, bbox_max_xy, self.config.goal_offset,
            self.config.goal_clearance, self.config.goal_max_offset,
        )
        self._candidates[location_id] = candidates
        return len(candidates)

    def candidate_goals(self, location_id: str) -> List[Tuple[Cell, Pose2D]]:
        """Precomputed goal candidates of a location (cell, pose)."""
        return list(self._candidates.get(location_id, ()))

    def has_candidates(self, location_id: str) -> bool:
        return bool(self._candidates.get(location_id))

    def evaluate(
        self, start: Pose2D, location_ids: Iterable[str]
    ) -> Dict[str, LocationCost]:
        """Reachable locations only; unreachable ones are left out."""
        location_ids = [l for l in location_ids if self.has_candidates(l)]
        if not location_ids:
            return {}

        start_cell = self.grid.nearest_free(
            self.grid.to_cell(start.x, start.y),
            self.config.start_snap_distance,
        )
        if start_cell is None:
            raise StartNotFree(f"시작 위치 {start}가 주행 가능 영역 밖")

        goals = {
            cell for lid in location_ids for cell, _ in self._candidates[lid]
        }
        reached, _ = shortest_paths(
            self.grid.free, start_cell, goals, self.grid.resolution
        )

        out = {}
        for lid in location_ids:
            scored = sorted(
                (
                    (reached[cell], pose)
                    for cell, pose in self._candidates[lid]
                    if cell in reached
                ),
                key=lambda item: item[0],
            )
            if not scored:
                continue
            out[lid] = LocationCost(
                location_id=lid,
                cost=scored[0][0],
                goals=spread_goals(
                    [pose for _, pose in scored],
                    self.config.max_goals_per_location,
                    self.config.goal_separation,
                ),
            )
        return out
