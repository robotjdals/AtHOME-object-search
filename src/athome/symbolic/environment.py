"""Review-only symbolic executor. GT never enters the planner graph.

Observation follows the real VisitExecutor protocol: after reaching the goal,
the robot turns to ``heading_count`` headings starting from the goal yaw and
observes once per heading (an isotropic observer covers them in one call).
Nothing is observed while driving. Visibility comes from an injected
observer: 2D wall line of sight (wall_los.py, default) or Habitat semantic
rendering (habitat_observer.py, reference for validation).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Protocol, Sequence, Tuple

from athome.execution.visit import VisitOutcome, VisitStatus, wrap_angle
from athome.navigation.grid import GridMap
from athome.navigation.path_cost import extract_path, shortest_paths
from athome.schemas import ObservationFrame, ObservedObject, Pose2D


@dataclass(frozen=True)
class GroundTruthObject:
    object_id: str
    category: str
    semantic_id: int
    centroid: Tuple[float, float, float]


@dataclass(frozen=True)
class TimeModel:
    """Modelled search time, so a symbolic episode can be held to the robot's
    time budget: travel at ``speed_mps`` plus, per visit, one full turn at
    ``rotation_speed_radps`` and ``observation_window_s`` per heading
    (VisitExecutor protocol)."""
    speed_mps: float
    rotation_speed_radps: float
    observation_window_s: float

    def __post_init__(self):
        if min(self.speed_mps, self.rotation_speed_radps) <= 0 or self.observation_window_s < 0:
            raise ValueError("시간 모델 값이 잘못됐습니다.")

    def elapsed(self, distance_m: float, visits: int, heading_count: int) -> float:
        per_visit = 2 * math.pi / self.rotation_speed_radps + heading_count * self.observation_window_s
        return distance_m / self.speed_mps + visits * per_visit


class Observer(Protocol):
    # False: isotropic sensor, observed once per visit instead of per heading.
    per_heading: bool

    def observe(self, x: float, y: float, floor_z: float, yaw: float) -> Mapping[int, float]:
        """Observed semantic IDs -> evidence (pixel count or distance)."""


class SymbolicEnvironment:
    def __init__(self, grid: GridMap, start: Pose2D,
                 objects: Sequence[GroundTruthObject], observer: Observer,
                 floor_z: Callable[[float, float], float], heading_count: int,
                 time_model: "TimeModel | None" = None):
        if heading_count <= 0:
            raise ValueError("heading_count는 양수여야 합니다.")
        self.grid = grid
        self.pose = start
        self._cell(start)
        self._objects = tuple(sorted(objects, key=lambda obj: obj.object_id))
        if len({o.object_id for o in self._objects}) != len(self._objects):
            raise ValueError("GT ID 중복")
        if len({o.semantic_id for o in self._objects}) != len(self._objects):
            raise ValueError("GT semantic ID 중복")
        for obj in self._objects:
            if len(obj.centroid) != 3 or not all(map(math.isfinite, obj.centroid)):
                raise ValueError("잘못된 GT 중심")
        self._observer = observer
        self._floor_z = floor_z
        self._heading_count = heading_count
        # Explicit mapping from HM3D string IDs to observation integer IDs.
        self.id_map = {i: obj.object_id for i, obj in enumerate(self._objects)}
        self.distance_m = 0.0
        self.visits = 0
        self.time_model = time_model
        self._tick = 0
        self.last_path = []
        self.last_detections = []

    def elapsed_s(self) -> float:
        if self.time_model is None:
            raise RuntimeError("시간 모델이 없습니다.")
        return self.time_model.elapsed(self.distance_m, self.visits, self._heading_count)

    def _cell(self, pose):
        if not all(map(math.isfinite, (pose.x, pose.y, pose.yaw))):
            raise ValueError("잘못된 Pose")
        cell = self.grid.to_cell(pose.x, pose.y)
        if not self.grid.is_free(cell):
            raise ValueError("Pose가 주행 가능한 셀 밖에 있습니다.")
        xy = self.grid.to_xy(cell)
        if not (math.isclose(pose.x, xy[0], abs_tol=1e-8, rel_tol=0)
                and math.isclose(pose.y, xy[1], abs_tol=1e-8, rel_tol=0)):
            raise ValueError("Symbolic Pose는 셀 중심이어야 합니다.")
        return cell

    def visit(self, decision):
        if not decision.goals:
            raise ValueError("Goal 없음")
        start = self._cell(self.pose)
        goal_pose = decision.goals[0]
        goal = self._cell(goal_pose)
        costs, parents = shortest_paths(
            self.grid.free, start, [goal], self.grid.resolution)
        if goal not in costs:
            raise ValueError("계획 이후 경로가 달라졌습니다.")
        if not math.isclose(costs[goal], decision.cost, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Planner와 Symbolic 경로 비용 불일치")
        path = extract_path(parents, start, goal)
        floor_z = self._floor_z(goal_pose.x, goal_pose.y)
        # Same heading sequence as VisitExecutor (final pose == goal here).
        # An isotropic observer already covers all headings in one call.
        if getattr(self._observer, "per_heading", True):
            headings = [wrap_angle(goal_pose.yaw + 2 * math.pi * k / self._heading_count)
                        for k in range(self._heading_count)]
        else:
            headings = [None]
        frames, detections = [], []
        seen = set()
        for index, yaw in enumerate(headings):
            self._tick += 1
            evidence = self._observer.observe(goal_pose.x, goal_pose.y, floor_z,
                                              goal_pose.yaw if yaw is None else yaw)
            observed = []
            for i, obj in enumerate(self._objects):
                if i not in seen and obj.semantic_id in evidence:
                    seen.add(i)
                    observed.append(ObservedObject(i, obj.category, 1.0, obj.centroid))
                    detections.append({
                        "object_id": obj.object_id,
                        "heading_index": None if yaw is None else index, "yaw": yaw,
                        "evidence": evidence[obj.semantic_id],
                        "observation_tick": self._tick})
            frames.append(ObservationFrame(float(self._tick), tuple(observed)))
        self.pose = goal_pose
        self.distance_m += costs[goal]
        self.visits += 1
        self.last_path = path
        self.last_detections = detections
        return VisitOutcome(
            decision.location_id, VisitStatus.COMPLETED,
            goal=goal_pose, final_pose=goal_pose, observations=frames, nav_attempts=1)
