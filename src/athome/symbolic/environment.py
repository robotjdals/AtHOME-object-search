"""Review-only symbolic executor. GT never enters the planner graph.

Observation follows the real VisitExecutor protocol: after reaching the goal,
the robot turns to ``heading_count`` headings starting from the goal yaw and
observes once per heading (an isotropic observer covers them in one call).
Nothing is observed while driving. Visibility comes from an injected
observer: 2D wall line of sight (wall_los.py, default) or Habitat semantic
rendering (habitat_observer.py, reference for validation).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Callable, Mapping, Protocol, Sequence, Tuple

from athome.execution.visit import VisitOutcome, VisitStatus, wrap_angle
from athome.navigation.grid import GridMap
from athome.navigation.path_cost import extract_path, shortest_paths
from athome.schemas import ObservationFrame, ObservedObject, Pose2D

# Observation IDs of false detections map to this prefix + label in id_map.
FALSE_DETECTION = "false_detection:"


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


@dataclass(frozen=True)
class DetectionModel:
    """Missed detections: an object in view is detected at a visit with
    probability ``recall`` (the detection probability of search theory, Stone
    1975; the true-positive rate of the detector in POMDP object search,
    Wandzel et al. 2019). Draws are independent across objects, locations and
    repeated visits of a location, and deterministic: a hash of (``seed``,
    object, location, visit number), so every policy evaluated on the same
    episode (``seed``) gets the same outcome for the same visit (common random
    numbers for paired comparisons). No false positives are modelled."""
    recall: float
    seed: str

    def __post_init__(self):
        if not 0.0 < self.recall <= 1.0:
            raise ValueError("recall은 (0, 1] 범위여야 합니다.")

    def detected(self, object_id: str, location_id: str, visit_number: int) -> bool:
        if self.recall >= 1.0:
            return True
        key = f"{self.seed}|{object_id}|{location_id}|{visit_number}".encode()
        u = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2.0**64
        return u < self.recall


@dataclass(frozen=True)
class ViewpointDetectionModel:
    """Missed and false detections as measured on the robot (412 records,
    scripts/fit_detection_412.py, configs/perception/detection_412.json): a
    detection depends on the viewpoint, the object and its category, not on
    chance at each look.

    - Each object has a detectability q ~ Beta(mean (1/rho - 1), (1 - mean)
      (1/rho - 1)): mean detection probability ``mean``, correlation ``rho``
      between the viewpoints of one object (rho 0: viewpoints independent).
    - A search location (viewpoint) detects the object with probability q,
      and gives the same outcome at every visit: looking again from where an
      object was missed does not find it (robot: 0.02-0.09 after a miss).
    - False detections are a property of the category: each label of
      ``false_labels`` gets a rate drawn from ``false_positive_rates`` (the
      measured rates of individual categories, most of them 0), and a
      location where no object of the label is in view reports one with that
      rate, again fixed per location.
    Draws are deterministic hashes of ``seed``, the scene (detectability and
    look-alike objects belong to the house): every policy and every start
    state of a scene meets the same detections (common random numbers)."""
    mean: float
    rho: float
    false_positive_rates: Tuple[float, ...]
    seed: str
    false_labels: Tuple[str, ...] = ()

    def __post_init__(self):
        if not 0.0 < self.mean <= 1.0 or not 0.0 <= self.rho < 1.0:
            raise ValueError("mean은 (0, 1], rho는 [0, 1) 범위여야 합니다.")
        if not self.false_positive_rates or not all(0.0 <= r < 1.0 for r in self.false_positive_rates):
            raise ValueError("false_positive_rates는 [0, 1) 값이 하나 이상 있어야 합니다.")

    def _uniform(self, *parts) -> float:
        key = "|".join((self.seed,) + parts).encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2.0**64

    def detectability(self, object_id: str) -> float:
        if self.rho == 0.0 or self.mean >= 1.0:
            return self.mean
        from scipy.special import betaincinv
        scale = 1.0 / self.rho - 1.0
        return float(betaincinv(self.mean * scale, (1.0 - self.mean) * scale, self._uniform("object", object_id)))

    def detected(self, object_id: str, location_id: str, visit_number: int) -> bool:
        return self._uniform("viewpoint", object_id, location_id) < self.detectability(object_id)

    def false_rate(self, label: str) -> float:
        rates = self.false_positive_rates
        return rates[min(int(self._uniform("false_rate", label) * len(rates)), len(rates) - 1)]

    def false_detection(self, location_id: str, label: str) -> bool:
        return self._uniform("false", location_id, label) < self.false_rate(label)


class Observer(Protocol):
    # False: isotropic sensor, observed once per visit instead of per heading.
    per_heading: bool

    def observe(self, x: float, y: float, floor_z: float, yaw: float) -> Mapping[int, float]:
        """Observed semantic IDs -> evidence (pixel count or distance)."""


class SymbolicEnvironment:
    def __init__(self, grid: GridMap, start: Pose2D,
                 objects: Sequence[GroundTruthObject], observer: Observer,
                 floor_z: Callable[[float, float], float], heading_count: int,
                 time_model: "TimeModel | None" = None, verify_path: bool = True,
                 detection: "DetectionModel | ViewpointDetectionModel | None" = None):
        """``verify_path``: recompute the path at every visit to check it
        against the planner's cost and record it (review tools). Training
        and evaluation loops may skip it: the travelled distance is the
        planner's cost of the same Dijkstra from the same cell either way.
        ``detection``: missed detections (DetectionModel) or missed and false
        detections (ViewpointDetectionModel); None detects every object in
        view. A false detection is reported as an object of its label at the
        visited location, with an observation ID that ``id_map`` maps to
        ``false_detection:<label>``."""
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
        self._false_ids = {label: len(self._objects) + k
                           for k, label in enumerate(getattr(detection, "false_labels", ()))}
        self.id_map.update({i: f"{FALSE_DETECTION}{label}" for label, i in self._false_ids.items()})
        self.distance_m = 0.0
        self.visits = 0
        self.time_model = time_model
        self._tick = 0
        self.last_path = []
        self.last_detections = []
        self.verify_path = verify_path
        self.detection = detection
        self._location_visits = {}
        self._location_spots = {}
        self.last_missed = []
        # Observation effort over all visits: headings looked at and the
        # in-place turning [rad] to reach them from each goal yaw.
        self.headings_observed = 0
        self.turn_rad = 0.0
        self.rotations = 0               # in-place rotation goals (turns > 1 deg)

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

    def _viewpoint(self, location_id: str, pose: Pose2D, same_within_m: float = 0.5) -> str:
        """Key of the detection draws: a location is one viewpoint per place
        the robot stood at (the first keeps the location ID)."""
        spots = self._location_spots.setdefault(location_id, [])
        for k, (x, y) in enumerate(spots):
            if math.hypot(pose.x - x, pose.y - y) <= same_within_m:
                break
        else:
            k = len(spots)
            spots.append((pose.x, pose.y))
        return location_id if k == 0 else f"{location_id}#{k}"

    def visit(self, decision, headings=None, stop=None):
        """``headings``: absolute yaws to observe in order (default: the
        visit's ``heading_count`` headings from the goal yaw); needs a
        per-heading observer. ``stop(frames)``: after each heading, True ends
        the observation (VisitRequest.satisfied)."""
        if not decision.goals:
            raise ValueError("Goal 없음")
        start = self._cell(self.pose)
        goal_pose = decision.goals[0]
        goal = self._cell(goal_pose)
        if self.verify_path:
            costs, parents = shortest_paths(
                self.grid.free, start, [goal], self.grid.resolution)
            if goal not in costs:
                raise ValueError("계획 이후 경로가 달라졌습니다.")
            if not math.isclose(costs[goal], decision.cost, rel_tol=0, abs_tol=1e-8):
                raise ValueError("Planner와 Symbolic 경로 비용 불일치")
            path = extract_path(parents, start, goal)
        else:
            costs, path = {goal: decision.cost}, []
        floor_z = self._floor_z(goal_pose.x, goal_pose.y)
        # Same heading sequence as VisitExecutor (final pose == goal here).
        # An isotropic observer already covers all headings in one call.
        per_heading = getattr(self._observer, "per_heading", True)
        if headings is not None:
            if not per_heading:
                raise ValueError("headings를 주려면 방향별 관측 모델이 필요합니다.")
            headings = list(headings)
        elif per_heading:
            headings = [wrap_angle(goal_pose.yaw + 2 * math.pi * k / self._heading_count)
                        for k in range(self._heading_count)]
        else:
            headings = [None]
        frames, detections = [], []
        seen = set()
        # One draw per object and visit (the robot looks once per heading;
        # recall is per visit, whatever the number of headings).
        viewpoint = self._viewpoint(decision.location_id, goal_pose)
        visit_number = self._location_visits.get(decision.location_id, 0) + 1
        self._location_visits[decision.location_id] = visit_number
        missed = set()
        for index, yaw in enumerate(headings):
            self._tick += 1
            evidence = self._observer.observe(goal_pose.x, goal_pose.y, floor_z,
                                              goal_pose.yaw if yaw is None else yaw)
            observed = []
            for i, obj in enumerate(self._objects):
                if i not in seen and i not in missed and obj.semantic_id in evidence:
                    if self.detection is not None and not self.detection.detected(
                            obj.object_id, viewpoint, visit_number):
                        missed.add(i)
                        continue
                    seen.add(i)
                    observed.append(ObservedObject(i, obj.category, 1.0, obj.centroid))
                    detections.append({
                        "object_id": obj.object_id,
                        "heading_index": None if yaw is None else index, "yaw": yaw,
                        "evidence": evidence[obj.semantic_id],
                        "observation_tick": self._tick})
            frames.append(ObservationFrame(float(self._tick), tuple(observed)))
            if yaw is not None:
                turn = abs(wrap_angle(yaw - (headings[index - 1] if index else goal_pose.yaw)))
                self.turn_rad += turn
                self.rotations += turn > math.radians(1.0)
            self.headings_observed += 1 if yaw is not None else self._heading_count
            if stop is not None and stop(frames):
                break
        in_view = {self._objects[i].category for i in seen | missed}
        false = [label for label in self._false_ids
                 if label not in in_view and self.detection.false_detection(viewpoint, label)]
        if false:
            phantom = tuple(ObservedObject(self._false_ids[label], label, 1.0,
                                           (goal_pose.x, goal_pose.y, floor_z + 0.75)) for label in false)
            frames[0] = replace(frames[0], objects=frames[0].objects + phantom)
        self.pose = goal_pose
        self.distance_m += costs[goal]
        self.visits += 1
        self.last_path = path
        self.last_detections = detections
        self.last_missed = sorted(self._objects[i].object_id for i in missed)
        return VisitOutcome(
            decision.location_id, VisitStatus.COMPLETED,
            goal=goal_pose, final_pose=goal_pose, observations=frames, nav_attempts=1)
