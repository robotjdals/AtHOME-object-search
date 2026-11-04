"""Execute one search location visit: navigate, then observe in 6 headings.

The executor only reports what happened. Whether the location becomes
Visited is decided by the search layer (only on COMPLETED).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple

from athome.execution.interfaces import (
    Clock,
    HandleState,
    MotionClient,
    MotionHandle,
    ObservationSource,
)
from athome.schemas import (
    MotionReason,
    MotionResult,
    MotionStatus,
    NavigationRequest,
    ObservationFrame,
    ObservedObject,
    Pose2D,
    RotationRequest,
)


class VisitStatus(Enum):
    COMPLETED = "completed"
    FAILED = "failed"        # this location could not be visited
    CANCELED = "canceled"    # canceled by the caller, robot stopped
    PAUSED = "paused"        # system problem, do not continue automatically


class VisitReason(Enum):
    NONE = "none"
    NAVIGATION_FAILED = "navigation_failed"
    ROTATION_FAILED = "rotation_failed"
    CANCELED_BY_REQUEST = "canceled_by_request"
    MOTION_REJECTED = "motion_rejected"
    MOTION_UNRESPONSIVE = "motion_unresponsive"
    UNEXPECTED_CANCEL = "unexpected_cancel"
    STOP_UNCONFIRMED = "stop_unconfirmed"
    LOCALIZATION_LOST = "localization_lost"
    MAP_MISMATCH = "map_mismatch"
    MOTION_INTERNAL_ERROR = "motion_internal_error"
    PERCEPTION_LOST = "perception_lost"


# Retry the same goal first.
_RETRY_REASONS = {
    MotionReason.BLOCKED,
    MotionReason.NO_PROGRESS,
    MotionReason.TIMEOUT,
}
# Retrying the same goal is pointless; try the next goal candidate.
_SKIP_GOAL_REASONS = {
    MotionReason.INVALID_GOAL,
    MotionReason.NO_PATH,
}
_PAUSE_REASONS = {
    MotionReason.LOCALIZATION_LOST: VisitReason.LOCALIZATION_LOST,
    MotionReason.MAP_MISMATCH: VisitReason.MAP_MISMATCH,
}


class Phase(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    ROTATING = "rotating"
    OBSERVING = "observing"
    WAITING = "waiting"        # pause before retrying the same goal
    CANCELING = "canceling"
    DONE = "done"


@dataclass(frozen=True)
class RelookPoint:
    """Something worth centring in the view: ``kind`` "found" (a target
    confirmed, to be checked again) or "near_miss" (named as the target, not
    passing the found check)."""
    x: float
    y: float
    target: str
    kind: str


@dataclass(frozen=True)
class VisitRequest:
    visit_id: str
    # Goal pose candidates of one search location, in priority order.
    goals: Tuple[Pose2D, ...]
    map_version: str
    # Target confirmation (SearchSession.confirmations): when a frame at a
    # heading shows an object of interest (a possible target), stay at that
    # heading until it was seen in ``confirm_frames`` frames or the
    # observation timeout, instead of turning on after the first frame.
    interest: Optional[Callable[[ObservedObject], bool]] = None
    confirm_frames: int = 1
    # After each heading, with the visit's frames so far: True when the
    # search needs nothing more from this visit (every open target of the
    # command confirmed), so the remaining headings are skipped.
    satisfied: Optional[Callable[[List[ObservationFrame]], bool]] = None
    # Re-look (VisitConfig.relook_max): relook(frames, passing) -> points seen
    # in the visit's frames so far or in ``passing``, the frames of the last
    # turn (for finding candidates only, never for the found check), most
    # important first. relook_result(point, frames) judges the frames of the
    # look at ``point``.
    relook: Optional[Callable[[List[ObservationFrame], List[ObservationFrame]],
                              Sequence[RelookPoint]]] = None
    relook_result: Optional[Callable[[RelookPoint, List[ObservationFrame]], bool]] = None
    # Map points looked at before the regular headings (Known objects), unless
    # the robot already faces them.
    look_first: Tuple[Tuple[float, float], ...] = ()


@dataclass(frozen=True)
class VisitConfig:
    nav_retries_per_goal: int = 1
    rotation_retries: int = 1
    # 60-degree steps < D435i RGB horizontal FOV 69.4 deg, so one turn covers
    # 360 degrees (VLFM turns a full circle; Habitat ObjectNav turns 30 deg with
    # a 79 deg FOV). The training observation model assumes this full disk.
    heading_count: int = 6
    # Order of the headings: "sequential" turns one way round from the goal
    # yaw (the training observation model's turn); "front_first" looks at the
    # goal yaw, one step right and one step left first, then on round
    # (athome.execution.headings.front_first).
    heading_order: str = "sequential"
    # Turns per visit to centre something seen off-centre (0: none). Turning
    # changes the view only for what no heading faces: bearings further than
    # ``central_angle`` from every heading of the visit.
    relook_max: int = 0
    # Half-width 15 deg of an object still fits a 70.6 deg view at this offset.
    central_angle: float = math.radians(20.0)
    # Yaw tolerance [rad] of each observation rotation; 0 leaves it to the
    # motion module. Set it with observation_yaw_tolerance() so neighbouring
    # views still overlap (the full-disk assumption holds).
    rotation_yaw_tolerance: float = 0.0
    # Includes path planning when the adapter plans before sending.
    accept_timeout: float = 5.0
    feedback_timeout: float = 1.0
    cancel_timeout: float = 5.0
    observation_window: float = 1.0
    observation_min_frames: int = 3
    # Frames without the possible target before it counts as not confirmed.
    confirm_patience_frames: int = 6
    observation_timeout: float = 8.0
    # Perception gaps shorter than this are waited out.
    perception_timeout: float = 3.0
    # Before retrying the same goal (a person in the way often moves on).
    retry_wait: float = 3.0
    # Executor-side limits, in case the motion module never finishes.
    # 0 disables. Treated as a TIMEOUT failure (retry rules apply).
    navigation_timeout: float = 180.0
    rotation_timeout: float = 30.0


def observation_yaw_tolerance(camera_hfov: float, heading_count: int) -> float:
    """Largest per-heading yaw error [rad] that leaves no gap between views.

    Headings are ``2*pi/n`` apart and targeted absolutely (errors do not
    accumulate); two neighbouring views of horizontal FOV ``camera_hfov``
    still touch while each is off by at most (FOV - 2*pi/n) / 2 (4.7 deg for
    the D435i RGB, 69.4 deg, at 6 headings)."""
    tolerance = (camera_hfov - 2 * math.pi / heading_count) / 2
    if tolerance <= 0:
        raise ValueError(
            f"카메라 시야 {math.degrees(camera_hfov):.1f}°로는 {heading_count}방향이 360°를 덮지 못함")
    return tolerance


def robot_visit_config(config) -> "VisitConfig":
    """VisitConfig of a robot config (athome.config.RobotConfig): rotation
    yaw tolerance from the camera FOV (sensors.camera_hfov_deg) and the
    per-heading wait from ``observation`` (perception processing time)."""
    o = config.observation
    kwargs = {
        "observation_window": float(o.get("window_s", VisitConfig.observation_window)),
        "observation_min_frames": int(o.get("min_frames", VisitConfig.observation_min_frames)),
        "observation_timeout": float(o.get("timeout_s", VisitConfig.observation_timeout)),
        "perception_timeout": float(o.get("perception_timeout_s", VisitConfig.perception_timeout)),
        "heading_order": str(o.get("heading_order", VisitConfig.heading_order)),
        "heading_count": int(o.get("heading_count", VisitConfig.heading_count)),
        "relook_max": int(o.get("relook_max", VisitConfig.relook_max)),
        "central_angle": math.radians(float(o.get("central_angle_deg",
                                                  math.degrees(VisitConfig.central_angle)))),
    }
    if config.camera_hfov_deg is not None:
        kwargs["rotation_yaw_tolerance"] = observation_yaw_tolerance(
            math.radians(config.camera_hfov_deg), kwargs["heading_count"])
    return VisitConfig(**kwargs)


@dataclass
class VisitOutcome:
    visit_id: str
    status: VisitStatus
    reason: VisitReason = VisitReason.NONE
    detail: str = ""
    # Goal the robot actually reached, if any.
    goal: Optional[Pose2D] = None
    final_pose: Optional[Pose2D] = None
    observations: List[ObservationFrame] = field(default_factory=list)
    nav_attempts: int = 0
    # Headings observed, of heading_count (fewer when the search was satisfied).
    headings_observed: int = 0
    # Looks at off-centre points: {target, kind, x, y, yaw, passed}.
    relooks: List[dict] = field(default_factory=list)


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class VisitExecutor:
    def __init__(
        self,
        motion: MotionClient,
        observation: ObservationSource,
        clock: Clock,
        config: VisitConfig = VisitConfig(),
    ):
        self._motion = motion
        self._observation = observation
        self._clock = clock
        self._config = config
        self._phase = Phase.IDLE
        self._outcome: Optional[VisitOutcome] = None
        # Canceled handle whose stop was never confirmed.
        self._unconfirmed: Optional[MotionHandle] = None

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def busy(self) -> bool:
        return self._phase not in (Phase.IDLE, Phase.DONE)

    @property
    def outcome(self) -> Optional[VisitOutcome]:
        return self._outcome

    def progress(self) -> dict:
        """Where the running visit is (read-only, for status views): the
        goal being driven to, the heading being observed, how long the
        observation has waited and in how many of its frames a possible
        target was seen."""
        if not self.busy:
            return {"phase": self._phase.value}
        p = {"phase": self._phase.value, "visit_id": self._request.visit_id,
             "goal": self._request.goals[self._goal_index],
             "goal_index": self._goal_index, "goal_count": len(self._request.goals),
             "nav_attempts": self._nav_attempts,
             "heading": self._heading_index, "heading_count": len(self._headings)}
        if self._relook_point is not None:
            p["relook"] = {"target": self._relook_point.target, "kind": self._relook_point.kind}
        if self._phase == Phase.OBSERVING:
            frames = self._observation.frames_since(self._observe_from)
            interest = self._request.interest
            p.update(observing_s=self._clock() - self._observe_from, frames=len(frames),
                     candidate_frames=(sum(any(interest(o) for o in f.objects) for f in frames)
                                       if interest is not None else 0),
                     confirm_frames=self._request.confirm_frames)
        return p

    @property
    def stop_confirmed(self) -> bool:
        """False while a timed-out cancel has not reported a stopped result."""
        h = self._unconfirmed
        if h is None:
            return True
        if h.state == HandleState.REJECTED or (
            h.state == HandleState.DONE and h.result.stopped
        ):
            self._unconfirmed = None
            return True
        return False

    def start(self, request: VisitRequest) -> None:
        if self.busy:
            raise RuntimeError("이미 실행 중인 방문이 있음")
        if not request.goals:
            raise ValueError(f"{request.visit_id}: Goal 후보 없음")
        if not self.stop_confirmed:
            raise RuntimeError(
                f"{self._unconfirmed.request_id}: 정지 확인 전 새 동작 금지"
            )

        self._request = request
        self._outcome = None
        self._seq = 0
        self._handle: Optional[MotionHandle] = None
        self._sent_at = 0.0
        self._goal_index = 0
        self._goal_attempt = 0
        self._nav_attempts = 0
        self._reached_goal: Optional[Pose2D] = None
        self._final_pose: Optional[Pose2D] = None
        self._frames: List[ObservationFrame] = []
        self._headings: List[float] = []
        self._heading_index = 0
        self._rotation_attempt = 0
        self._observe_from = 0.0
        self._turn_from: Optional[float] = None
        self._target_yaw = 0.0
        self._relook_point: Optional[RelookPoint] = None
        self._looked_yaws: List[float] = []
        self._relooks: List[dict] = []
        self._wait_until = 0.0
        self._cancel_then = None
        self._send_navigation()

    def cancel(self) -> None:
        if not self.busy:
            return
        if self._phase == Phase.CANCELING:
            if self._cancel_then is not None:
                # A timeout cancel is in progress: finish as canceled
                # instead of retrying afterwards.
                self._cancel_then = None
                self._cancel_status = VisitStatus.CANCELED
                self._cancel_reason = VisitReason.CANCELED_BY_REQUEST
                self._cancel_detail = ""
            return
        if self._handle is not None:
            self._begin_cancel(
                VisitStatus.CANCELED, VisitReason.CANCELED_BY_REQUEST, ""
            )
        else:
            # Observing: the robot is already stationary.
            self._finish(VisitStatus.CANCELED, VisitReason.CANCELED_BY_REQUEST)

    def step(self) -> Optional[VisitOutcome]:
        """Advance the visit. Returns the outcome once finished."""
        if self._phase in (Phase.NAVIGATING, Phase.ROTATING):
            self._step_motion()
        elif self._phase == Phase.OBSERVING:
            self._step_observing()
        elif self._phase == Phase.CANCELING:
            self._step_canceling()
        elif self._phase == Phase.WAITING:
            if self._clock() >= self._wait_until:
                self._send_navigation()
        return self._outcome

    # --- motion ---------------------------------------------------------

    def _next_id(self, kind: str) -> str:
        self._seq += 1
        return f"{self._request.visit_id}/{kind}/{self._seq}"

    def _send_navigation(self) -> None:
        goal = self._request.goals[self._goal_index]
        self._nav_attempts += 1
        self._handle = self._motion.navigate(
            NavigationRequest(
                request_id=self._next_id("nav"),
                goal=goal,
                map_version=self._request.map_version,
            )
        )
        self._sent_at = self._clock()
        self._phase = Phase.NAVIGATING

    def _send_rotation(self, yaw: Optional[float] = None) -> None:
        """Turn to ``yaw``, or to the current heading; a retry repeats the last turn."""
        if yaw is not None:
            self._target_yaw = yaw
        self._handle = self._motion.rotate(
            RotationRequest(
                request_id=self._next_id("rot"),
                target_yaw=self._target_yaw,
                map_version=self._request.map_version,
                yaw_tolerance=self._config.rotation_yaw_tolerance,
            )
        )
        self._sent_at = self._clock()
        if self._turn_from is None:
            self._turn_from = self._sent_at
        self._phase = Phase.ROTATING

    def _send_heading(self) -> None:
        self._turn_from = None
        self._send_rotation(self._headings[self._heading_index])

    def _step_motion(self) -> None:
        handle = self._handle
        now = self._clock()

        if handle.state == HandleState.PENDING:
            if now - self._sent_at > self._config.accept_timeout:
                self._begin_cancel(
                    VisitStatus.PAUSED,
                    VisitReason.MOTION_UNRESPONSIVE,
                    "goal 수락 응답 없음",
                )
            return

        if handle.state == HandleState.REJECTED:
            self._handle = None
            self._finish(
                VisitStatus.PAUSED,
                VisitReason.MOTION_REJECTED,
                f"{handle.request_id} 거절됨",
            )
            return

        if handle.state == HandleState.ACTIVE:
            last = handle.last_feedback_stamp
            if last is None:
                last = self._sent_at
            if now - last > self._config.feedback_timeout:
                self._begin_cancel(
                    VisitStatus.PAUSED,
                    VisitReason.MOTION_UNRESPONSIVE,
                    "feedback 끊김",
                )
                return
            navigating = self._phase == Phase.NAVIGATING
            limit = (
                self._config.navigation_timeout if navigating
                else self._config.rotation_timeout
            )
            if limit and now - self._sent_at > limit:
                timeout = MotionResult(
                    MotionStatus.FAILED, MotionReason.TIMEOUT,
                    detail=f"실행기 시간 제한 {limit:.0f}s 초과")
                on_result = (
                    self._on_navigation_result if navigating
                    else self._on_rotation_result
                )
                self._begin_cancel(
                    VisitStatus.PAUSED, VisitReason.STOP_UNCONFIRMED, "",
                    then=lambda: on_result(timeout),
                )
            return

        result = handle.result
        self._handle = None
        if result.final_pose is not None:
            self._final_pose = result.final_pose
        if not result.stopped:
            self._finish(
                VisitStatus.PAUSED,
                VisitReason.STOP_UNCONFIRMED,
                f"{handle.request_id} 결과가 정지 전에 반환됨",
            )
        elif self._phase == Phase.NAVIGATING:
            self._on_navigation_result(result)
        else:
            self._on_rotation_result(result)

    def _on_navigation_result(self, result: MotionResult) -> None:
        if result.status == MotionStatus.SUCCEEDED:
            self._reached_goal = self._request.goals[self._goal_index]
            yaw0 = (
                self._final_pose.yaw
                if self._final_pose is not None
                else self._reached_goal.yaw
            )
            from athome.execution import headings   # imports this module
            start = Pose2D(self._reached_goal.x, self._reached_goal.y, yaw0)
            order = {"sequential": headings.full, "front_first": headings.front_first}
            self._headings = order[self._config.heading_order](start, self._config.heading_count)
            self._heading_index = 0
            first = self._first_looks(start)
            self._headings[:0] = first
            self._looked_yaws = list(self._headings)
            if first:
                self._send_heading()
            else:
                self._begin_observing()
            return

        if self._pause_on_failure(result):
            return

        if (
            result.reason in _RETRY_REASONS
            and self._goal_attempt < self._config.nav_retries_per_goal
        ):
            self._goal_attempt += 1
            self._wait_until = self._clock() + self._config.retry_wait
            self._phase = Phase.WAITING
            return

        if result.reason in _RETRY_REASONS | _SKIP_GOAL_REASONS:
            self._goal_index += 1
            self._goal_attempt = 0
            if self._goal_index < len(self._request.goals):
                self._send_navigation()
            else:
                self._finish(
                    VisitStatus.FAILED,
                    VisitReason.NAVIGATION_FAILED,
                    f"모든 Goal 후보 실패 (마지막: {result.reason.value})",
                )
            return

        self._finish(
            VisitStatus.PAUSED,
            VisitReason.MOTION_INTERNAL_ERROR,
            result.detail or result.reason.value,
        )

    def _on_rotation_result(self, result: MotionResult) -> None:
        if result.status == MotionStatus.SUCCEEDED:
            self._rotation_attempt = 0
            self._begin_observing()
            return

        if self._pause_on_failure(result):
            return

        if self._rotation_attempt < self._config.rotation_retries:
            self._rotation_attempt += 1
            self._send_rotation()
            return

        where = "다시 보기" if self._relook_point is not None else f"heading {self._heading_index}"
        self._finish(
            VisitStatus.FAILED,
            VisitReason.ROTATION_FAILED,
            f"{where} 회전 실패 ({result.reason.value})",
        )

    def _pause_on_failure(self, result: MotionResult) -> bool:
        if result.status == MotionStatus.CANCELED:
            self._finish(
                VisitStatus.PAUSED,
                VisitReason.UNEXPECTED_CANCEL,
                result.detail or "요청하지 않은 취소",
            )
            return True
        if result.reason in _PAUSE_REASONS:
            self._finish(
                VisitStatus.PAUSED,
                _PAUSE_REASONS[result.reason],
                result.detail,
            )
            return True
        return False

    # --- cancel ---------------------------------------------------------

    def _begin_cancel(
        self, status: VisitStatus, reason: VisitReason, detail: str, then=None
    ) -> None:
        """Cancel the current motion. After the stop is confirmed, finish
        with ``status``, or call ``then`` to continue the visit."""
        self._cancel_then = then
        self._handle.request_cancel()
        self._cancel_status = status
        self._cancel_reason = reason
        self._cancel_detail = detail
        self._cancel_started = self._clock()
        self._phase = Phase.CANCELING

    def _step_canceling(self) -> None:
        handle = self._handle
        if handle.state == HandleState.REJECTED:
            # Never started moving.
            self._handle = None
            self._finish(
                self._cancel_status, self._cancel_reason, self._cancel_detail
            )
            return

        if handle.state == HandleState.DONE:
            result = handle.result
            self._handle = None
            if result.final_pose is not None:
                self._final_pose = result.final_pose
            if result.stopped and self._cancel_then is not None:
                then, self._cancel_then = self._cancel_then, None
                then()
            elif result.stopped:
                self._finish(
                    self._cancel_status,
                    self._cancel_reason,
                    self._cancel_detail,
                )
            else:
                self._finish(
                    VisitStatus.PAUSED,
                    VisitReason.STOP_UNCONFIRMED,
                    f"{self._cancel_reason.value}: 취소 후 정지 미확인",
                )
            return

        if self._clock() - self._cancel_started > self._config.cancel_timeout:
            # Its result may still arrive; block new motion until then.
            self._unconfirmed = handle
            self._handle = None
            self._finish(
                VisitStatus.PAUSED,
                VisitReason.STOP_UNCONFIRMED,
                f"{self._cancel_reason.value}: 취소 응답 시간 초과",
            )

    # --- observation ----------------------------------------------------

    def _begin_observing(self) -> None:
        # Only frames captured after the robot stopped at this heading count.
        self._observe_from = self._clock()
        self._phase = Phase.OBSERVING

    def _confirming(self, frames: List[ObservationFrame]) -> bool:
        """A possible target was seen at this heading but not yet in enough frames."""
        request = self._request
        if request.interest is None or request.confirm_frames <= 1:
            return False
        seen = [any(request.interest(o) for o in f.objects) for f in frames]
        hits = sum(seen)
        if not 0 < hits < request.confirm_frames:
            return False
        # A candidate seen once that the next frames do not show again was a
        # flicker: at 3 Hz, waiting out the observation timeout (meant for a
        # perception of ~7 s per frame) stopped the robot for 24 s (run 2026-11-04).
        since_last = len(seen) - 1 - max(i for i, s in enumerate(seen) if s)
        return since_last < self._config.confirm_patience_frames

    # --- re-look --------------------------------------------------------

    def _bearing(self, x: float, y: float) -> float:
        pose = self._final_pose if self._final_pose is not None else self._reached_goal
        return math.atan2(y - pose.y, x - pose.x)

    def _off_centre(self, yaw: float, yaws: Sequence[float]) -> bool:
        return all(abs(wrap_angle(yaw - y)) >= self._config.central_angle for y in yaws)

    def _first_looks(self, start: Pose2D) -> List[float]:
        looks: List[float] = []
        for x, y in self._request.look_first:
            yaw = wrap_angle(math.atan2(y - start.y, x - start.x))
            if self._off_centre(yaw, [start.yaw, *looks]):
                looks.append(yaw)
        return looks

    def _next_relook(self) -> Optional[RelookPoint]:
        """The first point no heading faces, within the visit's limit."""
        request = self._request
        if request.relook is None or len(self._relooks) >= self._config.relook_max:
            return None
        passing = []
        if self._turn_from is not None:
            passing = [f for f in self._observation.frames_since(self._turn_from)
                       if f.stamp <= self._observe_from]
        for point in request.relook(self._frames, passing):
            if self._off_centre(self._bearing(point.x, point.y), self._looked_yaws):
                return point
        return None

    def _end_relook(self, frames: List[ObservationFrame]) -> None:
        point, self._relook_point = self._relook_point, None
        passed = (self._request.relook_result(point, frames)
                  if self._request.relook_result is not None else False)
        self._relooks.append({"target": point.target, "kind": point.kind,
                              "x": round(point.x, 3), "y": round(point.y, 3),
                              "yaw": round(self._target_yaw, 4), "passed": bool(passed)})

    def _relook_waiting(self, frames: List[ObservationFrame]) -> bool:
        """A look at a point takes the frames a flicker gets, not the shortest
        wait of a heading (one frame): the point may just not be detected yet."""
        request = self._request
        if self._relook_point is None or request.interest is None:
            return False
        hits = sum(any(request.interest(o) for o in f.objects) for f in frames)
        return hits < request.confirm_frames and len(frames) < self._config.confirm_patience_frames

    # --- observation (continued) ----------------------------------------

    def _step_observing(self) -> None:
        now = self._clock()
        last = self._observation.last_stamp()
        waited = now - self._observe_from
        frames = self._observation.frames_since(self._observe_from)
        enough = (waited >= self._config.observation_window
                  and len(frames) >= self._config.observation_min_frames)
        if enough:
            if ((self._relook_waiting(frames) or self._confirming(frames))
                    and waited <= self._config.observation_timeout):
                return          # confirm a possible target before turning on
            self._frames.extend(frames)
            if self._relook_point is not None:
                self._end_relook(frames)
            else:
                self._heading_index += 1
            point = self._next_relook()
            if point is not None:
                self._relook_point = point
                yaw = wrap_angle(self._bearing(point.x, point.y))
                self._looked_yaws.append(yaw)
                self._turn_from = None
                self._send_rotation(yaw)
            elif self._request.satisfied is not None and self._request.satisfied(self._frames):
                self._finish(VisitStatus.COMPLETED)
            elif self._heading_index < len(self._headings):
                self._send_heading()
            else:
                self._finish(VisitStatus.COMPLETED)
            return

        stream_dead = (
            last is None or now - last > self._config.perception_timeout
        )
        if (stream_dead and waited > self._config.perception_timeout) or (
            waited > self._config.observation_timeout
        ):
            self._finish(
                VisitStatus.PAUSED,
                VisitReason.PERCEPTION_LOST,
                f"heading {self._heading_index} 관측 결과 없음",
            )
            return

    # --- result ---------------------------------------------------------

    def _finish(
        self,
        status: VisitStatus,
        reason: VisitReason = VisitReason.NONE,
        detail: str = "",
    ) -> None:
        self._outcome = VisitOutcome(
            visit_id=self._request.visit_id,
            status=status,
            reason=reason,
            detail=detail,
            goal=self._reached_goal,
            final_pose=self._final_pose,
            observations=list(self._frames),
            nav_attempts=self._nav_attempts,
            headings_observed=self._heading_index if self._headings else 0,
            relooks=list(self._relooks),
        )
        self._phase = Phase.DONE
