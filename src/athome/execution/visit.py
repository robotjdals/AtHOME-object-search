"""Execute one search location visit: navigate, then observe in 4 headings.

The executor only reports what happened. Whether the location becomes
Visited is decided by the search layer (only on COMPLETED).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

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
class VisitRequest:
    visit_id: str
    # Goal pose candidates of one search location, in priority order.
    goals: Tuple[Pose2D, ...]
    map_version: str


@dataclass(frozen=True)
class VisitConfig:
    nav_retries_per_goal: int = 1
    rotation_retries: int = 1
    heading_count: int = 4
    # Includes path planning when the adapter plans before sending.
    accept_timeout: float = 5.0
    feedback_timeout: float = 1.0
    cancel_timeout: float = 5.0
    observation_window: float = 1.0
    observation_min_frames: int = 3
    observation_timeout: float = 8.0
    # Perception gaps shorter than this are waited out.
    perception_timeout: float = 3.0
    # Before retrying the same goal (a person in the way often moves on).
    retry_wait: float = 3.0
    # Executor-side limits, in case the motion module never finishes.
    # 0 disables. Treated as a TIMEOUT failure (retry rules apply).
    navigation_timeout: float = 180.0
    rotation_timeout: float = 30.0


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

    def _send_rotation(self) -> None:
        self._handle = self._motion.rotate(
            RotationRequest(
                request_id=self._next_id("rot"),
                target_yaw=self._headings[self._heading_index],
                map_version=self._request.map_version,
            )
        )
        self._sent_at = self._clock()
        self._phase = Phase.ROTATING

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
            n = self._config.heading_count
            self._headings = [
                wrap_angle(yaw0 + 2 * math.pi * k / n) for k in range(n)
            ]
            self._heading_index = 0
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

        self._finish(
            VisitStatus.FAILED,
            VisitReason.ROTATION_FAILED,
            f"heading {self._heading_index} 회전 실패 ({result.reason.value})",
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

    def _step_observing(self) -> None:
        now = self._clock()
        last = self._observation.last_stamp()
        waited = now - self._observe_from

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

        frames = self._observation.frames_since(self._observe_from)
        if (
            waited < self._config.observation_window
            or len(frames) < self._config.observation_min_frames
        ):
            return

        self._frames.extend(frames)
        self._heading_index += 1
        if self._heading_index < len(self._headings):
            self._send_rotation()
        else:
            self._finish(VisitStatus.COMPLETED)

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
        )
        self._phase = Phase.DONE
