"""Manually driven fakes of the motion and perception modules for tests."""

from __future__ import annotations

from typing import List, Optional, Sequence

from athome.execution.interfaces import HandleState
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


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> float:
        self.now += dt
        return self.now


class FakeMotionHandle:
    def __init__(self, request, clock: FakeClock):
        self.request = request
        self._clock = clock
        self.state = HandleState.PENDING
        self.result: Optional[MotionResult] = None
        self.last_feedback_stamp: Optional[float] = None
        self.cancel_requested = False

    @property
    def request_id(self) -> str:
        return self.request.request_id

    def request_cancel(self) -> None:
        self.cancel_requested = True

    # --- driven by the test ---

    def accept(self) -> None:
        self.state = HandleState.ACTIVE
        self.last_feedback_stamp = self._clock()

    def reject(self) -> None:
        self.state = HandleState.REJECTED

    def feedback(self) -> None:
        self.last_feedback_stamp = self._clock()

    def succeed(self, final_pose: Optional[Pose2D] = None, stopped=True):
        self._finish(MotionResult(MotionStatus.SUCCEEDED, final_pose=final_pose,
                                  stopped=stopped))

    def fail(self, reason: MotionReason, stopped=True, detail=""):
        self._finish(MotionResult(MotionStatus.FAILED, reason, stopped=stopped,
                                  detail=detail))

    def finish_canceled(self, stopped=True):
        self._finish(MotionResult(MotionStatus.CANCELED, stopped=stopped))

    def _finish(self, result: MotionResult) -> None:
        if self.state in (HandleState.DONE, HandleState.REJECTED):
            return
        self.state = HandleState.DONE
        self.result = result


class FakeMotionClient:
    def __init__(self, clock: FakeClock):
        self._clock = clock
        self.handles: List[FakeMotionHandle] = []

    def navigate(self, request: NavigationRequest) -> FakeMotionHandle:
        return self._add(request)

    def rotate(self, request: RotationRequest) -> FakeMotionHandle:
        return self._add(request)

    @property
    def last(self) -> FakeMotionHandle:
        return self.handles[-1]

    def _add(self, request) -> FakeMotionHandle:
        handle = FakeMotionHandle(request, self._clock)
        self.handles.append(handle)
        return handle


class FakeObservationSource:
    def __init__(self):
        self.frames: List[ObservationFrame] = []

    def publish(
        self, stamp: float, objects: Sequence[ObservedObject] = ()
    ) -> None:
        self.frames.append(ObservationFrame(stamp, tuple(objects)))

    def last_stamp(self) -> Optional[float]:
        return max((f.stamp for f in self.frames), default=None)

    def frames_since(self, stamp: float) -> List[ObservationFrame]:
        return [f for f in self.frames if f.stamp > stamp]
