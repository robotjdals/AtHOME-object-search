"""Minimal robot simulator implementing the motion, perception and pose
interfaces. Motion is a straight line (obstacles ignored: the planner has
already checked reachability); perception reports world objects within
range and camera FOV.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence

from athome.execution.fake import FakeClock, FakeObservationSource
from athome.execution.interfaces import HandleState
from athome.execution.visit import wrap_angle
from athome.schemas import (
    MotionReason,
    MotionResult,
    MotionStatus,
    NavigationRequest,
    ObservedObject,
    Pose2D,
    RotationRequest,
)


class SimHandle:
    def __init__(self, request, clock):
        self.request = request
        self.request_id = request.request_id
        self.state = HandleState.PENDING
        self.result: Optional[MotionResult] = None
        self.last_feedback_stamp: Optional[float] = None
        self.cancel_requested = False
        self._clock = clock

    def request_cancel(self):
        self.cancel_requested = True


class SimRobot:
    def __init__(
        self,
        clock: FakeClock,
        world: Sequence[tuple],
        start: Pose2D,
        speed: float = 0.5,
        turn_rate: float = 1.0,
        view_range: float = 2.0,
        fov: float = math.radians(87),
        perception_rate: float = 10.0,
        # goal -> failure reason or None, to inject navigation failures.
        navigation_failure: Callable[[Pose2D], Optional[MotionReason]] = None,
    ):
        self.clock = clock
        self.world = list(world)
        self.pose = start
        self.speed = speed
        self.turn_rate = turn_rate
        self.view_range = view_range
        self.fov = fov
        self.perception = FakeObservationSource()
        self._period = 1.0 / perception_rate
        self._next_frame = clock()
        self._navigation_failure = navigation_failure or (lambda goal: None)
        self._active: Optional[SimHandle] = None
        self.handles: List[SimHandle] = []
        self.distance_traveled = 0.0

    # MotionClient
    def navigate(self, request: NavigationRequest) -> SimHandle:
        return self._submit(request)

    def rotate(self, request: RotationRequest) -> SimHandle:
        return self._submit(request)

    # ObservationSource
    def last_stamp(self):
        return self.perception.last_stamp()

    def frames_since(self, stamp):
        return self.perception.frames_since(stamp)

    # PoseSource
    def current_pose(self) -> Optional[Pose2D]:
        return self.pose

    def _submit(self, request) -> SimHandle:
        handle = SimHandle(request, self.clock)
        self.handles.append(handle)
        if self._active is not None:
            handle.state = HandleState.REJECTED
        else:
            handle.state = HandleState.ACTIVE
            handle.last_feedback_stamp = self.clock()
            self._active = handle
        return handle

    def update(self, dt: float) -> None:
        now = self.clock.advance(dt)
        if self._active is not None:
            self._move(self._active, dt, now)
        while self._next_frame <= now:
            self.perception.publish(self._next_frame, self._visible())
            self._next_frame += self._period

    def _finish(self, handle, status, reason=MotionReason.NONE):
        handle.result = MotionResult(status, reason, final_pose=self.pose, stopped=True)
        handle.state = HandleState.DONE
        self._active = None

    def _move(self, handle: SimHandle, dt: float, now: float) -> None:
        handle.last_feedback_stamp = now
        if handle.cancel_requested:
            self._finish(handle, MotionStatus.CANCELED)
            return
        req = handle.request
        x, y, yaw = self.pose.x, self.pose.y, self.pose.yaw

        if isinstance(req, NavigationRequest):
            reason = self._navigation_failure(req.goal)
            if reason is not None:
                self._finish(handle, MotionStatus.FAILED, reason)
                return
            dx, dy = req.goal.x - x, req.goal.y - y
            dist = math.hypot(dx, dy)
            step = min(dist, self.speed * dt)
            if dist > 0:
                x += dx / dist * step
                y += dy / dist * step
                self.distance_traveled += step
            target_yaw = req.goal.yaw
        else:
            dist = step = 0.0
            target_yaw = req.target_yaw

        dyaw = wrap_angle(target_yaw - yaw)
        yaw = wrap_angle(yaw + max(-self.turn_rate * dt, min(self.turn_rate * dt, dyaw)))
        self.pose = Pose2D(x, y, yaw)
        if dist - step <= 1e-6 and abs(wrap_angle(target_yaw - yaw)) <= 0.02:
            self._finish(handle, MotionStatus.SUCCEEDED)

    def _visible(self) -> List[ObservedObject]:
        return visible_objects(self.world, self.pose, self.view_range, self.fov)


def visible_objects(
    world: Sequence[tuple], pose: Pose2D, view_range: float, fov: float
) -> List[ObservedObject]:
    """World objects (object_id, category, centroid) inside the camera view."""
    out = []
    for i, (_, category, (ox, oy, oz)) in enumerate(world):
        dx, dy = ox - pose.x, oy - pose.y
        if math.hypot(dx, dy) > view_range:
            continue
        if abs(wrap_angle(math.atan2(dy, dx) - pose.yaw)) > fov / 2:
            continue
        out.append(ObservedObject(
            object_id=i, label=category, confidence=0.9, centroid=(ox, oy, oz)))
    return out
