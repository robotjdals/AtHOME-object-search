"""Boundaries to the motion and perception modules.

Implemented by ROS2 adapters on the robot and by fakes in tests. The executor
polls these objects from ``step()`` and never blocks on them.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, List, Optional, Protocol

from athome.schemas import (
    MotionResult,
    NavigationRequest,
    ObservationFrame,
    RotationRequest,
)

Clock = Callable[[], float]


class HandleState(Enum):
    PENDING = "pending"    # sent, not yet accepted
    ACTIVE = "active"
    REJECTED = "rejected"
    DONE = "done"          # result available


class MotionHandle(Protocol):
    """One motion goal. A new handle is created per request, so results of
    an earlier request can never be mistaken for the current one."""

    @property
    def request_id(self) -> str: ...

    @property
    def state(self) -> HandleState: ...

    @property
    def result(self) -> Optional[MotionResult]: ...

    @property
    def last_feedback_stamp(self) -> Optional[float]: ...

    def request_cancel(self) -> None: ...


class MotionClient(Protocol):
    def navigate(self, request: NavigationRequest) -> MotionHandle: ...

    def rotate(self, request: RotationRequest) -> MotionHandle: ...


class ObservationSource(Protocol):
    def last_stamp(self) -> Optional[float]: ...

    def frames_since(self, stamp: float) -> List[ObservationFrame]:
        """Frames captured strictly after ``stamp``."""
        ...
