"""Types exchanged with the motion module (MPPI). Units: m, s, rad. Frame: map."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


class MotionStatus(Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class MotionReason(Enum):
    NONE = "none"
    INVALID_GOAL = "invalid_goal"
    MAP_MISMATCH = "map_mismatch"
    NO_PATH = "no_path"
    BLOCKED = "blocked"
    NO_PROGRESS = "no_progress"
    TIMEOUT = "timeout"
    LOCALIZATION_LOST = "localization_lost"
    INTERNAL = "internal"


@dataclass(frozen=True)
class NavigationRequest:
    request_id: str
    goal: Pose2D
    map_version: str
    # Empty when the motion module plans the global path itself.
    path: Tuple[Pose2D, ...] = ()
    frame_id: str = "map"
    # 0 means "use the motion module default".
    xy_tolerance: float = 0.0
    yaw_tolerance: float = 0.0
    timeout_sec: float = 0.0


@dataclass(frozen=True)
class RotationRequest:
    """In-place rotation for observation: reach ``target_yaw`` at the
    current position."""
    request_id: str
    target_yaw: float
    map_version: str = ""
    frame_id: str = "map"
    yaw_tolerance: float = 0.0
    timeout_sec: float = 0.0


@dataclass(frozen=True)
class MotionResult:
    status: MotionStatus
    reason: MotionReason = MotionReason.NONE
    final_pose: Optional[Pose2D] = None
    # Whether the robot was stationary when the result was reported.
    stopped: bool = True
    detail: str = ""
