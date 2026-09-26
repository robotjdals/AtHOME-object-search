import math

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped

from athome.schemas import (
    MotionReason,
    MotionResult,
    MotionStatus,
    ObservationFrame,
    ObservedObject,
    Pose2D,
)

_STATUS = {
    1: MotionStatus.SUCCEEDED,
    2: MotionStatus.FAILED,
    3: MotionStatus.CANCELED,
}
# NavigateToGoal result reasons (motion module side). Unknown codes,
# including the reserved 1 and 3, are read as INTERNAL.
_REASON = {
    0: MotionReason.NONE,
    2: MotionReason.MAP_MISMATCH,
    4: MotionReason.BLOCKED,
    5: MotionReason.NO_PROGRESS,
    6: MotionReason.TIMEOUT,
    7: MotionReason.LOCALIZATION_LOST,
    8: MotionReason.INTERNAL,
}


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def yaw_from_quaternion(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def pose2d_from_msg(msg: PoseStamped):
    if not msg.header.frame_id:
        return None
    p = msg.pose
    return Pose2D(p.position.x, p.position.y, yaw_from_quaternion(p.orientation))


def pose2d_to_msg(pose: Pose2D, frame_id: str, stamp) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.pose.position.x = float(pose.x)
    msg.pose.position.y = float(pose.y)
    msg.pose.orientation.z = math.sin(pose.yaw / 2)
    msg.pose.orientation.w = math.cos(pose.yaw / 2)
    return msg


def motion_result_from_msg(goal_status: int, msg) -> MotionResult:
    status = _STATUS.get(msg.status)
    detail = msg.detail
    if status is None:
        # Server returned without filling the result (crash, abort, ...).
        status = (
            MotionStatus.CANCELED
            if goal_status == GoalStatus.STATUS_CANCELED
            else MotionStatus.FAILED
        )
        detail = detail or f"result status 미기입 (goal status {goal_status})"
    return MotionResult(
        status=status,
        reason=_REASON.get(msg.reason, MotionReason.INTERNAL),
        final_pose=pose2d_from_msg(msg.final_pose),
        # Unfilled results default to False, which is the safe reading.
        stopped=bool(msg.stopped),
        detail=detail,
    )


def observation_frame_from_msg(msg) -> ObservationFrame:
    objects = []
    for o in msg.objects:
        c = o.centroid
        b = o.bbox_center
        s = o.bbox_size
        objects.append(
            ObservedObject(
                object_id=int(o.object_id),
                label=o.label,
                confidence=float(o.confidence),
                centroid=(c.x, c.y, c.z),
                is_static=bool(o.is_static),
                bbox_center=(b.position.x, b.position.y, b.position.z),
                bbox_size=(s.x, s.y, s.z),
                bbox_yaw=yaw_from_quaternion(b.orientation),
                clip_feature=tuple(o.clip_feature) or None,
            )
        )
    return ObservationFrame(stamp_to_sec(msg.header.stamp), tuple(objects))
