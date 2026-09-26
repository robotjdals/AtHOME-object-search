"""ROS2 implementations of athome.execution interfaces.

All callbacks run on the node's executor thread, the same thread that calls
``VisitExecutor.step()``, so no locking is needed with a single-threaded
executor.
"""

from collections import deque
from typing import List, Optional

from action_msgs.msg import GoalStatus
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from rclpy.node import Node

from athome.execution import HandleState
from athome.schemas import (
    MotionReason,
    MotionResult,
    MotionStatus,
    NavigationRequest,
    ObservationFrame,
    Pose2D,
    RotationRequest,
)
from athome_interfaces.action import NavigateToGoal
from athome_interfaces.msg import ObservedObjectArray
from athome_ros.conversions import (
    motion_result_from_msg,
    observation_frame_from_msg,
    pose2d_to_msg,
)


def node_now(node: Node) -> float:
    return node.get_clock().now().nanoseconds * 1e-9


class RosMotionHandle:
    def __init__(self, node: Node, client: ActionClient, goal_msg, request_id):
        self._node = node
        self.request_id = request_id
        self.state = HandleState.PENDING
        self.result = None
        self.last_feedback_stamp: Optional[float] = None
        self._goal_handle = None
        self._cancel_pending = False
        future = client.send_goal_async(
            goal_msg, feedback_callback=self._on_feedback
        )
        future.add_done_callback(self._on_goal_response)

    def request_cancel(self) -> None:
        if self._goal_handle is not None and self.state == HandleState.ACTIVE:
            self._goal_handle.cancel_goal_async()
        else:
            # Not accepted yet: cancel as soon as it is.
            self._cancel_pending = True

    def _on_goal_response(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.state = HandleState.REJECTED
            return
        self._goal_handle = goal_handle
        self.state = HandleState.ACTIVE
        self.last_feedback_stamp = node_now(self._node)
        goal_handle.get_result_async().add_done_callback(self._on_result)
        if self._cancel_pending:
            goal_handle.cancel_goal_async()

    def _on_feedback(self, _msg) -> None:
        self.last_feedback_stamp = node_now(self._node)

    def _on_result(self, future) -> None:
        response = future.result()
        self.result = motion_result_from_msg(response.status, response.result)
        self.state = HandleState.DONE


class PlannedMotionHandle:
    """Plans a path with the Nav2 planner, then sends the navigation goal
    with that path. PENDING while planning; a planning failure is reported as
    FAILED/NO_PATH (the robot has not moved)."""

    def __init__(self, node, planner: ActionClient, nav: ActionClient,
                 goal_msg, request_id, planner_id: str):
        self._node = node
        self._nav = nav
        self._goal_msg = goal_msg
        self.request_id = request_id
        self._state = HandleState.PENDING
        self._result = None
        self._inner: Optional[RosMotionHandle] = None
        plan = ComputePathToPose.Goal()
        plan.goal = goal_msg.goal
        plan.planner_id = planner_id
        plan.use_start = False          # start from the current TF pose
        planner.send_goal_async(plan).add_done_callback(self._on_plan_accepted)

    @property
    def state(self):
        return self._inner.state if self._inner else self._state

    @property
    def result(self):
        return self._inner.result if self._inner else self._result

    @property
    def last_feedback_stamp(self):
        return self._inner.last_feedback_stamp if self._inner else None

    def request_cancel(self) -> None:
        if self._inner is not None:
            self._inner.request_cancel()
        elif self._state == HandleState.PENDING:
            # Nothing was sent to the motion module yet.
            self._finish(MotionStatus.CANCELED, MotionReason.NONE, "경로 계획 중 취소")

    def _finish(self, status, reason, detail) -> None:
        self._result = MotionResult(status, reason, stopped=True, detail=detail)
        self._state = HandleState.DONE

    def _on_plan_accepted(self, future) -> None:
        handle = future.result()
        if self._state != HandleState.PENDING:
            return
        if handle is None or not handle.accepted:
            self._finish(MotionStatus.FAILED, MotionReason.NO_PATH, "planner가 요청 거절")
            return
        handle.get_result_async().add_done_callback(self._on_plan)

    def _on_plan(self, future) -> None:
        if self._state != HandleState.PENDING:
            return                      # canceled while planning
        response = future.result()
        path = response.result.path
        if response.status != GoalStatus.STATUS_SUCCEEDED or not path.poses:
            self._finish(MotionStatus.FAILED, MotionReason.NO_PATH, "Nav2 경로 없음")
            return
        self._goal_msg.path = path
        self._inner = RosMotionHandle(self._node, self._nav, self._goal_msg, self.request_id)


class _DoneHandle:
    """Request that failed before anything was sent (robot did not move)."""

    def __init__(self, request_id, reason, detail):
        self.request_id = request_id
        self.state = HandleState.DONE
        self.result = MotionResult(MotionStatus.FAILED, reason, stopped=True, detail=detail)
        self.last_feedback_stamp = None

    def request_cancel(self) -> None:
        pass


class RosMotionClient:
    """Both navigation and in-place rotation go to the single
    NavigateToGoal action of the motion module."""

    def __init__(self, node: Node, navigate_action: str, pose_source,
                 planner_action: Optional[str] = None, planner_id: str = "GridBased"):
        self._node = node
        self._nav = ActionClient(node, NavigateToGoal, navigate_action)
        self._pose = pose_source
        self._planner = (
            ActionClient(node, ComputePathToPose, planner_action) if planner_action else None)
        self._planner_id = planner_id

    def servers_ready(self) -> bool:
        return self._nav.server_is_ready() and (
            self._planner is None or self._planner.server_is_ready())

    def navigate(self, request: NavigationRequest) -> RosMotionHandle:
        stamp = self._node.get_clock().now().to_msg()
        goal = NavigateToGoal.Goal()
        goal.request_id = request.request_id
        goal.goal = pose2d_to_msg(request.goal, request.frame_id, stamp)
        goal.path.header.frame_id = request.frame_id
        goal.path.header.stamp = stamp
        goal.path.poses = [
            pose2d_to_msg(p, request.frame_id, stamp) for p in request.path
        ]
        goal.map_version = request.map_version
        goal.xy_tolerance = float(request.xy_tolerance)
        goal.yaw_tolerance = float(request.yaw_tolerance)
        goal.timeout_sec = float(request.timeout_sec)
        if self._planner is not None:
            return PlannedMotionHandle(self._node, self._planner, self._nav, goal,
                                       request.request_id, self._planner_id)
        return RosMotionHandle(self._node, self._nav, goal, request.request_id)

    def rotate(self, request: RotationRequest):
        """Pose goal at the current position with the target yaw; the path
        is that single pose (no planner needed)."""
        pose = self._pose.current_pose()
        if pose is None:
            return _DoneHandle(request.request_id, MotionReason.LOCALIZATION_LOST,
                               "회전 요청 시 현재 위치 없음")
        stamp = self._node.get_clock().now().to_msg()
        target = Pose2D(pose.x, pose.y, request.target_yaw)
        goal = NavigateToGoal.Goal()
        goal.request_id = request.request_id
        goal.goal = pose2d_to_msg(target, request.frame_id, stamp)
        goal.path.header.frame_id = request.frame_id
        goal.path.header.stamp = stamp
        goal.path.poses = [pose2d_to_msg(target, request.frame_id, stamp)]
        goal.map_version = request.map_version
        goal.yaw_tolerance = float(request.yaw_tolerance)
        goal.timeout_sec = float(request.timeout_sec)
        return RosMotionHandle(self._node, self._nav, goal, request.request_id)


class RosObservationSource:
    def __init__(
        self,
        node: Node,
        topic: str,
        frame_id: str = "map",
        buffer_sec: float = 30.0,
    ):
        self._node = node
        self._frame_id = frame_id
        self._buffer_sec = buffer_sec
        self._frames = deque()
        self._last: Optional[float] = None
        node.create_subscription(ObservedObjectArray, topic, self._on_msg, 10)

    def last_stamp(self) -> Optional[float]:
        return self._last

    def frames_since(self, stamp: float) -> List[ObservationFrame]:
        return [f for f in self._frames if f.stamp > stamp]

    def _on_msg(self, msg: ObservedObjectArray) -> None:
        if msg.header.frame_id != self._frame_id:
            self._node.get_logger().warn(
                f"관측 frame_id '{msg.header.frame_id}' != '{self._frame_id}', 무시",
                throttle_duration_sec=5.0,
            )
            return
        frame = observation_frame_from_msg(msg)
        self._frames.append(frame)
        if self._last is None or frame.stamp > self._last:
            self._last = frame.stamp
        while self._frames and self._frames[0].stamp < self._last - self._buffer_sec:
            self._frames.popleft()
