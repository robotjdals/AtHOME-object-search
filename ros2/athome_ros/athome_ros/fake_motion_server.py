"""Stand-in for the MPPI node: serves NavigateToGoal (navigation and, with a
single-pose path, in-place rotation for observation).

Moves a virtual pose in straight lines. Also a reference for the server-side
rules: reject while busy, stop before reporting a result, abort when the
executor heartbeat is lost.

Failure injection (can be changed at runtime with ``ros2 param set``):
  nav_fail_reason                       REASON_* code, 0 = no failure
  feedback_enabled                      false simulates a silent server
  reject_cancel                         true simulates an unconfirmed stop
"""

import math
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

from athome.config import load_robot_config
from athome.execution.visit import wrap_angle
from athome.schemas import Pose2D
from athome_interfaces.action import NavigateToGoal
from athome_ros.conversions import pose2d_to_msg, yaw_from_quaternion

SUCCEEDED, FAILED, CANCELED = 1, 2, 3
REASON_MAP_MISMATCH, REASON_TIMEOUT, REASON_INTERNAL = 2, 6, 8


class FakeMotionServer(Node):
    def __init__(self):
        super().__init__("athome_fake_motion")
        x, y, yaw = self.declare_parameter("initial_pose", [0.0, 0.0, 0.0]).value
        self._pose = [x, y, yaw]
        self._map_version = self.declare_parameter("map_version", "map-v1").value
        config = self.declare_parameter("robot_config", "").value
        if config:
            # Same map check as the real motion module should do.
            self._map_version = load_robot_config(config).map_version
        self._v = self.declare_parameter("linear_speed", 0.5).value
        self._w = self.declare_parameter("angular_speed", 1.0).value
        self._dt = 1.0 / self.declare_parameter("rate_hz", 20.0).value
        self._hb_timeout = self.declare_parameter("heartbeat_timeout", 2.0).value
        for name, default in (
            ("nav_fail_reason", 0),
            ("feedback_enabled", True),
            ("reject_cancel", False),
        ):
            self.declare_parameter(name, default)

        self._lock = threading.Lock()
        self._busy = False
        self._last_hb = None
        group = ReentrantCallbackGroup()
        self.create_subscription(
            Header, "/athome/executor/heartbeat", self._on_heartbeat, 10,
            callback_group=group)
        common = dict(
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=group,
        )
        ActionServer(self, NavigateToGoal, "/athome/navigate_to_goal",
                     execute_callback=self._execute_nav, **common)
        # Stands in for localization: map -> base_link.
        self._tf = TransformBroadcaster(self)
        self.create_timer(0.05, self._publish_tf, callback_group=group)

    def _param(self, name):
        return self.get_parameter(name).value

    def _publish_tf(self):
        x, y, yaw = self._pose
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.rotation.z = math.sin(yaw / 2)
        t.transform.rotation.w = math.cos(yaw / 2)
        self._tf.sendTransform(t)

    def _on_heartbeat(self, _msg):
        self._last_hb = time.monotonic()

    def _on_goal(self, request):
        with self._lock:
            if self._busy:
                self.get_logger().warn(f"{request.request_id}: 실행 중이라 거절")
                return GoalResponse.REJECT
            self._busy = True
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle):
        if self._param("reject_cancel"):
            return CancelResponse.REJECT
        return CancelResponse.ACCEPT

    def _pose_msg(self):
        return pose2d_to_msg(Pose2D(*self._pose), "map",
                             self.get_clock().now().to_msg())

    def _finish(self, goal_handle, result, status, reason=0, detail=""):
        # Simulate braking, then report with the robot stationary.
        time.sleep(0.2)
        result.status = status
        result.reason = reason
        result.detail = detail
        result.final_pose = self._pose_msg()
        result.stopped = True
        if status == SUCCEEDED:
            goal_handle.succeed()
        elif status == CANCELED:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        with self._lock:
            self._busy = False
        self.get_logger().info(
            f"{goal_handle.request.request_id}: status={status} reason={reason} {detail}")
        return result

    def _check_interrupts(self, goal_handle, result, started):
        if goal_handle.is_cancel_requested:
            return self._finish(goal_handle, result, CANCELED)
        last_hb = started if self._last_hb is None else max(self._last_hb, started)
        if time.monotonic() - last_hb > self._hb_timeout:
            return self._finish(goal_handle, result, FAILED, REASON_INTERNAL,
                                "executor heartbeat 끊김, 정지")
        timeout = goal_handle.request.timeout_sec or 120.0
        if time.monotonic() - started > timeout:
            return self._finish(goal_handle, result, FAILED, REASON_TIMEOUT)
        return None

    def _execute_nav(self, goal_handle):
        req = goal_handle.request
        result = NavigateToGoal.Result()
        if req.map_version != self._map_version:
            return self._finish(goal_handle, result, FAILED, REASON_MAP_MISMATCH,
                                f"{req.map_version} != {self._map_version}")
        gx = req.goal.pose.position.x
        gy = req.goal.pose.position.y
        gyaw = yaw_from_quaternion(req.goal.pose.orientation)
        tol = req.xy_tolerance or 0.05
        started = time.monotonic()
        # Follow the given global path (as MPPI tracks it), then the goal.
        waypoints = [(p.pose.position.x, p.pose.position.y) for p in req.path.poses]
        waypoints.append((gx, gy))
        self.get_logger().info(f"{req.request_id}: 경로 {len(req.path.poses)}점 수신")

        while True:
            done = self._check_interrupts(goal_handle, result, started)
            if done is not None:
                return done
            fail = self._param("nav_fail_reason")
            if fail and time.monotonic() - started > 1.0:
                return self._finish(goal_handle, result, FAILED, fail, "주입된 실패")

            while len(waypoints) > 1 and math.hypot(
                    waypoints[0][0] - self._pose[0], waypoints[0][1] - self._pose[1]) < 0.1:
                waypoints.pop(0)
            tx, ty = waypoints[0]
            dx, dy = tx - self._pose[0], ty - self._pose[1]
            dist = math.hypot(gx - self._pose[0], gy - self._pose[1])
            seg = math.hypot(dx, dy)
            dyaw = wrap_angle(gyaw - self._pose[2])
            if dist <= tol and abs(dyaw) <= 0.05:
                return self._finish(goal_handle, result, SUCCEEDED)
            step = min(seg, self._v * self._dt)
            if seg > 0:
                self._pose[0] += dx / seg * step
                self._pose[1] += dy / seg * step
            self._pose[2] = wrap_angle(
                self._pose[2] + max(-self._w * self._dt, min(self._w * self._dt, dyaw)))

            if self._param("feedback_enabled"):
                fb = NavigateToGoal.Feedback()
                fb.current_pose = self._pose_msg()
                fb.remaining_distance = float(dist)
                goal_handle.publish_feedback(fb)
            time.sleep(self._dt)


def main(args=None):
    rclpy.init(args=args)
    node = FakeMotionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
