"""Run a single search location visit on the robot (integration check).

    ros2 run athome_ros run_visit --ros-args -p goals:="[1.0, 0.0, 0.0]" \
        -p map_version:=map-v1

Ctrl-C cancels the motion and waits for the stop confirmation.
"""

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from athome.execution import VisitConfig, VisitExecutor, VisitRequest, VisitStatus
from athome.schemas import Pose2D
from athome_ros.adapters import RosMotionClient, RosObservationSource, node_now
from athome_ros.common import HeartbeatPublisher, TfPoseSource, spin_until_done


class VisitRunner(Node):
    def __init__(self):
        super().__init__("athome_visit_runner")
        goals = self.declare_parameter("goals", [0.0, 0.0, 0.0]).value
        self._map_version = self.declare_parameter("map_version", "").value
        nav_action = self.declare_parameter(
            "navigate_action", "/athome/navigate_to_goal").value
        objects_topic = self.declare_parameter(
            "objects_topic", "/athome/perception/objects").value
        heartbeat_topic = self.declare_parameter(
            "heartbeat_topic", "/athome/executor/heartbeat").value
        rate = self.declare_parameter("rate_hz", 20.0).value

        if len(goals) % 3 != 0 or not goals:
            raise ValueError("goals는 [x, y, yaw, ...] 형식이어야 함")
        self._goals = tuple(
            Pose2D(*goals[i:i + 3]) for i in range(0, len(goals), 3)
        )

        self._motion = RosMotionClient(self, nav_action, TfPoseSource(self))
        observation = RosObservationSource(self, objects_topic)
        self._visit = VisitExecutor(
            self._motion, observation, lambda: node_now(self), VisitConfig()
        )
        self.started = False
        self.outcome = None

        HeartbeatPublisher(self, heartbeat_topic)
        self.create_timer(1.0 / rate, self._tick)

    def _tick(self) -> None:
        if self.outcome is not None:
            return
        if not self.started:
            if not self._motion.servers_ready():
                self.get_logger().info(
                    "주행 Action 서버 대기 중", throttle_duration_sec=5.0)
                return
            self._visit.start(
                VisitRequest("visit", self._goals, self._map_version))
            self.started = True
            self.get_logger().info(f"방문 시작: {self._goals}")
            return

        outcome = self._visit.step()
        if outcome is None:
            return
        self.outcome = outcome
        n_obj = sum(len(f.objects) for f in outcome.observations)
        message = (
            f"방문 종료: {outcome.status.value} ({outcome.reason.value}) "
            f"{outcome.detail} | 주행 시도 {outcome.nav_attempts}, "
            f"관측 프레임 {len(outcome.observations)}, 객체 {n_obj}, "
            f"최종 pose {outcome.final_pose}"
        )
        # rclpy forbids changing severity at one call site: separate calls.
        if outcome.status == VisitStatus.COMPLETED:
            self.get_logger().info(message)
        else:
            self.get_logger().warn(message)


def main(args=None):
    # Keep the context alive on Ctrl-C so the cancel can still be sent.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = VisitRunner()
    try:
        spin_until_done(
            node,
            cancel=node._visit.cancel,
            is_started=lambda: node.started,
            is_done=lambda: node.outcome is not None,
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
