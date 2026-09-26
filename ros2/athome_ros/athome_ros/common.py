"""Pieces shared by the executor nodes."""

from typing import Optional

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener

from athome.schemas import Pose2D
from athome_ros.conversions import stamp_to_sec, yaw_from_quaternion


class HeartbeatPublisher:
    """Tells the motion module the executor is alive (it stops otherwise)."""

    def __init__(self, node: Node, topic: str, period: float = 0.5):
        self._node = node
        self._pub = node.create_publisher(Header, topic, 10)
        self.publish()
        # Own group: keeps beating while the executor callbacks are busy.
        node.create_timer(period, self.publish,
                          callback_group=MutuallyExclusiveCallbackGroup())

    def publish(self) -> None:
        msg = Header()
        msg.stamp = self._node.get_clock().now().to_msg()
        self._pub.publish(msg)


class TfPoseSource:
    """Robot pose from TF. None when unavailable or stale."""

    def __init__(
        self,
        node: Node,
        map_frame: str = "map",
        base_frame: str = "base_link",
        max_age: float = 0.5,
    ):
        self._node = node
        self._map = map_frame
        self._base = base_frame
        self._max_age = max_age
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, node)

    def current_pose(self) -> Optional[Pose2D]:
        try:
            t = self._buffer.lookup_transform(self._map, self._base, Time())
        except TransformException:
            return None
        now = self._node.get_clock().now().nanoseconds * 1e-9
        if now - stamp_to_sec(t.header.stamp) > self._max_age:
            return None
        tr = t.transform
        return Pose2D(tr.translation.x, tr.translation.y,
                      yaw_from_quaternion(tr.rotation))


def spin_until_done(node, cancel, is_started, is_done) -> None:
    """Spin until finished. First Ctrl-C cancels and waits for the stop
    confirmation, a second one exits immediately.

    Requires ``rclpy.init(signal_handler_options=SignalHandlerOptions.NO)``
    so the context survives the first Ctrl-C.
    """
    try:
        try:
            while not is_done():
                rclpy.spin_once(node, timeout_sec=0.1)
        except KeyboardInterrupt:
            if not is_started():
                return
            node.get_logger().warn("취소 요청, 정지 확인 대기 (다시 Ctrl-C: 강제 종료)")
            cancel()
            while not is_done():
                rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().error("정지 확인 없이 강제 종료")
