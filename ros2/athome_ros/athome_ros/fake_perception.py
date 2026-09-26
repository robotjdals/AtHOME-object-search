"""Stand-in for the perception module: publishes ObservedObjectArray.

world:=""     fixed ``labels`` every frame
world:=toy    toy environment objects inside the camera view (robot pose
              from TF); ``moved`` entries "object_id=x,y,z" relocate objects

``enabled`` can be set to false at runtime to simulate a perception outage.
"""

import math

import rclpy
from rclpy.node import Node

from athome.schemas import ObservedObject as CoreObject
from athome.testing.sim import visible_objects
from athome_interfaces.msg import ObservedObject, ObservedObjectArray
from athome_ros.common import TfPoseSource


class FakePerception(Node):
    def __init__(self):
        super().__init__("athome_fake_perception")
        self._labels = self.declare_parameter("labels", ["table", "cup"]).value
        world = self.declare_parameter("world", "").value
        moved = self.declare_parameter("moved", [""]).value
        self._range = self.declare_parameter("view_range", 2.0).value
        self._fov = math.radians(self.declare_parameter("fov_deg", 87.0).value)
        self.declare_parameter("enabled", True)
        rate = self.declare_parameter("rate_hz", 10.0).value

        self._world = None
        if world == "toy":
            from athome.testing import toy_env

            self._world = toy_env.toy_world(_parse_moved(moved))
            self._pose = TfPoseSource(self)
        elif world:
            raise ValueError(f"알 수 없는 world: {world}")

        self._pub = self.create_publisher(
            ObservedObjectArray, "/athome/perception/objects", 10)
        self.create_timer(1.0 / rate, self._publish)

    def _objects(self):
        if self._world is None:
            return [
                CoreObject(i, label, 0.9, (float(i), 0.0, 0.0))
                for i, label in enumerate(self._labels)
            ]
        pose = self._pose.current_pose()
        if pose is None:
            return None
        return visible_objects(self._world, pose, self._range, self._fov)

    def _publish(self):
        if not self.get_parameter("enabled").value:
            return
        objects = self._objects()
        if objects is None:
            return
        msg = ObservedObjectArray()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.clip_model = "fake"
        for obj in objects:
            o = ObservedObject()
            o.object_id = obj.object_id
            o.is_static = True
            o.label = obj.label
            o.confidence = obj.confidence
            o.centroid.x, o.centroid.y, o.centroid.z = obj.centroid
            o.bbox_center.position = o.centroid
            o.bbox_center.orientation.w = 1.0
            o.bbox_size.x = o.bbox_size.y = o.bbox_size.z = 0.3
            msg.objects.append(o)
        self._pub.publish(msg)


def _parse_moved(items):
    moved = {}
    for item in items:
        if not item.strip():
            continue
        oid, xyz = item.split("=")
        moved[oid] = tuple(float(v) for v in xyz.split(","))
    return moved


def main(args=None):
    rclpy.init(args=args)
    node = FakePerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
