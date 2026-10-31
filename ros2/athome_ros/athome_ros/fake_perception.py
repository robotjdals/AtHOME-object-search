"""Stand-in for the perception module: publishes ObservedObjectArray.

world:=""     fixed ``labels`` every frame
world:=toy    toy environment objects inside the camera view (robot pose
              from TF); ``moved`` entries "object_id=x,y,z" relocate objects

object_id: the pre-map ID ("<label>_<id>" -> id) as perception's static
association gives it; -1 for moved and added objects, which perception
cannot associate with the pre-map (its association is geometric too).
world:=<path> objects of a perception Static Feature export (objects.json or
              its scene directory) as the world: a real pre-mapped scene with
              the robot and perception simulated (e.g. the lab run4 data);
              ``moved`` relocates them as well, ``added`` entries
              "label=x,y,z" place objects the pre-map does not have
visibility:=<path>  HM3D episode export (scripts/export_hm3d_robot_episodes.py):
              targets visible from each goal cell under the training
              observation model, published within the camera view at those
              cells only (the robot observes only there)

``enabled`` can be set to false at runtime to simulate a perception outage.
"""

import math
from dataclasses import replace

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
        added = self.declare_parameter("added", [""]).value
        self._range = self.declare_parameter("view_range", 2.0).value
        self._fov = math.radians(self.declare_parameter("fov_deg", 87.0).value)
        self.declare_parameter("enabled", True)
        rate = self.declare_parameter("rate_hz", 10.0).value

        visibility = self.declare_parameter("visibility", "").value

        self._world = None
        self._cells = None
        self._unlinked = set()
        if visibility:
            import json

            table = json.loads(open(visibility, encoding="utf-8").read())
            self._world = [(oid, o["label"], tuple(o["centroid"])) for oid, o in table["objects"].items()]
            self._cells = {(c["x"], c["y"]): set(c["visible"]) for c in table["cells"]}
            self._pose = TfPoseSource(self)
            self.get_logger().info(f"visibility {visibility}: 물체 {len(self._world)}개, "
                                   f"목표 셀 {len(self._cells)}개")
        elif world == "toy":
            from athome.testing import toy_env

            relocated = _parse_moved(moved)
            self._world = toy_env.toy_world(relocated)
            self._unlinked = set(relocated)
            self._pose = TfPoseSource(self)
        elif world:
            from athome.scene_graph.static_features import load_static_features

            relocated = _parse_moved(moved)
            self._world = [(o.object_id, o.label, relocated.get(o.object_id, tuple(o.centroid)))
                           for o in load_static_features(world).objects]
            unknown = set(relocated) - {oid for oid, _, _ in self._world}
            if unknown:
                raise ValueError(f"moved에 없는 물체: {sorted(unknown)}")
            self._world += [(f"added_{i}", label, xyz)
                            for i, (label, xyz) in enumerate(_parse_moved(added).items())]
            self._unlinked = set(relocated) | {oid for oid, _, _ in self._world if oid.startswith("added_")}
            self._pose = TfPoseSource(self)
            self.get_logger().info(f"world {world}: 물체 {len(self._world)}개")

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
        if self._cells is None:
            world = self._world
            seen = visible_objects(world, pose, self._range, self._fov)
        else:
            cell = min(self._cells, key=lambda c: math.hypot(c[0] - pose.x, c[1] - pose.y))
            if math.hypot(cell[0] - pose.x, cell[1] - pose.y) > 0.01:
                return []
            world = [o for o in self._world if o[0] in self._cells[cell]]
            seen = visible_objects(world, pose, math.inf, self._fov)
        return [replace(o, object_id=self._map_id(world[o.object_id][0])) for o in seen]

    def _map_id(self, oid) -> int:
        """Pre-map ID of a world object, -1 if perception could not associate it."""
        if oid in self._unlinked:
            return -1
        tail = str(oid).rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else -1

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
