"""The scene graph in RViz as a layered 3D scene graph (Armeni et al. 2019,
Hydra, HOV-SG): the map with its rooms on the floor, the objects at their
positions, and above them the search locations, the rooms and the house,
linked as in the graph.

    ros2 run athome_ros scene_graph_viz --ros-args -p robot_config:=configs/robot/room412_v2.yaml
    rviz2 -d ros2/athome_bringup/rviz/scene_graph.rviz

/athome/viz/scene_graph_map  OccupancyGrid: the navigation map (map.yaml)
/athome/viz/scene_graph      MarkerArray (latched)

With the online object layer on (scene_graph.online_update), the search
server's current graph (/athome/scene_graph/online) is drawn instead of the
file whenever it changes; objects added online are marked "(new)".

Search locations are the planner's (Search Location policy of the robot
config); standalone objects outside it are gray. Labels are the graph's
English category and room labels (RViz text has no Hangul glyphs).
"""

import array
import json

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from athome.config import load_robot_config
from athome.navigation.grid import load_occupancy
from athome.scene_graph.location_policy import excluded_categories
from athome.scene_graph.query import SceneGraph

from athome_ros.visualization import LATCHED

# Heights of the abstract layers above the floor [m]; objects stay at their
# own height (about 0-1.5 m).
LOCATION_Z, ROOM_Z, HOUSE_Z = 2.0, 3.0, 4.0
# Same room colors as the slide figures; other room types take the palette in order.
ROOM_COLORS = {"bedroom": (0.56, 0.36, 0.78), "kitchen": (0.17, 0.60, 0.24), "bathroom": (0.12, 0.44, 0.71)}
PALETTE = [(0.90, 0.47, 0.10), (0.80, 0.20, 0.45), (0.10, 0.60, 0.60), (0.55, 0.45, 0.20)]
GRAY = (0.65, 0.68, 0.71)
FLOOR_CELL = 0.1          # [m] room floor tiles


def _rgba(rgb, a=1.0):
    return ColorRGBA(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=float(a))


def _point(x, y, z):
    return Point(x=float(x), y=float(y), z=float(z))


class Builder:
    def __init__(self, frame, stamp):
        self.frame, self.stamp = frame, stamp
        self.out = MarkerArray()
        self.ids = {}

    def marker(self, ns, mtype, rgb=GRAY, alpha=1.0):
        m = Marker()
        m.header.frame_id, m.header.stamp = self.frame, self.stamp
        m.ns, m.type, m.action = ns, mtype, Marker.ADD
        m.id = self.ids[ns] = self.ids.get(ns, -1) + 1
        m.pose.orientation.w = 1.0
        m.color = _rgba(rgb, alpha)
        self.out.markers.append(m)
        return m

    def node(self, ns, mtype, xyz, size, rgb, alpha=1.0):
        m = self.marker(ns, mtype, rgb, alpha)
        m.pose.position = _point(*xyz)
        m.scale.x = m.scale.y = m.scale.z = float(size)
        return m

    def text(self, ns, xyz, text, height, rgb=(0.1, 0.1, 0.1)):
        m = self.marker(ns, Marker.TEXT_VIEW_FACING, rgb)
        m.pose.position = _point(*xyz)
        m.scale.z = float(height)
        m.text = text
        return m


def scene_graph_markers(raw: dict, graph: SceneGraph, rooms: np.ndarray, rooms_origin, rooms_resolution,
                        floor_z: float, frame: str, stamp) -> MarkerArray:
    """``rooms``: the room segmentation saved with the graph (label i = room_<i>)."""
    b = Builder(frame, stamp)
    clear = Marker()
    clear.action = Marker.DELETEALL
    b.out.markers.append(clear)

    room_ids = [r["room_id"] for r in raw["rooms"]]
    others = iter(PALETTE * len(room_ids))
    color = {r["room_id"]: ROOM_COLORS.get(r["room_label"]) or next(others) for r in raw["rooms"]}
    objects = {o["object_id"]: o for o in raw["objects"]}

    # Floor tiles of each room and the room nodes above their centroids.
    step = max(1, int(round(FLOOR_CELL / rooms_resolution)))
    tiles = b.marker("room_floor", Marker.CUBE_LIST, alpha=0.45)
    tiles.scale.x = tiles.scale.y = step * rooms_resolution
    tiles.scale.z = 0.01
    room_xy = {}
    for rid in room_ids:
        rr, cc = np.nonzero(rooms == int(rid.rsplit("_", 1)[1]))
        if not len(rr):
            continue
        room_xy[rid] = (rooms_origin[0] + (cc.mean() + 0.5) * rooms_resolution,
                        rooms_origin[1] + (rr.mean() + 0.5) * rooms_resolution)
        keep = (rr % step == 0) & (cc % step == 0)
        for r, c in zip(rr[keep], cc[keep]):
            tiles.points.append(_point(rooms_origin[0] + (c + step / 2) * rooms_resolution,
                                       rooms_origin[1] + (r + step / 2) * rooms_resolution, floor_z))
            tiles.colors.append(_rgba(color[rid], 0.45))

    edges = b.marker("edges", Marker.LINE_LIST)
    edges.scale.x = 0.02

    def link(p, q, rgb, alpha=0.9):
        edges.points += [_point(*p), _point(*q)]
        edges.colors += [_rgba(rgb, alpha), _rgba(rgb, alpha)]

    house = (float(np.mean([v[0] for v in room_xy.values()])), float(np.mean([v[1] for v in room_xy.values()])),
             floor_z + HOUSE_Z)
    b.node("house", Marker.CUBE, house, 0.35, (0.2, 0.22, 0.25))
    b.text("labels", (house[0], house[1], house[2] + 0.4), "house", 0.3)
    for room in raw["rooms"]:
        rid = room["room_id"]
        if rid not in room_xy:
            continue
        p = (*room_xy[rid], floor_z + ROOM_Z)
        link(house, p, (0.35, 0.35, 0.35))
        b.node("rooms", Marker.SPHERE, p, 0.35, color[rid])
        b.text("labels", (p[0], p[1], p[2] + 0.4), f"{room['room_label']} ({rid})", 0.28, color[rid])

    locations = set(graph.locations)
    for w in raw["workspaces"]:
        src = objects[w["source_object_id"]]["bbox"]["center"]
        p = (src[0], src[1], floor_z + LOCATION_Z)
        rgb = color[w["room_id"]]
        if w["room_id"] in room_xy:
            link((*room_xy[w["room_id"]], floor_z + ROOM_Z), p, rgb)
        b.node("search_locations", Marker.CUBE, p, 0.22, rgb)
        b.text("labels", (p[0], p[1], p[2] + 0.28), objects[w["source_object_id"]]["semantic_tag"], 0.18, rgb)
        for oid in [w["source_object_id"], *w["child_object_ids"]]:
            link(p, objects[oid]["bbox"]["center"], rgb, 0.6)

    for o in raw["objects"]:
        c = o["bbox"]["center"]
        rgb = color.get(o["room_id"], GRAY)
        if o["role"] == "standalone":
            located = f"standalone:{o['object_id']}" in locations
            shade = rgb if located else GRAY
            p = (c[0], c[1], floor_z + LOCATION_Z)
            if o["room_id"] in room_xy:
                link((*room_xy[o["room_id"]], floor_z + ROOM_Z), p, shade, 0.7)
            m = b.node("search_locations", Marker.CYLINDER, p, 0.2, shade)
            m.scale.z = 0.08
            b.text("labels", (p[0], p[1], p[2] + 0.25), o["semantic_tag"], 0.15, shade)
            link(p, c, shade, 0.6)
        b.node("objects", Marker.SPHERE, c, 0.1, rgb)
        label = o["semantic_tag"] + (" (new)" if o.get("online") else "")
        b.text("object_labels", (c[0], c[1], c[2] + 0.13), label, 0.09, (0.2, 0.2, 0.2))
    return b.out


def map_message(yaml_path, frame, stamp) -> OccupancyGrid:
    m = load_occupancy(yaml_path)
    msg = OccupancyGrid()
    msg.header.frame_id, msg.header.stamp = frame, stamp
    msg.info.resolution = float(m.resolution)
    msg.info.height, msg.info.width = m.occupancy.shape
    msg.info.origin.position.x, msg.info.origin.position.y = (float(v) for v in m.origin)
    msg.info.origin.orientation.w = 1.0
    msg.data = array.array("b", np.clip(m.occupancy, -1, 100).astype(np.int8).tobytes())
    return msg


class SceneGraphViz(Node):
    def __init__(self):
        super().__init__("scene_graph_viz")
        config = load_robot_config(self.declare_parameter("robot_config", "").value)
        if config.graph_path is None:
            raise ValueError("robot_config에 scene_graph.path 필요")
        self._config = config
        self._excluded = excluded_categories(config.search_locations)
        rooms_path = config.graph_path.with_name(config.graph_path.stem + ".rooms.npz")
        with np.load(rooms_path) as saved:
            self._rooms, self._origin = saved["labels"], tuple(saved["origin_xy_m"])
            self._resolution = float(saved["resolution_m"])
        self._floor_z = config.floor_z_m if config.floor_z_m is not None else 0.0
        stamp = self.get_clock().now().to_msg()
        self.create_publisher(OccupancyGrid, "/athome/viz/scene_graph_map", LATCHED).publish(
            map_message(config.map_yaml, config.map_frame, stamp))
        self._markers = self.create_publisher(MarkerArray, "/athome/viz/scene_graph", LATCHED)
        self._draw(json.loads(config.graph_path.read_text(encoding="utf-8")), config.graph_path.name)
        self.create_subscription(
            String, self.declare_parameter("online_graph_topic", "/athome/scene_graph/online").value,
            lambda msg: self._draw(json.loads(msg.data), "online"), LATCHED)

    def _draw(self, raw: dict, source: str) -> None:
        graph = SceneGraph(raw, excluded_categories=self._excluded)
        markers = scene_graph_markers(raw, graph, self._rooms, self._origin, self._resolution, self._floor_z,
                                      self._config.map_frame, self.get_clock().now().to_msg())
        self._markers.publish(markers)
        new = sum(1 for o in raw["objects"] if o.get("online"))
        self.get_logger().info(
            f"{source}: 방 {len(raw['rooms'])}, 탐색 위치 {len(graph.locations)}, "
            f"물체 {len(raw['objects'])} (온라인 추가 {new}) -> /athome/viz/scene_graph ({len(markers.markers)} markers)")


def main():
    rclpy.init()
    node = SceneGraphViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
