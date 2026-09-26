"""RViz markers for the search state (debugging on the robot).

/athome/viz/search   MarkerArray: rooms, search locations by state, goals,
                     found targets
/athome/viz/nav_grid OccupancyGrid: inflated grid used by A* (compare with
                     the MPPI costmap)
"""

import array
import math
from typing import Optional

import numpy as np

from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from athome.navigation import GridMap
from athome.scene_graph.query import WORKSPACE, SceneGraph
from athome.search import SearchDecision, SearchSession, TargetStatus

# state -> (r, g, b, a)
COLORS = {
    "open": (0.6, 0.6, 0.6, 0.35),
    "visited": (0.1, 0.8, 0.2, 0.5),
    "covered": (0.6, 0.9, 0.6, 0.35),
    "excluded": (0.9, 0.1, 0.1, 0.6),
    "current": (1.0, 0.85, 0.0, 0.8),
}


def _color(marker, rgba):
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba


def _marker(ns, mid, mtype, frame, stamp):
    m = Marker()
    m.header.frame_id = frame
    m.header.stamp = stamp
    m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
    m.pose.orientation.w = 1.0
    return m


def search_markers(
    graph: SceneGraph,
    session: Optional[SearchSession],
    decision: Optional[SearchDecision],
    frame: str,
    stamp,
) -> MarkerArray:
    out = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    out.markers.append(clear)

    room_points = {}
    for i, (lid, loc) in enumerate(sorted(graph.locations.items())):
        state = "open"
        if session is not None:
            if lid in session.visited:
                state = "visited"
            elif lid in session.covered:
                state = "covered"
            elif lid in session.excluded:
                state = "excluded"
        if decision is not None and decision.location_id == lid:
            state = "current"

        lo, hi = loc.bbox_min, loc.bbox_max
        box = _marker("locations", i, Marker.CUBE, frame, stamp)
        box.pose.position.x, box.pose.position.y, box.pose.position.z = (
            (a + b) / 2 for a, b in zip(lo, hi))
        box.scale.x, box.scale.y, box.scale.z = (max(b - a, 0.02) for a, b in zip(lo, hi))
        _color(box, COLORS[state])
        out.markers.append(box)

        text = _marker("labels", i, Marker.TEXT_VIEW_FACING, frame, stamp)
        text.pose.position.x, text.pose.position.y = box.pose.position.x, box.pose.position.y
        text.pose.position.z = hi[2] + 0.15
        text.scale.z = 0.15
        text.text = loc.category + (" [W]" if loc.kind == WORKSPACE else "")
        _color(text, (1.0, 1.0, 1.0, 0.9))
        out.markers.append(text)
        room_points.setdefault(loc.room_id, []).append(box.pose.position)

    for i, (rid, pts) in enumerate(sorted(room_points.items())):
        text = _marker("rooms", i, Marker.TEXT_VIEW_FACING, frame, stamp)
        text.pose.position.x = sum(p.x for p in pts) / len(pts)
        text.pose.position.y = sum(p.y for p in pts) / len(pts)
        text.pose.position.z = 2.2
        text.scale.z = 0.35
        text.text = f"{graph.rooms[rid].label} ({rid})"
        _color(text, (0.3, 0.8, 1.0, 1.0))
        out.markers.append(text)

    if decision is not None:
        for i, goal in enumerate(decision.goals):
            arrow = _marker("goals", i, Marker.ARROW, frame, stamp)
            arrow.points = [
                Point(x=goal.x, y=goal.y, z=0.05),
                Point(x=goal.x + 0.35 * math.cos(goal.yaw),
                      y=goal.y + 0.35 * math.sin(goal.yaw), z=0.05),
            ]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.05, 0.1, 0.1
            _color(arrow, (1.0, 0.5, 0.0, 1.0) if i == 0 else (1.0, 0.8, 0.5, 0.6))
            out.markers.append(arrow)

    if session is not None:
        for i, t in enumerate(session.targets):
            if t.status != TargetStatus.FOUND or t.found_object is None:
                continue
            x, y, z = t.found_object.centroid
            sphere = _marker("found", i, Marker.SPHERE, frame, stamp)
            sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = x, y, z
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.2
            _color(sphere, (1.0, 0.0, 1.0, 1.0))
            out.markers.append(sphere)
            text = _marker("found_labels", i, Marker.TEXT_VIEW_FACING, frame, stamp)
            text.pose.position.x, text.pose.position.y, text.pose.position.z = x, y, z + 0.3
            text.scale.z = 0.2
            text.text = f"FOUND {t.name}"
            _color(text, (1.0, 0.0, 1.0, 1.0))
            out.markers.append(text)
    return out


def grid_message(grid: GridMap, frame: str, stamp) -> OccupancyGrid:
    msg = OccupancyGrid()
    msg.header.frame_id = frame
    msg.header.stamp = stamp
    msg.info.resolution = float(grid.resolution)
    msg.info.height, msg.info.width = grid.shape
    msg.info.origin.position.x, msg.info.origin.position.y = grid.origin
    msg.info.origin.orientation.w = 1.0
    cells = np.where(grid.free, 0, 100).astype(np.int8)
    msg.data = array.array("b", cells.tobytes())
    return msg


LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
