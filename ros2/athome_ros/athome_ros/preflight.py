"""Check everything a search run needs, before the first command.

    ros2 run athome_ros preflight --ros-args -p robot_config:=$PWD/configs/robot/demo.yaml \
        [-p camera_info_topic:=/camera/color/camera_info]

Run it with the whole system up (search.launch.py, localization, MPPI,
perception), from the shell search_server runs in (API keys). One line per
check; exit code 1 if any check failed. Config and server checks are in
athome.preflight; here the running system is checked with search_server's
topic and action names and defaults.
"""

import math
import sys
import time

import numpy as np
import rclpy
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformException, TransformListener

from athome.config import load_robot_config
from athome.navigation import load_map_server, load_occupancy, traversable_from_costmap
from athome.preflight import (
    Check,
    Status,
    config_checks,
    free_space_agreement,
    guarded,
    report,
    server_checks,
)
from athome_interfaces.action import NavigateToGoal, SearchObjects
from athome_interfaces.msg import ObservedObjectArray
from athome_interfaces.srv import ParseCommand

# /map and the Nav2 costmap are latched (as search_server subscribes).
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
SERVER_WAIT_S = 2.0
MAX_OBJECTS_AGE_S = 2.0
MAX_HFOV_ERROR_DEG = 1.0
MIN_FREE_AGREEMENT = 0.95


def _grid(msg: OccupancyGrid) -> np.ndarray:
    return np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)


def _age_s(node: Node, stamp) -> float:
    return (node.get_clock().now() - Time.from_msg(stamp)).nanoseconds / 1e9


class Preflight(Node):
    def __init__(self):
        super().__init__("athome_preflight")
        p = self.declare_parameter
        path = p("robot_config", "").value
        if not path:
            raise ValueError("robot_config 파라미터 필요: --ros-args -p robot_config:=<yaml>")
        self.config = load_robot_config(path)
        self.wait_s = p("wait_s", 5.0).value
        self.max_tf_age_s = p("max_tf_age_s", 1.0).value
        self.map_topic = p("map_topic", "/map").value
        self.costmap_topic = p("costmap_topic", "/global_costmap/costmap").value
        self.planner_action = p("planner_action", "/compute_path_to_pose").value
        self.navigate_action = p("navigate_action", "/athome/navigate_to_goal").value
        self.objects_topic = p("objects_topic", "/athome/perception/objects").value
        self.search_action = p("search_action", "/athome/search_objects").value
        self.parse_service = p("parse_service", "/athome/parse_command").value
        self.camera_info_topic = p("camera_info_topic", "").value

        self.buffer = Buffer()
        self._tf_listener = TransformListener(self.buffer, self)
        self.map_msg = self.costmap_msg = self.camera_msg = None
        self.objects = []                 # (receive time [s], message)
        self.create_subscription(OccupancyGrid, self.map_topic,
                                 lambda m: setattr(self, "map_msg", m), LATCHED)
        self.create_subscription(OccupancyGrid, self.costmap_topic,
                                 lambda m: setattr(self, "costmap_msg", m), LATCHED)
        self.create_subscription(ObservedObjectArray, self.objects_topic,
                                 lambda m: self.objects.append((time.monotonic(), m)), 10)
        if self.camera_info_topic:
            self.create_subscription(CameraInfo, self.camera_info_topic,
                                     lambda m: setattr(self, "camera_msg", m),
                                     qos_profile_sensor_data)

    def collect(self) -> None:
        """Receive messages and TF for ``wait_s`` seconds."""
        deadline = time.monotonic() + self.wait_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

    def live_checks(self):
        nav2 = self.config.path_planner == "nav2"
        skip = "path_planner none: Nav2 안 씀"
        return [
            guarded("로봇 위치 TF", self.check_tf),
            guarded(f"지도 토픽 {self.map_topic}", self.check_map) if nav2
            else Check(f"지도 토픽 {self.map_topic}", Status.SKIP, skip),
            guarded("Nav2 경로 계획", self.check_planner) if nav2
            else Check("Nav2 경로 계획", Status.SKIP, skip),
            guarded(f"Nav2 costmap {self.costmap_topic}", self.check_costmap) if nav2
            else Check(f"Nav2 costmap {self.costmap_topic}", Status.SKIP, skip),
            guarded(f"주행 모듈 {self.navigate_action}", self.check_motion),
            guarded(f"인지 {self.objects_topic}", self.check_objects),
            guarded("카메라 시야", self.check_camera),
            guarded("탐색 서버", self.check_search_server),
        ]

    def check_tf(self) -> Check:
        name, cfg = "로봇 위치 TF", self.config
        try:
            t = self.buffer.lookup_transform(cfg.map_frame, cfg.base_frame, Time())
        except TransformException as e:
            return Check(name, Status.FAIL, f"{cfg.map_frame}→{cfg.base_frame} 없음 ({e}): "
                                            "위치 추정, body→base_link 정적 TF 확인")
        p, q = t.transform.translation, t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        where = f"x {p.x:.2f}, y {p.y:.2f}, yaw {math.degrees(yaw):.0f}°"
        if Time.from_msg(t.header.stamp).nanoseconds == 0:
            return Check(name, Status.WARN, f"{where}: 정적 변환만 있음 (위치 추정 발행 확인)")
        age = _age_s(self, t.header.stamp)
        if abs(age) > self.max_tf_age_s:
            return Check(name, Status.FAIL, f"{where}: {age:.1f}초 전 값 "
                                            "(위치 추정 멈춤 또는 장비 간 시계 차이)")
        # The robot center must be on a free cell of the map the graph was built on.
        m = load_occupancy(cfg.map_yaml)
        row = int(math.floor((p.y - m.origin[1]) / m.resolution))
        col = int(math.floor((p.x - m.origin[0]) / m.resolution))
        rows, cols = m.occupancy.shape
        if not (0 <= row < rows and 0 <= col < cols) or m.occupancy[row, col] != 0:
            return Check(name, Status.WARN, f"{where}: 지도의 빈 칸이 아님 "
                                            "(위치 추정 또는 지도 원점 확인)")
        return Check(name, Status.PASS, f"{where}, {age:.2f}초 전")

    def check_map(self) -> Check:
        name, msg = f"지도 토픽 {self.map_topic}", self.map_msg
        if msg is None:
            return Check(name, Status.FAIL, f"메시지 없음 ({self.wait_s:.0f}초): "
                                            "map_server 또는 위치 추정의 지도 발행 확인")
        m, info = load_occupancy(self.config.map_yaml), msg.info
        o = info.origin
        if max(abs(o.orientation.x), abs(o.orientation.y), abs(o.orientation.z)) > 1e-6:
            return Check(name, Status.FAIL, "회전된 지도 원점")
        shape, origin = (info.height, info.width), (o.position.x, o.position.y)
        if (shape != m.occupancy.shape or not np.allclose(origin, m.origin, atol=1e-6)
                or not math.isclose(info.resolution, m.resolution, rel_tol=1e-6)):
            return Check(name, Status.FAIL,
                         f"설정 지도와 다름: {info.width}×{info.height} 칸, 원점 {origin}, "
                         f"{info.resolution:.3f} m (설정 {m.occupancy.shape[1]}×"
                         f"{m.occupancy.shape[0]}, {m.origin}, {m.resolution:.3f} m)")
        differ = float(np.mean(_grid(msg) != m.occupancy))
        if differ > 0:
            return Check(name, Status.FAIL, f"설정 지도와 칸 {differ:.1%} 다름")
        return Check(name, Status.PASS,
                     f"설정 지도와 같음 ({info.width}×{info.height} 칸, {info.resolution:.3f} m)")

    def check_planner(self) -> Check:
        ready = ActionClient(self, ComputePathToPose, self.planner_action).wait_for_server(
            timeout_sec=SERVER_WAIT_S)
        if not ready:
            return Check("Nav2 경로 계획", Status.FAIL,
                         f"{self.planner_action} Action 서버 없음 (search.launch.py)")
        return Check("Nav2 경로 계획", Status.PASS, f"{self.planner_action} 있음")

    def check_costmap(self) -> Check:
        name, msg, cfg = f"Nav2 costmap {self.costmap_topic}", self.costmap_msg, self.config
        if msg is None:
            return Check(name, Status.FAIL, f"메시지 없음 ({self.wait_s:.0f}초): planner_server 확인")
        if msg.header.frame_id != cfg.map_frame:
            return Check(name, Status.FAIL, f"frame {msg.header.frame_id} ≠ {cfg.map_frame}")
        ours = load_map_server(cfg.map_yaml, cfg.inflation_radius, cfg.unknown_as_occupied)
        origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        agreement = free_space_agreement(
            ours, traversable_from_costmap(_grid(msg)), origin, msg.info.resolution)
        detail = (f"주행 가능 영역이 설정(지도 + 로봇 반경 {cfg.inflation_radius} m)과 "
                  f"{agreement:.1%} 일치")
        if agreement < MIN_FREE_AGREEMENT:
            return Check(name, Status.WARN, detail + ": costmap의 로봇 크기·지도 확인")
        return Check(name, Status.PASS, detail)

    def check_motion(self) -> Check:
        name = f"주행 모듈 {self.navigate_action}"
        if not ActionClient(self, NavigateToGoal, self.navigate_action).wait_for_server(
                timeout_sec=SERVER_WAIT_S):
            return Check(name, Status.FAIL, "Action 서버 없음 (주행 모듈 실행 확인)")
        return Check(name, Status.PASS, "Action 서버 있음")

    def check_objects(self) -> Check:
        name, cfg = f"인지 {self.objects_topic}", self.config
        if not self.objects:
            return Check(name, Status.FAIL, f"메시지 없음 ({self.wait_s:.0f}초): 인지 모듈 실행 확인 "
                                            "(빈 목록도 계속 발행해야 함)")
        first, (last_time, last) = self.objects[0][0], self.objects[-1]
        if last.header.frame_id != cfg.map_frame:
            return Check(name, Status.FAIL, f"frame {last.header.frame_id} ≠ {cfg.map_frame}")
        if cfg.matcher.get("type") == "clip" and last.clip_model != cfg.matcher["clip"]["model"]:
            return Check(name, Status.FAIL, f"CLIP 모델 {last.clip_model!r} ≠ 설정 "
                                            f"{cfg.matcher['clip']['model']!r}")
        rate = ((len(self.objects) - 1) / (last_time - first)
                if len(self.objects) > 1 and last_time > first else 0.0)
        static = sum(o.is_static for o in last.objects)
        detail = f"{rate:.1f} Hz, 물체 {len(last.objects)}개 (정적 {static})"
        age = _age_s(self, last.header.stamp)
        if abs(age) > MAX_OBJECTS_AGE_S:
            return Check(name, Status.WARN, f"{detail}, 인식 시각이 {age:.1f}초 전: "
                                            "지연 또는 장비 간 시계 차이(NTP) 확인")
        return Check(name, Status.PASS, f"{detail}, 인식 시각 {age:.2f}초 전")

    def check_camera(self) -> Check:
        name = "카메라 시야"
        if not self.camera_info_topic:
            return Check(name, Status.SKIP, "camera_info_topic 파라미터 없음")
        msg = self.camera_msg
        if msg is None:
            return Check(name, Status.FAIL, f"{self.camera_info_topic} 메시지 없음")
        hfov = math.degrees(2 * math.atan(msg.width / (2 * msg.k[0])))
        actual = f"{hfov:.1f}° ({msg.width}×{msg.height})"
        configured = self.config.camera_hfov_deg
        if configured is None:
            return Check(name, Status.WARN, f"실제 {actual}, sensors.camera_hfov_deg 없음")
        if abs(hfov - configured) > MAX_HFOV_ERROR_DEG:
            return Check(name, Status.FAIL, f"실제 {actual} ≠ 설정 {configured}° "
                                            "(해상도 또는 sensors.camera_hfov_deg 확인)")
        return Check(name, Status.PASS, f"{actual}, 설정과 같음")

    def check_search_server(self) -> Check:
        missing = []
        if not ActionClient(self, SearchObjects, self.search_action).wait_for_server(
                timeout_sec=SERVER_WAIT_S):
            missing.append(self.search_action)
        if not self.create_client(ParseCommand, self.parse_service).wait_for_service(
                timeout_sec=SERVER_WAIT_S):
            missing.append(self.parse_service)
        if missing:
            return Check("탐색 서버", Status.FAIL,
                         f"{', '.join(missing)} 없음: search_server 실행 확인 (search.launch.py)")
        return Check("탐색 서버", Status.PASS, "search_server 실행 중")


def main():
    rclpy.init()
    try:
        node = Preflight()
    except (OSError, ValueError, KeyError) as e:
        rclpy.shutdown()
        sys.exit(f"robot config 문제: {e}")
    try:
        sections = [("설정·파일", config_checks(node.config)),
                    ("서버", server_checks(node.config))]
        node.collect()
        sections.append(("실행 중인 시스템", node.live_checks()))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    lines, code = report(sections)
    print("\n".join(lines))
    sys.exit(code)


if __name__ == "__main__":
    main()
