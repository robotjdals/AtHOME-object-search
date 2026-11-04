"""Re-look, Known object first and second look on the ROS stack without the
robot: search_server with the room 406 setup, fake motion, and scripted
detections that depend on where the camera points (no planner server, CLIP or
API: planner min_cost, label check). Run on a private ROS domain:

    source /opt/ros/humble/setup.bash && source install/setup.bash
    ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 PYTHONPATH=src:scripts:$PYTHONPATH \
        python3 scripts/check_relook_scenarios.py --output outputs/relook_scenarios
"""
import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from athome.config import load_robot_config
from athome.navigation.grid import load_map_server
from athome_interfaces.action import SearchObjects
from athome_interfaces.msg import ObservedObject, ObservedObjectArray

ROOT = Path(__file__).resolve().parents[1]
HALF_VIEW = math.radians(35.3)
CENTRE = math.radians(15.0)
RANGE_M = 3.0


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Scenario:
    """``target``, what is seen from ``pose`` (x, y, yaw) as (label, x, y, confidence)."""
    target = "remote"

    def __init__(self):
        self.first_stop = None
        self.frames_seen = 0

    def point(self, degrees, r=0.9):
        x, y, yaw = self.first_stop
        a = yaw + math.radians(degrees)
        return x + r * math.cos(a), y + r * math.sin(a)

    def offset(self, xy, pose):
        """Angle from the view centre; out of view beyond the camera's range."""
        if math.hypot(xy[0] - pose[0], xy[1] - pose[1]) > RANGE_M:
            return math.pi
        return abs(wrap(math.atan2(xy[1] - pose[1], xy[0] - pose[0]) - pose[2]))

    def seen(self, pose):
        return []


class SeamNearMiss(Scenario):
    """As the towel of 2026-11-04: 0.08 on the seam of two views, 0.5 centred."""

    def seen(self, pose):
        if self.first_stop is None:
            return []
        xy = self.point(-30)
        off = self.offset(xy, pose)
        return [] if off > HALF_VIEW else [(self.target, *xy, 0.5 if off <= CENTRE else 0.08)]

    def check(self, visits, target):
        first = visits[0]
        relook = first["relooks"][0] if first["relooks"] else {}
        return {"첫 방문에 후보로 다시 봄": relook.get("kind") == "near_miss" and relook.get("passed") is True,
                "첫 방문에서 발견": first["found"] == [self.target],
                "방문 1회로 끝": len(visits) == 1, "최종 발견": target["status"] == "found"}


class SeamFalseFind(Scenario):
    """Confirmed on the seam, gone when centred: the find is taken back."""

    def seen(self, pose):
        if self.first_stop is None:
            return []
        xy = self.point(-30)
        off = self.offset(xy, pose)
        return [(self.target, *xy, 0.5)] if CENTRE < off <= HALF_VIEW else []

    def check(self, visits, target):
        first = visits[0]
        relook = first["relooks"][0] if first["relooks"] else {}
        return {"발견 확인에서 실패": relook.get("kind") == "found" and relook.get("passed") is False,
                "첫 방문에서 발견 아님": first["found"] == [],
                "바로 다른 자리 재확인": (len(visits) > 1 and visits[1]["stage"] == "recheck"
                                 and visits[1]["location_id"] == first["location_id"]),
                "위치마다 재확인 1회": _once(visits), "최종 못 찾음": target["status"] == "failed"}


class PlainMiss(Scenario):
    def check(self, visits, target):
        stages = [v["stage"] for v in visits]
        first = stages.index("recheck") if "recheck" in stages else len(stages)
        return {"다시 보기 없음": not any(v["relooks"] for v in visits),
                "다른 위치 먼저": first >= 2 and all(s == "recheck" for s in stages[first:]),
                "위치마다 재확인 1회": _once(visits), "최종 못 찾음": target["status"] == "failed"}


class Flicker(Scenario):
    """Seen once in front, above the floor: a likely candidate, no re-look."""

    def seen(self, pose):
        if self.first_stop is None or self.frames_seen >= 1:
            return []
        self.frames_seen += 1
        return [(self.target, *self.point(0), 0.5)]

    def check(self, visits, target):
        return {"다시 보기 없음": not visits[0]["relooks"],
                "바로 다른 자리 재확인": (len(visits) > 1 and visits[1]["stage"] == "recheck"
                                 and visits[1]["location_id"] == visits[0]["location_id"]),
                "최종 못 찾음": target["status"] == "failed"}


class KnownFirst(Scenario):
    """Known towel, detected only when centred."""
    target = "towel"
    towel = (-1.4, -4.41)

    def seen(self, pose):
        return [(self.target, *self.towel, 0.5)] if self.offset(self.towel, pose) <= CENTRE else []

    def check(self, visits, target):
        return {"아는 위치로 이동": visits[0]["stage"] == "known", "첫 방문에서 발견": visits[0]["found"] == [self.target],
                "방문 1회로 끝": len(visits) == 1}


def _once(visits):
    again = [v["location_id"] for v in visits if v["stage"] == "recheck"]
    return len(again) == len(set(again))


SCENARIOS = {"seam_near_miss": SeamNearMiss, "seam_false_find": SeamFalseFind, "plain_miss": PlainMiss,
             "flicker": Flicker, "known_first": KnownFirst}


def test_config(path: Path) -> Path:
    """Room 406 setup with absolute paths; no planner server, CLIP or parser."""
    base = ROOT / "configs/robot"
    raw = yaml.safe_load((base / "room406.yaml").read_text(encoding="utf-8"))

    def absolute(node):
        if isinstance(node, dict):
            return {k: absolute(v) for k, v in node.items()}
        if isinstance(node, str) and node.startswith("../"):
            return str((base / node).resolve())
        return node
    raw = absolute(raw)
    raw["planner"] = {"type": "min_cost"}
    raw["matcher"]["verify"]["enabled"] = False
    raw["navigation"]["path_planner"] = "none"
    raw.pop("command_parser", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


class Harness(Node):
    def __init__(self):
        super().__init__("athome_relook_check")
        self.scenario = None
        self.events = []
        self.state = None
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.pub = self.create_publisher(ObservedObjectArray, "/athome/perception/objects", 10)
        self.create_subscription(String, "/athome/search/events", lambda m: self.events.append(json.loads(m.data)), 100)
        self.create_subscription(String, "/athome/search/state", self._on_state, 10)
        self.create_timer(0.2, self._publish)
        self.client = ActionClient(self, SearchObjects, "/athome/search_objects")

    def _on_state(self, msg):
        self.state = msg.data
        if self.state == "observing" and self.scenario is not None and self.scenario.first_stop is None:
            self.scenario.first_stop = self.pose()

    def pose(self):
        try:
            t = self.tf.lookup_transform("map", "base_link", rclpy.time.Time())
        except TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def _publish(self):
        pose = self.pose()
        if pose is None:
            return
        msg = ObservedObjectArray()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.clip_model = "scripted"
        for label, x, y, confidence in (self.scenario.seen(pose) if self.scenario is not None else []):
            o = ObservedObject()
            o.object_id, o.is_static, o.label, o.confidence = -1, True, label, float(confidence)
            o.centroid.x, o.centroid.y, o.centroid.z = float(x), float(y), 0.8
            o.bbox_center.position = o.centroid
            o.bbox_center.orientation.w = 1.0
            o.bbox_size.x = o.bbox_size.y = o.bbox_size.z = 0.2
            msg.objects.append(o)
        self.pub.publish(msg)

    def spin_until(self, done, seconds):
        end = time.monotonic() + seconds
        while not done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return done()

    def run(self, scenario, timeout):
        self.scenario, self.events = scenario, []
        seen = lambda kind: any(e["event"] == kind for e in self.events)  # noqa: E731
        if not self.client.wait_for_server(timeout_sec=60):
            raise RuntimeError("search_server 없음")
        deadline = time.monotonic() + 60
        while not seen("command_start"):
            future = self.client.send_goal_async(SearchObjects.Goal(targets=[scenario.target], max_steps=30))
            self.spin_until(lambda: seen("command_start") or (future.done() and future.result().accepted), 5)
            self.spin_until(lambda: seen("command_start"), 2)
            if time.monotonic() > deadline:
                raise RuntimeError("명령 거절됨")
        if not self.spin_until(lambda: seen("command_end"), timeout):
            raise RuntimeError(f"{timeout}s 안에 끝나지 않음")
        self.spin_until(lambda: False, 1.0)
        starts = {e["step"]: e for e in self.events if e["event"] == "visit_start"}
        visits = [{"step": e["step"], "stage": starts[e["step"]]["stage"], "location_id": e["location_id"],
                   "status": e["status"], "found": e["found"], "relooks": e["relooks"],
                   "goal": e["goal"]} for e in self.events if e["event"] == "visit_end"]
        end = next(e for e in self.events if e["event"] == "command_end")
        self.scenario = None
        return visits, end


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/relook_scenarios")
    parser.add_argument("--only", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") in (None, "", "25"):
        raise SystemExit("개인 도메인에서 실행: ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1")
    config = test_config(args.output / "robot.yaml")
    loaded = load_robot_config(config)
    grid = load_map_server(loaded.map_yaml, 0.3)
    start = grid.to_xy(grid.nearest_free(grid.to_cell(2.0, -3.0), 3.0))
    log = (args.output / "server.log").open("w")
    procs = [subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) for cmd in (
        ["ros2", "run", "athome_ros", "search_server", "--ros-args", "-p", f"robot_config:={config}"],
        ["ros2", "run", "athome_ros", "fake_motion_server", "--ros-args", "-p", f"robot_config:={config}",
         "-p", f"initial_pose:=[{start[0]}, {start[1]}, 0.0]", "-p", "exact_arrival:=true",
         "-p", "linear_speed:=2.0", "-p", "angular_speed:=3.0"])]
    rclpy.init()
    node = Harness()
    results, failed = {}, 0
    try:
        for name in args.only:
            scenario = SCENARIOS[name]()
            visits, end = node.run(scenario, args.timeout)
            target = end["targets"][0]
            checks = scenario.check(visits, target)
            results[name] = {"visits": visits, "target": target, "checks": checks}
            failed += not all(checks.values())
            print(f"\n[{name}] {target['status']}")
            for v in visits:
                print(f"  step {v['step']} {v['stage']:9s} {v['location_id']} {v['status']} 발견 {v['found']} 다시 보기 "
                      + str([(r['kind'], r['passed']) for r in v['relooks']]))
            for what, ok in checks.items():
                print(f"  {'통과' if ok else '실패'}: {what}")
    finally:
        for p in procs:
            os.killpg(p.pid, signal.SIGINT)
        for p in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        (args.output / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        node.destroy_node()
        rclpy.try_shutdown()
    print(f"\n시나리오 {len(results)}개, 실패 {failed}개")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
