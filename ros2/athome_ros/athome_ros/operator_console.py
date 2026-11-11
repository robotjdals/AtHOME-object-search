"""Operator console as a web page: the command path of command_cli (parse ->
the operator confirms or edits the targets -> search), a stop button, resume,
the camera view as the search judges it, the visits so far, what was found
with its photo, and manual checks of the results.

    ros2 run athome_ros operator_console --ros-args -p robot_config:=$PWD/configs/robot/room406.yaml
    # http://localhost:8080

Stop cancels the running command whoever started it (the action's cancel-all
request); the search server cancels the motion and waits for the robot to
stop. It is not an emergency stop: it needs the search server, the motion
module and the network.

Manual checks are what only a person can tell: a target reported found that
is not it, a target that was there but not found. Each goes to
<graph dir>/manual_checks.jsonl with the status at that moment and the current
camera view (<graph dir>/manual_checks/), and to /athome/search/manual_check
(in rosbag recordings of the run).

Standard library web server, local address by default: the page can move the
robot. For other computers: ``-p host:=0.0.0.0``; then every request needs the
access token (parameter ``token``, env ATHOME_CONSOLE_TOKEN, else generated at
start), given once in the logged URL and kept in a cookie. Plain HTTP: anyone
capturing the network traffic can read it, so use it on a network you trust.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import socket
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import rclpy
from action_msgs.srv import CancelGoal
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from athome.config import load_robot_config
from athome.execution.console_view import SearchTimeline, found_photo, perception_state, photo_path
from athome_interfaces.action import SearchObjects
from athome_interfaces.msg import ObservedObjectArray
from athome_interfaces.srv import ParseCommand
from athome_ros.operator_console_page import PAGE

ACTION = "/athome/search_objects"
RESULT_TEXT = {1: "완료", 2: "최대 탐색 횟수 도달", 3: "취소됨", 4: "일시정지", 5: "시간 예산 소진"}
VERDICTS = {"correct": "찾은 것 맞음", "false_positive": "오검출 (그 물체 아님)", "missed": "미검출 (있었는데 못 찾음)"}
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
SERVICE_WAIT_S = 3.0
JUDGED_VIEW_STALE_S = 1.5
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
COOKIE = "athome_console"


class OperatorConsole(Node):
    def __init__(self):
        super().__init__("athome_operator_console")
        p = self.declare_parameter
        config = load_robot_config(p("robot_config", "").value)
        self.host = p("host", "127.0.0.1").value
        self.port = p("port", 8080).value
        # Only for addresses other computers can reach.
        self.token = None
        self.share_url = None
        if self.host not in LOCAL_HOSTS:
            self.token = (p("token", "").value or os.environ.get("ATHOME_CONSOLE_TOKEN")
                          or secrets.token_urlsafe(9))
        self._group = ReentrantCallbackGroup()
        self._parse = self.create_client(ParseCommand, "/athome/parse_command", callback_group=self._group)
        self._search = ActionClient(self, SearchObjects, ACTION, callback_group=self._group)
        self._cancel = self.create_client(CancelGoal, ACTION + "/_action/cancel_goal", callback_group=self._group)
        self._check_pub = self.create_publisher(String, "/athome/search/manual_check", 10)
        self.create_subscription(String, "/athome/search/status", self._on_status, LATCHED,
                                 callback_group=self._group)
        self.create_subscription(CompressedImage, "/athome/search/debug_image/compressed", self._on_image, 1,
                                 callback_group=self._group)
        # The plain camera while the judged view does not come (perception or
        # the search server down), so the page keeps showing the camera.
        self.create_subscription(CompressedImage, p("camera_topic", "/camera/camera/color/image_raw/compressed").value,
                                 self._on_camera, qos_profile_sensor_data, callback_group=self._group)
        # A command waits for perception: the detector loads for minutes after a launch.
        self._perception_at = None
        self._perception_stale_s = float(config.observation.get("perception_timeout_s", 15.0))
        self.create_subscription(ObservedObjectArray, p("objects_topic", "/athome/perception/objects").value,
                                 self._on_objects, 10, callback_group=self._group)
        self._watch_pub = self.create_publisher(String, "/athome/search/watch", LATCHED)
        self._timeline = SearchTimeline()
        self.create_subscription(String, p("events_topic", "/athome/search/events").value, self._on_event, 50,
                                 callback_group=self._group)
        self.create_subscription(String, p("decision_topic", "/athome/planner/decision").value, self._on_decision, 50,
                                 callback_group=self._group)
        self.found_dir = Path(p("found_image_dir", str(config.graph_path.parent / "found")).value)
        self._image_at = None
        self._judged_at = None
        self._judged = 0
        self._checks_file = config.graph_path.parent / "manual_checks.jsonl"
        self._checks_dir = config.graph_path.parent / "manual_checks"
        self._lock = threading.Lock()
        self._image = None
        self._image_seq = 0
        self._image_cv = threading.Condition()
        self._status = None
        self._status_at = None
        self._running = False
        self._feedback = None
        self._result = None
        self._checks = []
        if self._checks_file.is_file():
            lines = self._checks_file.read_text(encoding="utf-8").splitlines()
            self._checks = [json.loads(l) for l in lines[-20:] if l.strip()]

    # --- ROS inputs --------------------------------------------------------

    def _on_objects(self, _msg):
        with self._lock:
            self._perception_at = time.monotonic()

    def _perception(self):
        at = self._perception_at
        return perception_state(None if at is None else time.monotonic() - at, self._perception_stale_s)

    def _on_status(self, msg):
        with self._lock:
            self._status = json.loads(msg.data)
            self._status_at = time.monotonic()
            self._timeline.on_status(self._status)

    def _on_event(self, msg):
        with self._lock:
            self._timeline.on_event(json.loads(msg.data))

    def _on_decision(self, msg):
        with self._lock:
            self._timeline.on_decision(json.loads(msg.data))

    def _on_image(self, msg):
        with self._image_cv:
            self._image = bytes(msg.data)
            self._image_seq += 1
            self._image_at = self._judged_at = time.monotonic()
            self._judged += 1
            first = self._judged == 1
            self._image_cv.notify_all()
        if first:           # the last step of the start: perception and search both work
            self.announce("조작 콘솔 준비됨")

    def _on_camera(self, msg):
        with self._image_cv:
            if self._judged_at is not None and time.monotonic() - self._judged_at < JUDGED_VIEW_STALE_S:
                return
            self._image = bytes(msg.data)
            self._image_seq += 1
            self._image_at = time.monotonic()
            self._image_cv.notify_all()

    def announce(self, title):
        """The addresses as a block that stands out in the launch's output."""
        line = "=" * 64
        lines = [line, title, f"  이 컴퓨터:   http://localhost:{self.port}"]
        if self.share_url:
            lines.append(f"  다른 컴퓨터: {self.share_url}")
        self.get_logger().info("\n" + "\n".join(lines + [line]))

    def latest_image(self):
        """The page asks for each picture: a stream it holds freezes on the
        last frame, without an error, when the console restarts or the network
        drops for a moment."""
        with self._image_cv:
            return self._image

    def next_image(self, seq, timeout=2.0):
        """(jpeg, seq) newer than ``seq``, or (None, seq) after ``timeout``."""
        with self._image_cv:
            self._image_cv.wait_for(lambda: self._image_seq != seq, timeout)
            return (self._image, self._image_seq) if self._image_seq != seq else (None, seq)

    # --- commands ----------------------------------------------------------

    def _wait(self, future, timeout):
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        return future.result() if done.wait(timeout) else None

    def parse(self, instruction):
        if not instruction.strip():
            return {"ok": False, "message": "명령을 입력하세요"}
        if not self._parse.wait_for_service(timeout_sec=SERVICE_WAIT_S):
            return {"ok": False, "message": "명령 해석 서비스 없음 (탐색 서버 실행 확인)"}
        r = self._wait(self._parse.call_async(ParseCommand.Request(instruction=instruction)), 30.0)
        if r is None:
            return {"ok": False, "message": "명령 해석 응답 없음"}
        if not r.success:
            return {"ok": False, "message": f"찾을 물체 없음: {r.message}"}
        return {"ok": True, "targets": [{"name": n, "known": bool(k)}
                                        for n, k in zip(r.targets, r.in_training_vocabulary)]}

    def start(self, targets=(), resume=False):
        targets = [t.strip() for t in targets if t.strip()]
        if not targets and not resume:
            return {"ok": False, "message": "찾을 물체가 없음"}
        with self._lock:
            perception = self._perception()
        if not perception["ready"]:
            return {"ok": False, "message": perception["text"]}
        if not self._search.wait_for_server(timeout_sec=SERVICE_WAIT_S):
            return {"ok": False, "message": "탐색 서버 없음"}
        with self._lock:
            self._feedback, self._result = None, None
        goal = SearchObjects.Goal(targets=targets, resume=resume)
        handle = self._wait(self._search.send_goal_async(goal, feedback_callback=self._on_feedback), 10.0)
        if handle is None or not handle.accepted:
            return {"ok": False, "message": "탐색 요청 거절됨 (실행 중인 명령, costmap 대기 등: 탐색 서버 로그 확인)"}
        with self._lock:
            self._running = True
        handle.get_result_async().add_done_callback(self._on_result)
        return {"ok": True}

    def _on_feedback(self, message):
        fb = message.feedback
        with self._lock:
            self._feedback = {"step": fb.step, "target": fb.target, "stage": fb.stage, "location": fb.location_id,
                              "path_cost": round(float(fb.path_cost), 2), "found": list(fb.found_targets)}

    def _on_result(self, future):
        r = future.result().result
        with self._lock:
            self._running = False
            self._result = {
                "status": int(r.status), "text": RESULT_TEXT.get(r.status, str(r.status)),
                "reason": r.reason, "detail": r.detail, "location": r.location_id, "steps": r.steps,
                "targets": [{"name": t.name, "status": t.status, "location": t.found_location,
                             "label": t.found_label, "detail": t.detail} for t in r.targets]}

    def stop(self):
        """Cancel every goal of the action (zero goal id and stamp): also a
        command started from another console."""
        if not self._cancel.wait_for_service(timeout_sec=SERVICE_WAIT_S):
            return {"ok": False, "message": "탐색 서버 없음: 조이스틱으로 정지하세요"}
        r = self._wait(self._cancel.call_async(CancelGoal.Request()), 5.0)
        if r is None:
            return {"ok": False, "message": "취소 응답 없음: 조이스틱으로 정지하세요"}
        if not r.goals_canceling:
            return {"ok": True, "message": "실행 중인 명령 없음"}
        return {"ok": True, "message": "취소 요청됨: 로봇 정지 확인 중"}

    def watch(self, target="", everything=False):
        self._watch_pub.publish(String(data=json.dumps({"target": target, "all": bool(everything)}, ensure_ascii=False)))
        return {"ok": True}

    def check(self, target, verdict, note=""):
        if verdict not in VERDICTS:
            return {"ok": False, "message": "모르는 판정"}
        with self._lock:
            status = dict(self._status or {})
        with self._image_cv:
            image = self._image
        now = time.time()
        stem = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}_{verdict}_{target.replace(' ', '_') or 'none'}"
        record = {"time": round(now, 3), "clock": time.strftime("%H:%M:%S", time.localtime(now)), "target": target,
                  "verdict": verdict, "verdict_text": VERDICTS[verdict], "note": note,
                  "command_id": status.get("command_id"), "state": status.get("state"), "status": status.get("text"),
                  "location": status.get("location"), "last": status.get("last"), "image": None}
        self._checks_dir.mkdir(parents=True, exist_ok=True)
        if image is not None:
            (self._checks_dir / f"{stem}.jpg").write_bytes(image)
            record["image"] = f"manual_checks/{stem}.jpg"
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._checks_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._checks = (self._checks + [record])[-20:]
        self._check_pub.publish(String(data=line))
        self.get_logger().info(f"수동 체크: {target} {VERDICTS[verdict]} {note}")
        return {"ok": True}

    def state(self):
        with self._lock:
            age = None if self._status_at is None else time.monotonic() - self._status_at
            timeline = self._timeline.view()
            command_id = (self._status or {}).get("command_id")
            for name, found in timeline["found"].items():
                found["photo"] = found_photo(self.found_dir, command_id, name, self._timeline.started_wall or 0.0)
            return {"status": self._status, "status_age_s": age, "running": self._running,
                    "feedback": self._feedback, "result": self._result, "checks": self._checks[::-1],
                    "timeline": timeline, "image_seq": self._image_seq,
                    "perception": self._perception(),
                    "image_age_s": None if self._image_at is None else time.monotonic() - self._image_at,
                    "judged": self._judged_at is not None
                    and time.monotonic() - self._judged_at < JUDGED_VIEW_STALE_S}


def make_handler(console: OperatorConsole):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, body: bytes, content_type: str, code=200):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value, code=200):
            self._send(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", code)

        def _local(self):
            return self.client_address[0] in ("127.0.0.1", "::1")

        def _cookie_ok(self):
            # This computer needs no token (the page shows it the other computers' address).
            if self._local():
                return True
            cookie = SimpleCookie(self.headers.get("Cookie", "")).get(COOKIE)
            return cookie is not None and hmac.compare_digest(cookie.value, console.token)

        def do_GET(self):
            url = urlsplit(self.path)
            if console.token is not None and not self._cookie_ok():
                given = parse_qs(url.query).get("token", [""])[0]
                if not hmac.compare_digest(given, console.token):
                    self._send("접속 토큰이 필요합니다: 콘솔 로그의 주소로 여세요".encode("utf-8"),
                               "text/plain; charset=utf-8", 401)
                    return
                # Keep it in a cookie and drop it from the address bar.
                self.send_response(303)
                self.send_header("Set-Cookie", f"{COOKIE}={console.token}; Path=/; HttpOnly; SameSite=Strict")
                self.send_header("Location", "/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if url.path == "/":
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif url.path == "/state":
                state = console.state()
                if self._local():
                    state["share_url"] = console.share_url
                self._json(state)
            elif url.path == "/frame.jpg":
                image = console.latest_image()
                if image is None:
                    self._json({"ok": False, "message": "no image"}, 404)
                else:
                    self._send(image, "image/jpeg")
            elif url.path == "/stream.mjpg":
                self._stream()
            elif url.path.startswith("/found/"):
                path = photo_path(console.found_dir, unquote(url.path[len("/found/"):]))
                if path is None:
                    self._json({"ok": False, "message": "not found"}, 404)
                else:
                    self._send(path.read_bytes(), "image/jpeg")
            else:
                self._json({"ok": False, "message": "not found"}, 404)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            seq = -1
            try:
                while rclpy.ok():
                    image, seq = console.next_image(seq)
                    if image is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(image))
                    self.wfile.write(image + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if console.token is not None and not self._cookie_ok():
                self._json({"ok": False, "message": "접속 토큰이 필요합니다"}, 401)
                return
            # JSON bodies only: a page on another site cannot send this type
            # to the local address without a CORS preflight, which is refused.
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                self._json({"ok": False, "message": "application/json only"}, 415)
                return
            size = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(size) or b"{}")
            except ValueError:
                self._json({"ok": False, "message": "잘못된 JSON"}, 400)
                return
            routes = {
                "/parse": lambda: console.parse(body.get("instruction", "")),
                "/start": lambda: console.start(body.get("targets", [])),
                "/resume": lambda: console.start(resume=True),
                "/stop": console.stop,
                "/watch": lambda: console.watch(body.get("target", ""), body.get("all", False)),
                "/check": lambda: console.check(body.get("target", ""), body.get("verdict", ""), body.get("note", "")),
            }
            if self.path not in routes:
                self._json({"ok": False, "message": "not found"}, 404)
                return
            self._json(routes[self.path]())

    return Handler


def main(args=None):
    rclpy.init(args=args)
    node = OperatorConsole()
    server = ThreadingHTTPServer((node.host, node.port), make_handler(node))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if node.token is not None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))      # no packet: the outgoing interface's address
            address = s.getsockname()[0]
        node.share_url = f"http://{address}:{node.port}/?token={node.token}"
    node.announce("조작 콘솔 시작 (카메라 영상이 들어오면 다시 알려 줌)")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
