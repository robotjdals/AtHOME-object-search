"""Planner comparison runs on the robot (athome.execution.experiment has the plan
and the records). Used by the operator console's Experiment tab and by
scripts/run_final_experiment.py; the node's executor must spin in another
thread (the operator console's does), every step here blocks a worker thread.

Per scenario the operator places the object (``placed``); for each planner
variant the runner sets the search server's planner_variant parameter, drives
the robot to the saved start pose through the motion module (Nav2 path +
NavigateToGoal, as the search does; the search server's heartbeat keeps the
motion module's watchdog fed), waits for fresh perception, records the actual
start pose, sends the target as a confirmed command and waits for its end.
The operator then marks the run valid or invalid (``verdict``); an invalid run
is run again. ``abort`` cancels the running move or command and ends the session.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from std_msgs.msg import String

from athome.execution import HandleState
from athome.execution.experiment import (
    abort_run,
    append_run,
    check_starts,
    finish_run,
    load_plan,
    load_runs,
    pair_order,
    progress,
    remaining,
    start_names,
)
from athome.execution.screen_record import ScreenRecorder
from athome.schemas import MotionStatus, NavigationRequest, Pose2D
from athome_interfaces.action import SearchObjects
from athome_interfaces.msg import ObservedObjectArray
from athome_ros.adapters import RosMotionClient
from athome_ros.common import TfPoseSource

SEARCH_SERVER = "athome_search_server"
MOVE_TIMEOUT_S = 180.0
PERCEPTION_WAIT_S = 240.0       # the detector loads for 2-3 min after a launch
COMMAND_MARGIN_S = 150.0        # over the time budget: a visit under way ends first

PHASE_TEXT = {"idle": "대기", "place": "물체 배치", "running": "진행 중", "manual": "운영자 조치 필요",
              "verdict": "판정", "done": "완료", "error": "오류"}


class Aborted(Exception):
    pass


class ExperimentRunner:
    def __init__(self, node, config, plan_path, out_dir, callback_group=None, found_dir=None, record=""):
        self.node = node
        self.config = config
        self.plan_path = Path(plan_path)
        self.plan = load_plan(plan_path)
        self.out = Path(out_dir)
        self.found_dir = Path(found_dir) if found_dir else config.graph_path.parent / "found"
        self.runs_path = self.out / "runs.jsonl"
        self.starts_path = self.out / "starts.json"
        g = callback_group
        self.pose = TfPoseSource(node, config.map_frame, config.base_frame)
        self.motion = RosMotionClient(
            node, "/athome/navigate_to_goal", self.pose,
            planner_action="/compute_path_to_pose" if config.path_planner == "nav2" else None)
        self._search = ActionClient(node, SearchObjects, "/athome/search_objects", callback_group=g)
        self._params = node.create_client(SetParameters, f"/{SEARCH_SERVER}/set_parameters", callback_group=g)
        self._capture = threading.Lock()
        self._events, self._decisions = [], []
        self._perception_at = None
        self._stale_s = float(config.observation.get("perception_timeout_s", 15.0))
        node.create_subscription(String, "/athome/search/events", self._on_event, 100, callback_group=g)
        node.create_subscription(String, "/athome/planner/decision", self._on_decision, 100, callback_group=g)
        node.create_subscription(ObservedObjectArray, "/athome/perception/objects", self._on_objects, 10,
                                 callback_group=g)
        self.timeout_s = float(config.time_budget_s or 600.0) + COMMAND_MARGIN_S
        # Recording per run ("" off, "window": the console's browser window, a monitor name, or "all").
        self.recorder = ScreenRecorder(record) if record else None
        # Session state, read by view() from other threads.
        self._lock = threading.Condition()
        self._phase, self._text = "idle", ""
        self._row, self._variant, self._record = None, None, None
        self._action = None
        self._abort = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._move = None
        self._command = None
        self._check = None

    # --- ROS inputs ------------------------------------------------------------------------
    def _on_event(self, msg):
        with self._capture:
            self._events.append(json.loads(msg.data))

    def _on_decision(self, msg):
        with self._capture:
            self._decisions.append(json.loads(msg.data))

    def _on_objects(self, _msg):
        self._perception_at = time.monotonic()

    # --- blocking helpers (worker thread) --------------------------------------------------
    def _sleep_until(self, done, seconds: float) -> bool:
        end = time.monotonic() + seconds
        while not done():
            if self._abort.is_set():
                raise Aborted()
            if time.monotonic() > end:
                return False
            time.sleep(0.1)
        return True

    def _future(self, future, seconds: float):
        return future.result() if self._sleep_until(future.done, seconds) else None

    def _fresh_perception(self) -> bool:
        return self._perception_at is not None and time.monotonic() - self._perception_at <= self._stale_s

    def _say(self, text: str) -> None:
        with self._lock:
            self._text = text

    def _set(self, phase: str, text: str = "", **fields) -> None:
        with self._lock:
            self._phase, self._text = phase, text
            for k, v in fields.items():
                setattr(self, "_" + k, v)
            self._lock.notify_all()

    def _wait_action(self, allowed):
        """Block until the operator sends one of ``allowed`` (or aborts)."""
        with self._lock:
            self._action = None
            while True:
                if self._abort.is_set():
                    raise Aborted()
                if self._action is not None and self._action[0] in allowed:
                    action, self._action = self._action, None
                    return action
                self._lock.wait(0.2)

    # --- steps -----------------------------------------------------------------------------
    def set_variant(self, variant: str) -> None:
        if not self._params.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("탐색 서버의 파라미터 서비스 없음 (탐색 서버 실행 확인)")
        value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=variant)
        response = self._future(self._params.call_async(
            SetParameters.Request(parameters=[Parameter(name="planner_variant", value=value)])), 10.0)
        if response is None:
            raise RuntimeError("planner_variant 설정 응답 없음")
        if not response.results[0].successful:
            raise RuntimeError(f"planner_variant={variant} 거절: {response.results[0].reason}")

    def go_to(self, goal: Pose2D, request_id: str) -> dict:
        if not self._sleep_until(self.motion.servers_ready, 30.0):
            return {"status": "failed", "reason": "no_server", "detail": "주행 모듈 또는 Nav2 planner 없음"}
        started = time.monotonic()
        handle = self.motion.navigate(NavigationRequest(
            request_id=request_id, goal=goal, map_version=self.config.map_version, timeout_sec=MOVE_TIMEOUT_S))
        self._move = handle
        try:
            if not self._sleep_until(lambda: handle.state in (HandleState.DONE, HandleState.REJECTED),
                                     MOVE_TIMEOUT_S + 10):
                handle.request_cancel()
                self._sleep_until(lambda: handle.state == HandleState.DONE, 15.0)
                return {"status": "failed", "reason": "timeout", "detail": "시간 안에 도착하지 않음",
                        "seconds": round(time.monotonic() - started, 1)}
        except Aborted:
            handle.request_cancel()
            end = time.monotonic() + 15.0
            while handle.state != HandleState.DONE and time.monotonic() < end:
                time.sleep(0.1)
            raise
        finally:
            self._move = None
        if handle.state == HandleState.REJECTED:
            return {"status": "failed", "reason": "rejected", "detail": "주행 모듈이 요청 거절",
                    "seconds": round(time.monotonic() - started, 1)}
        r = handle.result
        return {"status": r.status.value, "reason": r.reason.value, "detail": r.detail,
                "seconds": round(time.monotonic() - started, 1)}

    def search(self, target: str):
        """One confirmed command; (events, decisions, action status). The end is
        read from the events: a goal response can be lost just after a server
        starts (scripts/run_hm3d_episodes.py)."""
        with self._capture:
            self._events, self._decisions = [], []

        def seen(kind):
            with self._capture:
                return any(e.get("event") == kind for e in self._events)

        if not self._search.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("탐색 서버 없음")
        goal = SearchObjects.Goal(targets=[target])
        handle, deadline = None, time.monotonic() + 30
        while not seen("command_start"):
            future = self._search.send_goal_async(goal)
            self._sleep_until(lambda: future.done() or seen("command_start"), 10)
            if future.done() and future.result() is not None and future.result().accepted:
                handle = future.result()
                break
            if time.monotonic() > deadline:
                raise RuntimeError("탐색 명령 거절됨 (실행 중인 명령 또는 costmap 대기: 탐색 서버 로그 확인)")
            self._sleep_until(lambda: seen("command_start"), 1.0)
        self._command = handle
        result = handle.get_result_async() if handle is not None else None
        try:
            if not self._sleep_until(lambda: seen("command_end"), self.timeout_s):
                if handle is not None:
                    handle.cancel_goal_async()
                self._wait_plain(lambda: seen("command_end"), 30.0)
                raise RuntimeError(f"명령이 {self.timeout_s:.0f}초 안에 끝나지 않아 취소함")
        except Aborted:
            if handle is not None:
                handle.cancel_goal_async()
                self._wait_plain(lambda: seen("command_end"), 30.0)
            raise
        finally:
            self._command = None
        self._wait_plain(lambda: result is None or result.done(), 2.0)
        status = result.result().status if result is not None and result.done() else None
        with self._capture:
            return list(self._events), list(self._decisions), status

    @staticmethod
    def _wait_plain(done, seconds):
        end = time.monotonic() + seconds
        while not done() and time.monotonic() < end:
            time.sleep(0.1)

    def _run_one(self, record: dict, row: dict, variant: str, index: int, starts: dict) -> dict:
        """Fills ``record`` in place, so an abort keeps what the run got to."""
        record.update({"run": index, "scenario": row["scenario"], "variant": variant, "object": row["object"],
                          "location_id": row["location_id"], "room": row.get("room"), "start": row["start"],
                          "start_pose_saved": starts[row["start"]], "typical": row.get("typical"),
                          "clock": time.strftime("%Y-%m-%d %H:%M:%S")})
        self._say(f"planner {variant} 설정")
        self.set_variant(variant)
        x, y, yaw = starts[row["start"]]
        self._say(f"시작 위치({row['start']})로 이동 중")
        move = self.go_to(Pose2D(x, y, yaw), f"exp{index}_start")
        record["move_to_start"] = move
        if move["status"] != MotionStatus.SUCCEEDED.value:
            self._set("manual", f"시작 위치 이동 실패({move['reason']}: {move['detail']}). "
                                "로봇을 손으로 시작 위치에 놓고 계속을 누르세요.")
            self._wait_action({"continue"})
            self._set("running", "")
        self._say("인지 결과 기다리는 중")
        if not self._sleep_until(self._fresh_perception, PERCEPTION_WAIT_S):
            self._set("manual", "인지 결과가 없습니다. 카메라, 인지 노드, 검출 서버를 확인하고 계속을 누르세요.")
            self._wait_action({"continue"})
            self._set("running", "")
        pose = self.pose.current_pose()
        record["start_pose_actual"] = None if pose is None else [round(pose.x, 3), round(pose.y, 3), round(pose.yaw, 4)]
        record["t_start"] = time.time()
        self._say(f"{row['object']} 탐색 중 (planner {variant})")
        # The video covers the command only (start pose to its end), as t_start..t_end.
        name = f"videos/run_{index:02d}_s{row['scenario']:02d}_{variant}.mp4"
        recording = self.recorder is not None and self.recorder.start(self.out / name)
        try:
            events, decisions, status = self.search(row["object"])
        finally:      # also on an abort: the aborted run keeps its video
            video = self.recorder.stop() if recording else None
            record["video"] = name if video is not None else None
            if self.recorder is not None and video is None:
                record["video_error"] = self.recorder.error or "녹화 파일 없음"
        return finish_run(record, events, decisions, status, self.out, self.found_dir)

    def _loop(self) -> None:
        pending = None      # the run under way until its verdict is written
        try:
            starts = json.loads(self.starts_path.read_text(encoding="utf-8"))
            todo = remaining(self.plan, load_runs(self.runs_path))
            placed, i = None, 0
            while i < len(todo):
                row, variant = todo[i]
                if placed != row["scenario"]:
                    self._set("place", f"{row['object']}을(를) {row.get('room', '')} {row.get('desk_hint', '')}에 "
                                       "카메라에서 보이는 쪽으로 놓고 '놓았음'을 누르세요.", row=row, variant=None,
                              record=None)
                    action = self._wait_action({"placed", "skip"})
                    if action[0] == "skip":
                        while i < len(todo) and todo[i][0]["scenario"] == row["scenario"]:
                            i += 1
                        placed = None
                        continue
                    placed = row["scenario"]
                index = len(load_runs(self.runs_path)) + 1
                self._set("running", "", row=row, variant=variant, record=None)
                pending = {}
                record = self._run_one(pending, row, variant, index, starts)
                self._set("verdict", "결과를 보고 유효 또는 무효(이유)를 누르세요. 무효면 같은 회를 다시 합니다.",
                          record=record)
                _, data = self._wait_action({"verdict"})
                record["valid"] = bool(data.get("valid"))
                record["note"] = str(data.get("note", "")).strip()
                append_run(self.runs_path, record)
                pending = None
                if record["valid"]:
                    i += 1
            self._set("done", "모든 실행이 끝났습니다.", row=None, variant=None)
        except Aborted:
            if pending:
                with self._capture:     # search() cleared them when this run's search began
                    events, decisions = list(self._events), list(self._decisions)
                append_run(self.runs_path, abort_run(pending, events, decisions, self.out, self.found_dir))
            self._set("idle", "실험을 중단했습니다. 진행 중이던 회는 '판정 전 중단'으로 기록했고, "
                              "다시 시작하면 유효한 회는 건너뜁니다.", row=None, variant=None)
        except Exception as e:  # noqa: BLE001 - shown to the operator
            self._set("error", f"{type(e).__name__}: {e}")
        finally:
            self._abort.clear()

    # --- operator actions (any thread) -----------------------------------------------------
    def busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(self) -> dict:
        if self.busy():
            return {"ok": False, "message": "이미 진행 중"}
        starts = self.starts()
        missing = [n for n in start_names(self.plan) if n not in starts]
        if missing:
            return {"ok": False, "message": f"시작 위치를 먼저 저장하세요: {', '.join(missing)}"}
        if not remaining(self.plan, load_runs(self.runs_path)):
            return {"ok": False, "message": "남은 실행이 없습니다"}
        self.out.mkdir(parents=True, exist_ok=True)
        self._abort.clear()
        self._worker = threading.Thread(target=self._loop, daemon=True, name="experiment")
        self._worker.start()
        return {"ok": True}

    def act(self, name: str, **data) -> dict:
        with self._lock:
            expected = {"placed": "place", "skip": "place", "verdict": "verdict", "continue": "manual"}
            if self._phase != expected.get(name):
                return {"ok": False, "message": f"지금은 '{name}'을(를) 할 수 없습니다 ({PHASE_TEXT.get(self._phase)})"}
            self._action = (name, data)
            self._lock.notify_all()
        return {"ok": True}

    def abort(self) -> dict:
        if not self.busy():
            return {"ok": True, "message": "진행 중인 실험 없음"}
        self._abort.set()
        with self._lock:
            self._lock.notify_all()
        return {"ok": True, "message": "중단 요청: 진행 중인 이동이나 명령을 취소합니다"}

    def starts(self) -> dict:
        if not self.starts_path.is_file():
            return {}
        return json.loads(self.starts_path.read_text(encoding="utf-8"))

    def save_start(self, name: str) -> dict:
        if self.busy():
            return {"ok": False, "message": "실험 중에는 시작 위치를 바꿀 수 없습니다"}
        if name not in start_names(self.plan):
            return {"ok": False, "message": f"모르는 시작 위치: {name}"}
        pose = self.pose.current_pose()
        if pose is None:
            return {"ok": False, "message": "위치 추정(TF map→base_link)이 없습니다: 측위를 확인하세요"}
        starts = self.starts()
        starts[name] = [round(pose.x, 3), round(pose.y, 3), round(pose.yaw, 4)]
        self.out.mkdir(parents=True, exist_ok=True)
        self.starts_path.write_text(json.dumps(starts, ensure_ascii=False, indent=1), encoding="utf-8")
        self._check = None
        return {"ok": True, "message": f"{name} 저장: ({pose.x:.2f}, {pose.y:.2f})"}

    def check(self) -> dict:
        rows = check_starts(self.config, self.plan, self.starts())
        self._check = rows
        bad = [r for r in rows if not r["ok"]]
        return {"ok": not bad, "rows": rows,
                "message": "모든 시나리오 1-30 m" if not bad else f"문제 {len(bad)}개: 시작 위치를 다시 정하세요"}

    def view(self) -> dict:
        runs = load_runs(self.runs_path)
        with self._lock:
            row, record = self._row, self._record
            order = None
            if row is not None:
                order = list(pair_order(row["scenario"]))
            return {
                "plan": str(self.plan_path), "out": str(self.out), "phase": self._phase,
                "phase_text": PHASE_TEXT.get(self._phase, self._phase), "text": self._text,
                "busy": self.busy(), "progress": progress(self.plan, runs),
                "recording": self.recorder is not None and self.recorder.active,
                "record_target": self.recorder.target if self.recorder is not None else "",
                "scenario": None if row is None else {
                    "scenario": row["scenario"], "object": row["object"], "room": row.get("room"),
                    "desk": row.get("desk_hint"), "start": row["start"], "typical": row.get("typical"),
                    "order": order},
                "variant": self._variant, "record": record,
                "starts": self.starts(), "start_names": start_names(self.plan), "check": self._check,
                "runs": [{k: r.get(k) for k in ("run", "scenario", "variant", "object", "status", "target_status",
                                                "steps_completed", "duration_s", "valid", "note", "found_photo")}
                         for r in runs[-10:]][::-1],
            }
