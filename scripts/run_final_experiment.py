"""Final experiment on the robot: every placement of an object the scene graph does
not hold is searched by the trained planner and by the untrained base model.

    source /opt/ros/humble/setup.bash && source install/setup.bash
    # once, the robot localized: stand it at each start place and save the poses
    python3 scripts/run_final_experiment.py save-starts --robot-config configs/robot/room_baram.yaml \\
        --plan experiments/room_baram_final/scenarios.csv --out outputs/final_experiment/<day>
    # the runs (resumes: pairs recorded as valid are skipped)
    python3 scripts/run_final_experiment.py run --robot-config ... --plan ... --out ...

The search stack (realtime_search.launch.py), localization and the motion module
run as for a demo. Per scenario the operator places the object (prompt), then
for each planner variant the script: sets the search server's planner_variant
parameter (refused while a command runs), drives the robot to the saved start
pose through the motion module (Nav2 path + NavigateToGoal, as the search
does), waits for fresh perception, records the actual start pose, sends the
target as a confirmed command (the command parser is not part of the compared
variable) and waits for the command's end. The operator then marks the run
valid or not (invalid: a cause unrelated to the system, see the protocol).
Ctrl-C cancels the running command or move (the robot stops) and exits.

Records (<out>/): starts.json, runs.jsonl (one line per run), events_<run>.jsonl
and decisions_<run>.jsonl (the search server's events and planner queries).
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Header, String

from athome.config import load_robot_config
from athome.execution import HandleState
from athome.schemas import MotionStatus, NavigationRequest, Pose2D
from athome_interfaces.action import SearchObjects
from athome_interfaces.msg import ObservedObjectArray
from athome_ros.adapters import RosMotionClient
from athome_ros.common import TfPoseSource
from athome_ros.conversions import stamp_to_sec

VARIANTS = ("trained", "base")
SEARCH_SERVER = "athome_search_server"
MOVE_TIMEOUT_S = 180.0
PERCEPTION_WAIT_S = 240.0       # the detector loads for 2-3 min after a launch
COMMAND_MARGIN_S = 150.0        # over the time budget: a visit under way ends first


def load_plan(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["scenario"] = int(r["scenario"])
    return rows


def pair_order(scenario: int):
    """Counterbalanced: odd scenarios run the trained planner first, even ones the base."""
    return VARIANTS if scenario % 2 == 1 else VARIANTS[::-1]


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


class Runner(Node):
    def __init__(self, config, out: Path):
        super().__init__("athome_experiment_runner")
        self.config = config
        self.out = out
        self.pose = TfPoseSource(self, config.map_frame, config.base_frame)
        self.motion = RosMotionClient(
            self, "/athome/navigate_to_goal", self.pose,
            planner_action="/compute_path_to_pose" if config.path_planner == "nav2" else None)
        self._heartbeat = self.create_publisher(Header, "/athome/executor/heartbeat", 10)
        self._heartbeat_timer = None
        self._search = ActionClient(self, SearchObjects, "/athome/search_objects")
        self._params = self.create_client(SetParameters, f"/{SEARCH_SERVER}/set_parameters")
        self.events, self.decisions = [], []
        self.state = None
        self._perception_at = None
        self._stale_s = float(config.observation.get("perception_timeout_s", 15.0))
        self.create_subscription(String, "/athome/search/events",
                                 lambda m: self.events.append(json.loads(m.data)), 100)
        self.create_subscription(String, "/athome/planner/decision",
                                 lambda m: self.decisions.append(json.loads(m.data)), 100)
        self.create_subscription(String, "/athome/search/state", self._on_state, 10)
        self.create_subscription(ObservedObjectArray, "/athome/perception/objects", self._on_objects, 10)

    # --- helpers ---------------------------------------------------------------------------
    def _on_state(self, msg):
        self.state = msg.data

    def _on_objects(self, msg):
        self._perception_at = stamp_to_sec(msg.header.stamp)

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def spin_until(self, done, seconds: float) -> bool:
        end = time.monotonic() + seconds
        while not done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        return done()

    def current_pose(self, wait_s: float = 3.0):
        self.spin_until(lambda: self.pose.current_pose() is not None, wait_s)
        return self.pose.current_pose()

    def perception_age(self):
        return None if self._perception_at is None else self.now() - self._perception_at

    def wait_perception(self, seconds: float = PERCEPTION_WAIT_S) -> bool:
        def fresh():
            age = self.perception_age()
            return age is not None and age <= self._stale_s
        if not fresh():
            print(f"  인지 결과를 기다리는 중 (최대 {seconds:.0f}초)...")
        return self.spin_until(fresh, seconds)

    def wait_search_server(self, seconds: float = 60.0) -> bool:
        return (self._search.wait_for_server(timeout_sec=seconds)
                and self._params.wait_for_service(timeout_sec=seconds)
                and self.spin_until(lambda: self.state == "idle", seconds))

    # --- steps -----------------------------------------------------------------------------
    def set_variant(self, variant: str) -> None:
        value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=variant)
        request = SetParameters.Request(parameters=[Parameter(name="planner_variant", value=value)])
        future = self._params.call_async(request)
        if not self.spin_until(future.done, 10.0) or future.result() is None:
            raise RuntimeError("탐색 서버의 planner_variant 설정 응답 없음")
        result = future.result().results[0]
        if not result.successful:
            raise RuntimeError(f"planner_variant={variant} 거절: {result.reason}")

    def go_to(self, goal: Pose2D, request_id: str) -> dict:
        """Drive to ``goal`` through the motion module; the runner beats the
        executor heartbeat only while its own move is active."""
        if not self.spin_until(self.motion.servers_ready, 30.0):
            return {"status": "failed", "reason": "no_server", "detail": "주행 모듈 또는 Nav2 planner 없음"}
        self._heartbeat_timer = self.create_timer(0.5, lambda: self._heartbeat.publish(Header()))
        started = time.monotonic()
        handle = self.motion.navigate(NavigationRequest(
            request_id=request_id, goal=goal, map_version=self.config.map_version,
            timeout_sec=MOVE_TIMEOUT_S))
        try:
            if not self.spin_until(lambda: handle.state in (HandleState.DONE, HandleState.REJECTED),
                                   MOVE_TIMEOUT_S + 10):
                handle.request_cancel()
                self.spin_until(lambda: handle.state == HandleState.DONE, 15.0)
                return {"status": "failed", "reason": "timeout", "detail": f"{MOVE_TIMEOUT_S:.0f}초 안에 도착하지 않음",
                        "seconds": round(time.monotonic() - started, 1)}
            if handle.state == HandleState.REJECTED:
                return {"status": "failed", "reason": "rejected", "detail": "주행 모듈이 요청 거절",
                        "seconds": round(time.monotonic() - started, 1)}
            r = handle.result
            return {"status": r.status.value, "reason": r.reason.value, "detail": r.detail,
                    "seconds": round(time.monotonic() - started, 1)}
        except KeyboardInterrupt:
            handle.request_cancel()
            self.spin_until(lambda: handle.state == HandleState.DONE, 15.0)
            raise
        finally:
            self._heartbeat_timer.cancel()
            self.destroy_timer(self._heartbeat_timer)
            self._heartbeat_timer = None

    def search(self, target: str, timeout_s: float):
        """One confirmed command; (events, decisions, action status). The command's
        end is read from the events (a goal response can be lost just after a
        server starts, as scripts/run_hm3d_episodes.py found)."""
        self.events, self.decisions = [], []

        def seen(kind):
            return any(e.get("event") == kind for e in self.events)

        goal = SearchObjects.Goal(targets=[target])
        handle, deadline = None, time.monotonic() + 30
        while not seen("command_start"):
            future = self._search.send_goal_async(goal)
            self.spin_until(lambda: future.done() or seen("command_start"), 10)
            if future.done() and future.result() is not None and future.result().accepted:
                handle = future.result()
                break
            if time.monotonic() > deadline:
                raise RuntimeError("탐색 명령 거절됨 (탐색 서버 로그 확인)")
            self.spin_until(lambda: seen("command_start"), 1.0)
        result = handle.get_result_async() if handle is not None else None
        try:
            if not self.spin_until(lambda: seen("command_end"), timeout_s):
                if handle is not None:
                    handle.cancel_goal_async()
                self.spin_until(lambda: seen("command_end"), 30.0)
                raise RuntimeError(f"명령이 {timeout_s:.0f}초 안에 끝나지 않아 취소함")
        except KeyboardInterrupt:
            if handle is not None:
                print("\n취소 요청: 로봇이 멈출 때까지 기다리는 중...")
                handle.cancel_goal_async()
                self.spin_until(lambda: seen("command_end"), 30.0)
            raise
        self.spin_until(lambda: result is None or result.done(), 2.0)
        status = result.result().status if result is not None and result.done() else None
        return list(self.events), list(self.decisions), status


# --- records --------------------------------------------------------------------------------
def summarize(events, decisions):
    start = next((e for e in events if e["event"] == "command_start"), {})
    end = next((e for e in events if e["event"] == "command_end"), {})
    visits = [e for e in events if e["event"] == "visit_end"]
    target = (end.get("targets") or [{}])[0]
    return {
        "planner_variant_used": (start.get("setup") or {}).get("planner_variant"),
        "status": end.get("status"), "reason": end.get("reason"), "detail": end.get("detail"),
        "steps_logged": end.get("steps"),
        "steps_completed": sum(1 for v in visits if v.get("status") == "completed"),
        "nav_attempts": sum(int(v.get("nav_attempts") or 0) for v in visits),
        "visited": [v.get("location_id") for v in visits],
        "target_status": target.get("status"),
        "found_location": target.get("found_location"),
        "found_label": target.get("found_label"),
        "found_position": target.get("found_position"),
        "decisions": len(decisions),
        "fallbacks": sum(1 for d in decisions if d.get("fallback")),
    }


def write_lines(path: Path, records) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_runs(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- commands -------------------------------------------------------------------------------
def save_starts(args, config, plan):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(r["start"] for r in plan))
    starts = {}
    if args.from_plan:
        for r in plan:
            starts.setdefault(r["start"], [float(r["start_x"]), float(r["start_y"]), 0.0])
    else:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = Runner(config, out)
        try:
            for name in names:
                while True:
                    if ask(f"로봇을 '{name}' 시작 위치에 세우고 Enter (q: 중단): ") == "q":
                        return
                    pose = node.current_pose()
                    if pose is None:
                        print("  위치 추정(TF map→base_link)이 없음: 측위를 확인하고 다시")
                        continue
                    print(f"  {name}: x {pose.x:.2f}, y {pose.y:.2f}, yaw {math.degrees(pose.yaw):.0f}°")
                    starts[name] = [round(pose.x, 3), round(pose.y, 3), round(pose.yaw, 4)]
                    break
        finally:
            node.destroy_node()
            rclpy.shutdown()
    (out / "starts.json").write_text(json.dumps(starts, ensure_ascii=False, indent=1))
    print(f"저장: {out / 'starts.json'}")
    check_starts(config, plan, starts)


def check_starts(config, plan, starts) -> None:
    """Shortest path from each start to its scenario's location (the SPL rule
    needs 1-30 m and a reachable goal), with the search server's own planner."""
    from athome.navigation import load_map_server
    from athome.navigation.planner import NavigationPlanner
    from athome.scene_graph.location_policy import excluded_categories
    from athome.scene_graph.query import STANDALONE, SceneGraph

    graph = SceneGraph(json.loads(config.graph_path.read_text(encoding="utf-8")),
                       excluded_categories=excluded_categories(config.search_locations))
    grid = load_map_server(config.map_yaml, config.map_inflation, config.unknown_as_occupied)
    nav = NavigationPlanner(grid, config.navigation_config(), config.sight_grid(), config.goal_rooms())
    for lid, loc in graph.locations.items():
        nav.add_location(lid, loc.bbox_min, loc.bbox_max, loc.kind == STANDALONE, loc.room_id,
                         graph.other_boxes(lid))
    problems = 0
    print("scenario  object      start         l* (m)")
    for r in plan:
        x, y, yaw = starts[r["start"]]
        try:
            cost = nav.evaluate(Pose2D(x, y, yaw), [r["location_id"]]).get(r["location_id"])
        except Exception as e:  # noqa: BLE001 - reported per row
            cost = None
            print(f"  {r['scenario']:>2}: {e}")
        l_star = None if cost is None else cost.cost
        flag = "" if l_star is not None and 1.0 <= l_star <= 30.0 else "  <-- 1-30 m 밖 또는 경로 없음"
        problems += bool(flag)
        print(f"{r['scenario']:>8}  {r['object']:<10}  {r['start']:<12}  {'-' if l_star is None else f'{l_star:.2f}'}{flag}")
    print("모두 1-30 m" if not problems else f"문제 {problems}개: 시작 위치를 다시 정하거나 시나리오를 바꿀 것")


def run(args, config, plan):
    out = Path(args.out)
    starts = json.loads((out / "starts.json").read_text(encoding="utf-8"))
    runs_path = out / "runs.jsonl"
    done = {(r["scenario"], r["variant"]) for r in load_runs(runs_path) if r.get("valid")}
    todo = [(r, v) for r in plan for v in pair_order(r["scenario"]) if (r["scenario"], v) not in done]
    if not todo:
        print("남은 실행 없음")
        return
    print(f"실행 {len(todo)}개 남음 (유효 기록 {len(done)}개)")
    timeout_s = float(config.time_budget_s or 600.0) + COMMAND_MARGIN_S

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Runner(config, out)
    index = len(load_runs(runs_path))
    try:
        print("탐색 서버를 기다리는 중...")
        if not node.wait_search_server():
            raise RuntimeError("탐색 서버가 대기 상태가 아님 (실행 중인 명령 또는 costmap 대기)")
        placed = None
        for row, variant in todo:
            if placed != row["scenario"]:
                answer = ask(f"\n[시나리오 {row['scenario']}] {row['object']}을(를) {row['room']} {row['desk_hint']}에 놓고 "
                             f"Enter (s: 이 시나리오 건너뜀, q: 종료): ")
                if answer == "q":
                    return
                if answer == "s":
                    placed = None
                    continue
                placed = row["scenario"]
            index += 1
            print(f"\n--- 실행 {index}: 시나리오 {row['scenario']} {row['object']} · planner {variant} · "
                  f"시작 {row['start']}")
            record = {"run": index, "scenario": row["scenario"], "variant": variant, "object": row["object"],
                      "location_id": row["location_id"], "start": row["start"],
                      "start_pose_saved": starts[row["start"]], "typical": row.get("typical"),
                      "clock": time.strftime("%Y-%m-%d %H:%M:%S")}
            node.set_variant(variant)
            x, y, yaw = starts[row["start"]]
            print(f"  시작 위치로 이동 ({x:.2f}, {y:.2f})...")
            move = node.go_to(Pose2D(x, y, yaw), f"exp{index}_start")
            record["move_to_start"] = move
            if move["status"] != MotionStatus.SUCCEEDED.value:
                print(f"  이동 실패: {move['reason']} {move['detail']}")
                if ask("  로봇을 손으로 시작 위치에 놓고 Enter (q: 종료): ") == "q":
                    return
            if not node.wait_perception():
                print("  인지 결과 없음: 카메라, 인지 노드, 검출 서버 확인")
                if ask("  해결했으면 Enter (q: 종료): ") == "q":
                    return
            pose = node.current_pose()
            record["start_pose_actual"] = None if pose is None else [round(pose.x, 3), round(pose.y, 3),
                                                                     round(pose.yaw, 4)]
            record["t_start"] = node.now()
            print(f"  명령: {row['object']} (최대 {timeout_s:.0f}초)")
            events, decisions, status = node.search(row["object"], timeout_s)
            record["t_end"] = node.now()
            record["duration_s"] = round(record["t_end"] - record["t_start"], 1)
            record["action_status"] = status
            record.update(summarize(events, decisions))
            write_lines(out / f"events_{index}.jsonl", events)
            write_lines(out / f"decisions_{index}.jsonl", decisions)
            record["events_file"], record["decisions_file"] = f"events_{index}.jsonl", f"decisions_{index}.jsonl"
            if record["planner_variant_used"] != variant:
                print(f"  경고: 탐색 서버가 기록한 planner는 {record['planner_variant_used']}")
            print(f"  결과: {record['status']} ({record['reason'] or ''}) · 방문 {record['steps_completed']} · "
                  f"{record['duration_s']}초 · 발견 {record['found_label']} @ {record['found_location']}"
                  f" · planner 대체 {record['fallbacks']}/{record['decisions']}")
            answer = ask("  유효한 실행? [Enter=유효 / x 이유: 무효 (다시 시행) / q: 종료]: ")
            if answer == "q":
                record["valid"], record["note"] = False, "quit"
                with open(runs_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                return
            record["valid"] = not answer.startswith("x")
            record["note"] = answer[1:].strip() if answer.startswith("x") else ""
            with open(runs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not record["valid"]:
                todo.insert(todo.index((row, variant)) + 1, (row, variant))   # the same run again
    except KeyboardInterrupt:
        print("\n중단")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["save-starts", "run"])
    parser.add_argument("--robot-config", required=True)
    parser.add_argument("--plan", required=True, help="scenarios.csv")
    parser.add_argument("--out", required=True, help="record folder (starts.json, runs.jsonl, ...)")
    parser.add_argument("--from-plan", action="store_true",
                        help="save-starts: the plan's start_x/start_y with yaw 0 instead of the robot's poses")
    args = parser.parse_args()
    config = load_robot_config(args.robot_config)
    plan = load_plan(args.plan)
    if args.command == "save-starts":
        save_starts(args, config, plan)
    else:
        run(args, config, plan)


if __name__ == "__main__":
    sys.exit(main())
