"""Object search system node: serves the SearchObjects action.

    ros2 run athome_ros search_server --ros-args -p robot_config:=configs/robot/toy.yaml
    ros2 action send_goal --feedback /athome/search_objects \
        athome_interfaces/action/SearchObjects "{targets: [cup, remote]}"

Threading: the search executor, motion/perception adapters and TF all run
in the node's default (mutually exclusive) callback group, so the core
never sees concurrent callbacks. The action execute callback only waits
for the result in a separate reentrant group.
"""

import json
import math
import threading
import time
import traceback
from typing import Optional

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from athome.config import load_robot_config
from athome.execution.command import (
    CommandExecutor,
    CommandPhase,
    CommandResult,
    CommandStatus,
)
from athome.execution.run_log import RunLog
from athome.execution.visit import VisitConfig, VisitExecutor, observation_yaw_tolerance
from athome.inference.command_parser import COMMAND_PROMPT_VERSION, CommandParseError
from athome.inference.factory import make_command_parser, make_matcher, make_policy
from athome.navigation import (
    GridMap,
    NavigationConfig,
    NavigationPlanner,
    load_map_server,
    traversable_from_costmap,
)
from athome.scene_graph.location_policy import excluded_categories
from athome.search.coverage import RoomCoverage, occupancy_line_of_sight, room_points_from_labels
from athome.scene_graph.query import DEFAULT_EXCLUDED_CATEGORIES, SceneGraph
from athome.scene_graph.vocabulary import Vocabulary
from athome.search.decision_log import DecisionLog
from athome.search import SearchSession, TargetStatus
from athome_interfaces.action import SearchObjects
from athome_interfaces.srv import ParseCommand
from athome_interfaces.msg import TargetResult
from athome_ros.adapters import RosMotionClient, RosObservationSource, node_now
from athome_ros.common import HeartbeatPublisher, TfPoseSource
from athome_ros.visualization import LATCHED, grid_message, search_markers

COSTMAP_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)

_STATUS = {
    CommandStatus.COMPLETED: SearchObjects.Result.STATUS_COMPLETED,
    CommandStatus.MAX_STEPS: SearchObjects.Result.STATUS_MAX_STEPS,
    CommandStatus.TIME_BUDGET: SearchObjects.Result.STATUS_TIME_BUDGET,
    CommandStatus.CANCELED: SearchObjects.Result.STATUS_CANCELED,
    CommandStatus.PAUSED: SearchObjects.Result.STATUS_PAUSED,
}


class _Job:
    def __init__(self, goal_handle):
        self.goal_handle = goal_handle
        self.targets = [t for t in goal_handle.request.targets if t.strip()]
        self.max_steps = goal_handle.request.max_steps
        self.resume = goal_handle.request.resume
        self.instruction = goal_handle.request.instruction
        self.received = time.monotonic()
        self.started = False
        self.cancel_requested = False
        self.cancel_sent = False
        self.result: Optional[CommandResult] = None
        self.done = threading.Event()


class SearchServer(Node):
    def __init__(self):
        super().__init__("athome_search_server")
        p = self.declare_parameter
        config = load_robot_config(p("robot_config", "").value)
        self._default_max_steps = p("max_steps", 30).value
        self._startup_timeout = p("startup_timeout", 10.0).value
        nav_action = p("navigate_action", "/athome/navigate_to_goal").value
        objects_topic = p("objects_topic", "/athome/perception/objects").value
        heartbeat_topic = p("heartbeat_topic", "/athome/executor/heartbeat").value
        rate = p("rate_hz", 20.0).value
        if config.graph_path is None:
            raise ValueError("robot_config에 scene_graph.path 필요")

        # Same Search Location policy as the training data (planner inputs must match).
        excluded = (excluded_categories(config.search_locations)
                    if config.search_locations else DEFAULT_EXCLUDED_CATEGORIES)
        # Targets and perception labels grouped into the training GT's categories.
        vocabulary = Vocabulary.load(config.target_categories)
        self._vocabulary = vocabulary
        # A target also counts as found/Known through its subtypes (training GT rule).
        self._graph = SceneGraph.load(config.graph_path, excluded_categories=excluded,
                                      category_key=vocabulary.canonical,
                                      target_keys=vocabulary.target_keys)
        if not self._graph.room_floor_z:
            self.get_logger().warn(
                "그래프에 방 바닥 높이 없음: 1.5 m 탐색 높이 범위 미적용 (학습 데이터와 다름)")
        self._config = config
        self._coverage_setup = self._load_coverage(config)
        self._navigation: Optional[NavigationPlanner] = None
        self._viz_grid = self.create_publisher(OccupancyGrid, "/athome/viz/nav_grid", LATCHED)
        use_nav2 = config.path_planner == "nav2"
        if use_nav2:
            # Candidate costs on exactly the grid the Nav2 planner uses.
            self.create_subscription(
                OccupancyGrid, p("costmap_topic", "/global_costmap/costmap").value,
                self._on_costmap, COSTMAP_QOS)
        else:
            self._set_grid(load_map_server(
                config.map_yaml, config.inflation_radius, config.unknown_as_occupied))

        # Every planner query as JSON (HTTP calls are not in rosbag otherwise).
        self._decisions = self.create_publisher(
            String, p("decision_topic", "/athome/planner/decision").value, 50)
        # Command, visits and result as JSON for the run report (an action
        # result is not in rosbag otherwise).
        self._events = self.create_publisher(
            String, p("events_topic", "/athome/search/events").value, 50)
        run_log = RunLog(self._publish_event, on_error=self._on_event_error, setup={
            "map_version": config.map_version,
            "graph": str(config.graph_path),
            "planner": config.planner.get("type", "min_cost"),
        })
        self._policy = DecisionLog(make_policy(config.planner), self._publish_decision)
        self._matcher = make_matcher(config.matcher, vocabulary.canonical, vocabulary.target_keys)
        # API key is read when the first instruction arrives, not at startup.
        self._parser_config = config.command_parser
        self._parser = None
        self._parser_lock = threading.Lock()
        # Human-in-the-loop: instructions are parsed by /athome/parse_command,
        # confirmed by the operator and sent as targets (InteLiPlan-style
        # confirmation); unconfirmed instruction goals are rejected.
        self._confirm_instructions = p("confirm_instructions", True).value

        self._pose = TfPoseSource(self, config.map_frame, config.base_frame)
        self._motion = RosMotionClient(
            self, nav_action, self._pose,
            planner_action=p("planner_action", "/compute_path_to_pose").value if use_nav2 else None)
        visit = VisitExecutor(
            self._motion,
            RosObservationSource(self, objects_topic, frame_id=config.map_frame),
            lambda: node_now(self),
            self._visit_config(config),
        )
        self._command = CommandExecutor(
            visit, self._pose, config.map_version, on_step=self._log_step, observer=run_log)

        self._lock = threading.Lock()
        self._job: Optional[_Job] = None
        self._last_decision = None

        HeartbeatPublisher(self, heartbeat_topic)
        self.create_timer(1.0 / rate, self._tick)

        self._map_frame = config.map_frame
        self._viz = self.create_publisher(MarkerArray, "/athome/viz/search", 1)
        self.create_timer(1.0, self._publish_markers)
        # Own group: parsing waits for the LLM server and must not block the search loop.
        self.create_service(ParseCommand, "/athome/parse_command", self._on_parse,
                            callback_group=MutuallyExclusiveCallbackGroup())
        ActionServer(
            self, SearchObjects, "/athome/search_objects",
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(
            f"준비: Room {len(self._graph.rooms)}, 탐색 위치 "
            f"{len(self._graph.locations)}, 지도 {config.map_version}, "
            f"planner {config.planner.get('type')}, matcher {config.matcher.get('type')}, "
            f"경로 {config.path_planner}")

    def _visit_config(self, config) -> VisitConfig:
        """Observation rotations accurate enough for neighbouring views to
        overlap (full 360-degree coverage, the training observation model)."""
        if config.camera_hfov_deg is None:
            self.get_logger().warn(
                "sensors.camera_hfov_deg 없음: 관측 회전 yaw 허용 오차를 주행 모듈 기본값에 맡김 "
                "(방향 사이 사각지대 가능)")
            return VisitConfig()
        heading_count = VisitConfig().heading_count
        tolerance = observation_yaw_tolerance(math.radians(config.camera_hfov_deg), heading_count)
        self.get_logger().info(
            f"관측 {heading_count}방향, 회전 yaw 허용 오차 {math.degrees(tolerance):.1f}° "
            f"(카메라 시야 {config.camera_hfov_deg}°)")
        return VisitConfig(rotation_yaw_tolerance=tolerance)

    def _load_coverage(self, config):
        """Room samples (room map of build_scene_graph.py) and occupancy line
        of sight: the observed part of each room is shown at room selection,
        as in the planner's training data."""
        rooms_path = config.graph_path.with_name(config.graph_path.stem + ".rooms.npz")
        if config.observation_range_m is None or not rooms_path.is_file():
            self.get_logger().warn(
                f"방 관측 비율 비활성({rooms_path.name} 또는 search.observation_range_m 없음): "
                "플래너 입력이 학습 데이터와 다름")
            return None
        raw = load_map_server(config.map_yaml, 0.0, config.unknown_as_occupied)
        with np.load(rooms_path, allow_pickle=False) as saved:
            points = room_points_from_labels(saved["labels"], tuple(saved["origin_xy_m"]),
                                             float(saved["resolution_m"].item()))
        missing = set(self._graph.rooms) - set(points)
        if missing:
            self.get_logger().warn(f"방 지도에 없는 Room: {sorted(missing)}")
        sees = occupancy_line_of_sight(~raw.free, raw.origin, raw.resolution,
                                       config.observation_range_m)
        return points, sees

    def _set_grid(self, grid) -> None:
        navigation = NavigationPlanner(grid, NavigationConfig(
            goal_offset=self._config.goal_offset, goal_clearance=self._config.goal_clearance,
            goal_max_offset=self._config.goal_max_offset))
        for lid, loc in self._graph.locations.items():
            navigation.add_location(lid, loc.bbox_min, loc.bbox_max)
        self._navigation = navigation
        self._viz_grid.publish(grid_message(
            grid, self._config.map_frame, self.get_clock().now().to_msg()))

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        # Static environment: the first costmap is used for the whole run.
        if self._navigation is not None:
            return
        if msg.header.frame_id != self._config.map_frame:
            self.get_logger().error(f"costmap frame {msg.header.frame_id} != {self._config.map_frame}")
            return
        info = msg.info
        values = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        grid = GridMap(traversable_from_costmap(values),
                       (info.origin.position.x, info.origin.position.y), info.resolution)
        self._set_grid(grid)
        self.get_logger().info(
            f"Nav2 costmap 수신: {info.width}x{info.height}, 주행 가능 {int(grid.free.sum())} cells")

    # --- action (reentrant group) -----------------------------------------

    def _on_goal(self, request):
        with self._lock:
            if self._job is not None:
                self.get_logger().warn("실행 중인 명령이 있어 거절")
                return GoalResponse.REJECT
        if self._navigation is None:
            self.get_logger().warn("costmap 대기 중이라 거절")
            return GoalResponse.REJECT
        if request.resume:
            last = self._command.result
            if last is None or last.status != CommandStatus.PAUSED:
                self.get_logger().warn("재개할 일시중지 명령이 없어 거절")
                return GoalResponse.REJECT
            return GoalResponse.ACCEPT
        if not any(t.strip() for t in request.targets):
            if not request.instruction.strip() or not self._parser_config:
                self.get_logger().warn("targets 없음 (자연어 명령은 command_parser 설정 필요)")
                return GoalResponse.REJECT
            if self._confirm_instructions:
                self.get_logger().warn(
                    "확인되지 않은 자연어 명령 거절: /athome/parse_command로 해석하고 "
                    "확인한 targets로 요청 (confirm_instructions:=false로 끌 수 있음)")
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _parse(self, instruction: str):
        with self._parser_lock:
            if self._parser is None:
                self._parser = make_command_parser(self._parser_config,
                                                   self._vocabulary.categories)
            parser = self._parser
        return parser(instruction)

    def _on_parse(self, request, response):
        response.prompt_version = COMMAND_PROMPT_VERSION
        if not self._parser_config:
            response.success, response.message = False, "command_parser 설정 없음"
            return response
        try:
            targets = self._parse(request.instruction)
        except (CommandParseError, RuntimeError) as e:
            response.success, response.message = False, str(e)
            self.get_logger().info(f"명령 해석 실패: {request.instruction!r}: {e}")
            return response
        response.success = True
        response.targets = targets
        response.in_training_vocabulary = [
            not self._vocabulary.categories or self._vocabulary.in_training_vocabulary(t)
            for t in targets]
        self.get_logger().info(f"명령 해석(확인 대기): {request.instruction!r} -> {targets}")
        return response

    def _execute(self, goal_handle):
        job = _Job(goal_handle)
        with self._lock:
            self._job = job
        while not job.done.wait(0.1):
            if goal_handle.is_cancel_requested:
                job.cancel_requested = True

        result = self._result_msg(job.result)
        if job.result.status == CommandStatus.CANCELED:
            goal_handle.canceled()
        elif job.result.status == CommandStatus.PAUSED:
            goal_handle.abort()
        else:
            goal_handle.succeed()
        return result

    # --- search loop (default group) --------------------------------------

    def _tick(self) -> None:
        # An exception here would stop the executor (and all callbacks).
        # The command executor already guards the search itself.
        try:
            self._tick_job()
        except Exception:  # noqa: BLE001
            self.get_logger().error(
                "tick 예외:\n" + traceback.format_exc(), throttle_duration_sec=5.0)

    def _tick_job(self) -> None:
        job = self._job
        if job is None:
            return
        if not job.started:
            self._try_start(job)
            return
        if job.cancel_requested and not job.cancel_sent:
            job.cancel_sent = True
            self._command.cancel()

        result = self._command.step()
        decision = self._command.decision
        if decision is not None and decision is not self._last_decision:
            self._last_decision = decision
            self._publish_feedback(job, decision)
        if result is not None:
            self._finish(job, result)

    def _try_start(self, job: _Job) -> None:
        if job.cancel_requested:
            self._finish(job, CommandResult(CommandStatus.CANCELED, "canceled_by_request"))
            return
        missing = None
        if not self._motion.servers_ready():
            missing = "motion_unavailable"
        elif self._pose.current_pose() is None:
            missing = "localization_unavailable"
        if missing is not None:
            if time.monotonic() - job.received > self._startup_timeout:
                self._finish(job, CommandResult(CommandStatus.PAUSED, missing))
            return

        if job.resume:
            try:
                self._command.resume()
            except RuntimeError as e:
                self._finish(job, CommandResult(CommandStatus.PAUSED, "not_resumable", str(e)))
                return
            self.get_logger().info("일시중지된 명령 재개 (Visited 유지)")
            job.started = True
            return

        if not job.targets:
            try:
                job.targets = self._parse(job.instruction)
            except (CommandParseError, RuntimeError) as e:
                self._finish(job, CommandResult(CommandStatus.PAUSED, "instruction_not_parsed", str(e)))
                return
            self.get_logger().info(f"명령 해석: {job.instruction!r} -> {job.targets}")

        # New session per command: Visited is reset. The time budget counts
        # from the start of the search (robot config search.time_budget_s).
        started = node_now(self)
        budget = self._config.time_budget_s
        session = SearchSession(
            self._graph, self._navigation, job.targets,
            policy=self._policy, matcher=self._matcher,
            max_steps=job.max_steps or self._default_max_steps,
            time_budget_s=budget,
            elapsed_s=(lambda: node_now(self) - started) if budget is not None else None,
            coverage=RoomCoverage(*self._coverage_setup) if self._coverage_setup else None,
        )
        for t in session.targets:
            self.get_logger().info(f"Target {t.name}: {'known' if t.known else 'unknown'}")
            if self._vocabulary.categories and not self._vocabulary.in_training_vocabulary(t.name):
                # Still searched: the planner generalizes, matching relies on perception.
                self.get_logger().warn(f"Target {t.name}: 학습 어휘 밖 목표 (Planner 일반화에 의존)")
        self._command.start(session)
        job.started = True

    def _finish(self, job: _Job, result: CommandResult) -> None:
        # rclpy forbids changing severity at one call site: separate calls.
        message = f"명령 종료: {result.status.value} {result.reason} {result.detail}"
        if result.location_id:
            message += f" (위치 {result.location_id})"
        if result.status == CommandStatus.PAUSED and result.location_id:
            message += " - 원인 해결 후 resume 시 같은 위치부터 다시 시도"
        if result.status == CommandStatus.COMPLETED:
            self.get_logger().info(message)
        else:
            self.get_logger().warn(message)
        if result.reason == "internal_error" and self._command.error is not None:
            self.get_logger().error("".join(traceback.format_exception(
                type(self._command.error), self._command.error,
                self._command.error.__traceback__)))
        with self._lock:
            self._job = None
        job.result = result
        job.done.set()

    def _publish_feedback(self, job: _Job, decision) -> None:
        self.get_logger().info(
            f"step {decision.step} [{decision.target}] {decision.stage.value} "
            f"→ {decision.location_id} (A* {decision.cost:.2f} m)")
        fb = SearchObjects.Feedback()
        fb.step = decision.step
        fb.target = decision.target
        fb.stage = decision.stage.value
        fb.room_id = decision.room_id
        fb.location_id = decision.location_id
        fb.path_cost = float(decision.cost)
        fb.found_targets = [
            t.name for t in self._command.session.targets
            if t.status == TargetStatus.FOUND
        ]
        job.goal_handle.publish_feedback(fb)

    def _publish_event(self, record: dict) -> None:
        record["stamp"] = self.get_clock().now().nanoseconds * 1e-9
        self._events.publish(String(data=json.dumps(record, ensure_ascii=False)))

    def _on_event_error(self, error: Exception) -> None:
        self.get_logger().error(f"탐색 기록 발행 실패: {error}", throttle_duration_sec=10.0)

    def _publish_decision(self, record: dict) -> None:
        record["stamp"] = self.get_clock().now().nanoseconds * 1e-9
        self._decisions.publish(String(data=json.dumps(record, ensure_ascii=False)))
        if record["fallback"]:
            self.get_logger().warn(
                f"planner {record['stage']} 대체(최소 비용): {record['error']}")

    def _log_step(self, decision, record, outcome) -> None:
        extra = f", 발견: {record.found}" if record.found else ""
        if record.policy_fallback:
            extra += f", planner 대체(최소 비용): {record.fallback_reason}"
        self.get_logger().info(
            f"step {decision.step} 결과: {outcome.status.value} ({outcome.reason.value}){extra}")

    def _publish_markers(self) -> None:
        try:
            self._viz.publish(search_markers(
                self._graph, self._command.session, self._command.decision,
                self._map_frame, self.get_clock().now().to_msg()))
        except Exception:  # noqa: BLE001 - visualization must never stop the search
            self.get_logger().error("marker 예외:\n" + traceback.format_exc(),
                                    throttle_duration_sec=10.0)

    def stop_motion(self, executor, timeout: float = 8.0) -> None:
        """On shutdown: cancel a running command and wait for the stop."""
        if self._command.phase not in (CommandPhase.VISITING, CommandPhase.ABORTING):
            return
        self.get_logger().warn("종료 요청: 주행 취소 후 정지 확인 대기")
        job = self._job
        if job is not None:
            job.cancel_requested = True
        else:
            self._command.cancel()
        deadline = time.monotonic() + timeout
        while self._command.phase != CommandPhase.DONE and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if self._command.phase != CommandPhase.DONE:
            self.get_logger().error("정지 확인 없이 종료 (주행 모듈 heartbeat 정지에 의존)")

    @staticmethod
    def _result_msg(result: CommandResult) -> SearchObjects.Result:
        msg = SearchObjects.Result()
        msg.status = _STATUS[result.status]
        msg.reason = result.reason
        msg.detail = result.detail
        msg.location_id = result.location_id
        msg.steps = len(result.history)
        for t in result.targets:
            tr = TargetResult()
            tr.name = t.name
            tr.known = t.known
            tr.status = t.status.value
            tr.detail = t.detail
            if t.found_location:
                tr.found_location = t.found_location
            if t.found_object is not None:
                tr.found_label = t.found_object.label
                tr.found_position.x, tr.found_position.y, tr.found_position.z = (
                    float(v) for v in t.found_object.centroid)
            msg.targets.append(tr)
        return msg


def main(args=None):
    # Keep the context alive on Ctrl-C to cancel the motion before exiting.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = SearchServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        try:
            node.stop_motion(executor)
        except KeyboardInterrupt:
            pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
