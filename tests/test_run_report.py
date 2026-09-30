import json

import pytest

from athome.execution.command import CommandExecutor, CommandStatus
from athome.execution.fake import FakeClock
from athome.execution.run_log import RunLog
from athome.execution.visit import VisitExecutor
from athome.navigation import NavigationPlanner, OccupancyMap
from athome.run_report import build_runs, markdown, render_run, run_metrics, summarize
from athome.scene_graph.query import SceneGraph
from athome.schemas import MotionReason, Pose2D
from athome.search import SearchSession
from athome.search.decision_log import DecisionLog
from athome.search.policy import MinCostPolicy
from athome.testing import toy_env
from athome.testing.sim import SimRobot

GRID = toy_env.toy_grid()


class Recorder:
    """What the search server publishes and a bag records, on the node clock."""

    def __init__(self, clock):
        self.clock = clock
        self.events, self.decisions, self.trajectory = [], [], []

    def event(self, record):
        self.events.append(json.loads(json.dumps({**record, "stamp": self.clock()})))

    def decision(self, record):
        self.decisions.append(json.loads(json.dumps({**record, "stamp": self.clock()})))


def start(targets, hide=(), navigation_failure=None):
    clock = FakeClock()
    robot = SimRobot(clock, toy_env.toy_world(), Pose2D(2.5, 3.0, 0.0),
                     navigation_failure=navigation_failure)
    rec = Recorder(clock)
    session = SearchSession(SceneGraph(toy_env.toy_graph(hide)), NavigationPlanner(GRID), targets,
                            policy=DecisionLog(MinCostPolicy(), rec.decision, clock=clock))
    command = CommandExecutor(VisitExecutor(robot, robot, clock), robot, "toy",
                              observer=RunLog(rec.event))
    command.start(session)
    return clock, robot, command, rec


def run(clock, robot, command, rec, limit=1200.0):
    result = None
    while result is None and clock() < limit:
        result = command.step()
        robot.update(0.05)
        rec.trajectory.append((clock(), robot.pose.x, robot.pose.y, robot.pose.yaw))
    return result


def test_run_log_records_the_command():
    clock, robot, command, rec = start(["cup", "remote"], hide=["remote"])
    result = run(clock, robot, command, rec)
    assert result.status == CommandStatus.COMPLETED
    kinds = [e["event"] for e in rec.events]
    assert kinds[0] == "command_start" and kinds[-1] == "command_end"
    assert kinds.count("visit_start") == kinds.count("visit_end") == len(result.history)
    end = rec.events[-1]
    assert [t["status"] for t in end["targets"]] == ["found", "found"]
    assert end["targets"][1]["found_location"] == "workspace:R_B:coffee table_7"
    visit_ends = [e for e in rec.events if e["event"] == "visit_end"]
    assert any("remote" in e["found"] for e in visit_ends)
    assert all(e["frames"] > 0 and e["seen"] is not None for e in visit_ends)


def test_failing_sink_never_stops_the_search():
    errors = []

    def broken(record):
        raise OSError("publisher gone")

    clock = FakeClock()
    robot = SimRobot(clock, toy_env.toy_world(), Pose2D(2.5, 3.0, 0.0))
    log = RunLog(broken, on_error=errors.append)
    session = SearchSession(SceneGraph(toy_env.toy_graph()), NavigationPlanner(GRID), ["cup"])
    command = CommandExecutor(VisitExecutor(robot, robot, clock), robot, "toy", observer=log)
    command.start(session)
    result = None
    while result is None and clock() < 1200:
        result = command.step()
        robot.update(0.05)
    assert result.status == CommandStatus.COMPLETED
    assert log.errors == len(errors) > 0


def test_report_metrics_match_the_run():
    clock, robot, command, rec = start(["cup", "remote"], hide=["remote"])
    result = run(clock, robot, command, rec)
    runs = build_runs(rec.events, rec.decisions, rec.trajectory)
    assert len(runs) == 1
    m = run_metrics(runs[0], step_cost_m=3.0)
    assert m["status"] == "completed" and m["success"] and m["found"] == 2
    assert m["visits"] == len(result.history)
    assert m["distance_m"] == pytest.approx(robot.distance_traveled, abs=0.05)
    assert m["cost_m"] == pytest.approx(m["distance_m"] + 3.0 * m["visits"], abs=0.01)
    assert m["planner_queries"] == len(rec.decisions) and m["fallbacks"] == 0
    # Every planner query belongs to the visit it led to.
    assert sum(len(v.decisions) for v in runs[0].visits) == len(rec.decisions)
    assert sum(v.distance_m for v in runs[0].visits) <= m["distance_m"] + 0.01   # rounded


def test_pause_and_resume_stay_one_run():
    obstacle = {"present": True}

    def blocked(goal):
        return MotionReason.BLOCKED if obstacle["present"] and goal.x < 2.6 and goal.y < 2.4 else None

    clock, robot, command, rec = start(["remote"], hide=["remote"], navigation_failure=blocked)
    assert run(clock, robot, command, rec).status == CommandStatus.PAUSED
    clock.advance(60.0)                   # a person clears the way
    obstacle["present"] = False
    command.resume()
    assert run(clock, robot, command, rec, limit=clock() + 1200).status == CommandStatus.COMPLETED

    runs = build_runs(rec.events, rec.decisions, rec.trajectory)
    assert len(runs) == 1 and len(runs[0].ends) == 2 and len(runs[0].resumes) == 1
    m = run_metrics(runs[0], step_cost_m=3.0)
    assert m["pauses"] == 1 and m["status"] == "completed"
    assert m["duration_s"] - m["active_s"] == pytest.approx(60.0, abs=0.2)
    text = markdown(runs, [m], SceneGraph(toy_env.toy_graph()), {1: "run1.png"}, "test")
    assert "일시정지: navigation_failed (table)" in text
    assert "| 1 | remote |" in text and "실패 (navigation_failed)" in text


def test_recording_started_mid_command_is_marked_partial():
    clock, robot, command, rec = start(["cup"])
    run(clock, robot, command, rec)
    runs = build_runs(rec.events[1:], rec.decisions, rec.trajectory)
    assert runs[0].partial
    m = run_metrics(runs[0], step_cost_m=3.0)
    assert m["partial_recording"] and m["status"] == "completed"


def test_summary_over_runs():
    metrics = [
        {"status": "completed", "success": True, "active_s": 10.0, "distance_m": 5.0,
         "visits": 2, "cost_m": 11.0, "pauses": 0, "fallbacks": 0, "planner_queries": 3},
        {"status": "completed", "success": False, "active_s": 30.0, "distance_m": 9.0,
         "visits": 4, "cost_m": 21.0, "pauses": 1, "fallbacks": 1, "planner_queries": 6},
        {"status": "unfinished", "success": False, "active_s": 1.0, "distance_m": 0.0,
         "visits": 0, "cost_m": 0.0, "pauses": 0, "fallbacks": 0, "planner_queries": 0},
    ]
    s = summarize(metrics)
    assert s["runs"] == 3 and s["finished"] == 2 and s["success_rate"] == 0.5
    assert s["mean_cost_m"] == 16.0 and s["pauses"] == 1


def test_render_run_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    clock, robot, command, rec = start(["cup", "remote"], hide=["remote"])
    run(clock, robot, command, rec)
    runs = build_runs(rec.events, rec.decisions, rec.trajectory)
    m = run_metrics(runs[0], step_cost_m=3.0)
    occupancy = OccupancyMap(toy_env.toy_occupancy(), (0.0, 0.0), toy_env.RESOLUTION)
    render_run(tmp_path / "run1.png", runs[0], m, SceneGraph(toy_env.toy_graph()),
               occupancy, GRID, title="test")
    assert (tmp_path / "run1.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
