from athome.execution.command import CommandExecutor, CommandStatus
from athome.execution.fake import FakeClock
from athome.execution.visit import VisitExecutor
from athome.navigation import NavigationPlanner
from athome.scene_graph.query import SceneGraph
from athome.schemas import MotionReason, Pose2D
from athome.search import SearchSession, TargetStatus
from athome.testing import toy_env
from athome.testing.sim import SimRobot

GRID = toy_env.toy_grid()


def setup(targets, hide=(), moved=None, start=Pose2D(2.5, 3.0, 0.0), **robot_kwargs):
    clock = FakeClock()
    robot = SimRobot(clock, toy_env.toy_world(moved), start, **robot_kwargs)
    s = SearchSession(
        SceneGraph(toy_env.toy_graph(hide)), NavigationPlanner(GRID), targets)
    command = CommandExecutor(VisitExecutor(robot, robot, clock), robot, "toy")
    command.start(s)
    return clock, robot, command


def run(clock, robot, command, until=None, limit=1200.0):
    result = None
    while result is None and clock() < limit:
        if until is not None and until():
            return None
        result = command.step()
        robot.update(0.05)
    return result


def test_command_finds_known_and_unknown_targets():
    clock, robot, command = setup(["cup", "remote"], hide=["remote"])
    result = run(clock, robot, command)
    assert result.status == CommandStatus.COMPLETED
    assert [t.status for t in result.targets] == [TargetStatus.FOUND] * 2
    assert result.targets[1].found_location == "workspace:R_B:coffee table_7"


def test_exhausted_navigation_retries_pause_without_blacklist():
    # Every approach to the table fails.
    def blocked(goal):
        return MotionReason.BLOCKED if goal.x < 2.6 and goal.y < 2.4 else None

    clock, robot, command = setup(["remote"], hide=["remote"],
                                  navigation_failure=blocked)
    result = run(clock, robot, command)
    assert result.status == CommandStatus.PAUSED
    first = result.history[0]
    assert first.decision.location_id == "workspace:R_A:table_1"
    assert first.status.value == "failed"
    assert not command.session.visited
    assert not hasattr(command.session, "excluded")
    assert len(result.history) == 1
    assert result.location_id == "workspace:R_A:table_1"


def test_resume_after_cleared_failure_retries_same_location():
    obstacle = {"present": True}

    def blocked(goal):
        if obstacle["present"] and goal.x < 2.6 and goal.y < 2.4:
            return MotionReason.BLOCKED
        return None

    clock, robot, command = setup(["remote"], hide=["remote"],
                                  navigation_failure=blocked)
    assert run(clock, robot, command).status == CommandStatus.PAUSED
    obstacle["present"] = False           # operator removed the cause
    command.resume()
    result = run(clock, robot, command, limit=clock() + 1200.0)
    assert result.status == CommandStatus.COMPLETED
    assert result.history[1].decision.location_id == "workspace:R_A:table_1"
    assert result.history[1].status.value == "completed"
    assert result.targets[0].status == TargetStatus.FOUND


def test_cancel_during_visit_waits_for_stop():
    clock, robot, command = setup(["banana"])
    run(clock, robot, command, until=lambda: clock() > 2.0)
    command.cancel()
    result = run(clock, robot, command)
    assert result.status == CommandStatus.CANCELED
    assert result.history == []


def test_start_outside_free_space_pauses():
    clock, robot, command = setup(["cup"], start=Pose2D(1.5, 1.4, 0.0))
    result = run(clock, robot, command)
    assert result.status == CommandStatus.PAUSED
    assert result.reason == "start_not_free"


class FlakyPose:
    """Pose source that drops out between ``off`` and ``on`` seconds."""

    def __init__(self, robot, clock, off, on):
        self.robot, self.clock, self.off, self.on = robot, clock, off, on

    def current_pose(self):
        if self.off <= self.clock() < self.on:
            return None
        return self.robot.current_pose()


def command_with(clock, robot, targets, hide=(), pose=None):
    s = SearchSession(
        SceneGraph(toy_env.toy_graph(hide)), NavigationPlanner(GRID), targets)
    command = CommandExecutor(VisitExecutor(robot, robot, clock), pose or robot, "toy")
    command.start(s)
    return command


def test_short_localization_gap_is_waited_out():
    clock = FakeClock()
    robot = SimRobot(clock, toy_env.toy_world(), Pose2D(2.5, 3.0, 0.0))
    command = command_with(clock, robot, ["cup"], pose=FlakyPose(robot, clock, 0.0, 3.0))
    result = run(clock, robot, command)
    assert result.status == CommandStatus.COMPLETED


def test_long_localization_loss_pauses():
    clock = FakeClock()
    robot = SimRobot(clock, toy_env.toy_world(), Pose2D(2.5, 3.0, 0.0))
    command = command_with(clock, robot, ["cup"], pose=FlakyPose(robot, clock, 0.0, 99.0))
    result = run(clock, robot, command)
    assert result.status == CommandStatus.PAUSED
    assert result.reason == "localization_unavailable"
    assert clock() > 5.0


def test_internal_error_stops_robot_before_pausing(monkeypatch):
    clock, robot, command = setup(["banana"])
    run(clock, robot, command, until=lambda: clock() > 2.0)
    assert robot.handles and robot.handles[-1].state.value == "active"

    def boom(self):
        raise RuntimeError("bug")

    monkeypatch.setattr(VisitExecutor, "_step_motion", boom, raising=True)
    command.step()
    assert robot.handles[-1].cancel_requested
    monkeypatch.undo()
    result = run(clock, robot, command)
    assert result.status == CommandStatus.PAUSED
    assert result.reason == "internal_error" and "bug" in result.detail
    assert robot.handles[-1].state.value == "done"


def test_resume_after_pause_keeps_visited():
    clock, robot, command = setup(["banana"])
    # Perception outage after the first location.
    run(clock, robot, command, until=lambda: len(command.session.visited) == 1)
    robot.perception.frames.clear()
    real_publish = robot.perception.publish
    robot.perception.publish = lambda *a, **k: None     # motion keeps working
    result = run(clock, robot, command)
    assert result.status == CommandStatus.PAUSED
    assert result.reason == "perception_lost"
    first = set(command.session.visited)

    robot.perception.publish = real_publish
    command.resume()
    result = run(clock, robot, command)
    assert result.status == CommandStatus.COMPLETED
    assert first <= command.session.visited
    visited = [r.decision.location_id for r in result.history if r.status.value == "completed"]
    assert len(visited) == len(set(visited))
