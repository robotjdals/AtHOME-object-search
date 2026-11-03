"""The status line follows the executors: driving, turning, observing each
heading, confirming a possible target, inference, result."""

from athome.execution import VisitConfig, VisitExecutor
from athome.execution.command import CommandExecutor
from athome.execution.fake import FakeClock
from athome.execution.status import describe
from athome.navigation import NavigationPlanner
from athome.scene_graph.query import SceneGraph
from athome.schemas import Pose2D
from athome.search import SearchSession
from athome.search.policy import Stage
from athome.testing import toy_env
from athome.testing.sim import SimRobot


def run_states(confirmations=2):
    clock = FakeClock()
    robot = SimRobot(clock, toy_env.toy_world(), Pose2D(2.5, 3.0, 0.0), perception_rate=2.0)
    graph = SceneGraph(toy_env.toy_graph(["remote"]))
    s = SearchSession(graph, NavigationPlanner(toy_env.toy_grid()), ["remote"], confirmations=confirmations)
    visit = VisitExecutor(robot, robot, clock, VisitConfig(observation_window=0.5, observation_min_frames=1))
    command = CommandExecutor(visit, robot, "toy")
    seen = [describe(command, visit.progress(), graph, clock=clock)]
    command.start(s)
    result = None
    while result is None and clock() < 3600:
        result = command.step()
        robot.update(0.05)
        st = describe(command, visit.progress(), graph, clock=clock, pose=robot.current_pose())
        if st["text"].split(":")[0] != seen[-1]["text"].split(":")[0] or st["state"] != seen[-1]["state"]:
            seen.append(st)
    return seen


def test_states_of_a_command_in_order():
    seen = run_states()
    states = [s["state"] for s in seen]
    assert states[0] == "idle" and states[-1] == "done"
    for s in ("navigating", "rotating", "observing"):
        assert s in states
    nav = next(s for s in seen if s["state"] == "navigating")
    assert nav["text"].startswith("이동 중 → ") and "남은 직선" in nav["text"] and nav["location_name"]
    obs = next(s for s in seen if s["state"] == "observing")
    assert obs["heading"][1] == 6 and obs["text"].startswith(f"관측 {obs['heading'][0]}/6")
    assert seen[-1]["text"].startswith("완료") and seen[-1]["last"].startswith("step ")


def test_possible_target_waiting_for_its_second_frame_is_shown():
    assert "confirming" in [s["state"] for s in run_states(confirmations=2)]


def test_inference_parsing_and_confirmation_texts():
    clock = FakeClock()
    graph = SceneGraph(toy_env.toy_graph(["remote"]))
    s = SearchSession(graph, NavigationPlanner(toy_env.toy_grid()), ["remote"])
    robot = SimRobot(clock, toy_env.toy_world(), Pose2D(2.5, 3.0, 0.0))
    visit = VisitExecutor(robot, robot, clock, VisitConfig())
    command = CommandExecutor(visit, robot, "toy")
    assert describe(command, {}, graph, clock=clock, parsing=("컵 찾아줘", clock()))["state"] == "parsing"
    assert describe(command, {}, graph, clock=clock, awaiting=["cup"])["text"] == "사용자 확인 대기: cup"
    command.start(s)
    st = describe(command, visit.progress(), graph, query=(Stage.ROOM, "remote", clock()), clock=clock)
    assert st["state"] == "inferring" and st["text"].startswith("추론 중: 방 선택 (remote)")
