"""Run a search command in the toy environment with the simulated robot.

    python3 scripts/run_search_demo.py --targets cup remote --hide remote
    python3 scripts/run_search_demo.py --targets cup --move cup_2=7.0,2.4,0.5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from athome.execution.command import CommandExecutor  # noqa: E402
from athome.execution.fake import FakeClock  # noqa: E402
from athome.execution.visit import VisitExecutor  # noqa: E402
from athome.navigation import NavigationPlanner  # noqa: E402
from athome.scene_graph.query import SceneGraph  # noqa: E402
from athome.schemas import Pose2D  # noqa: E402
from athome.search import SearchSession  # noqa: E402
from athome.testing import toy_env  # noqa: E402
from athome.testing.sim import SimRobot  # noqa: E402


def parse_move(items):
    moved = {}
    for item in items:
        oid, xyz = item.split("=")
        moved[oid] = tuple(float(v) for v in xyz.split(","))
    return moved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--hide", nargs="*", default=[],
                        help="categories removed from the scene graph (unknown)")
    parser.add_argument("--move", nargs="*", default=[],
                        help="object_id=x,y,z actual position in the world")
    parser.add_argument("--start", nargs=3, type=float, default=[2.5, 3.0, 0.0])
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()

    clock = FakeClock()
    robot = SimRobot(clock, toy_env.toy_world(parse_move(args.move)),
                     Pose2D(*args.start))
    graph = SceneGraph(toy_env.toy_graph(args.hide))
    session = SearchSession(
        graph, NavigationPlanner(toy_env.toy_grid()), args.targets,
        max_steps=args.max_steps)

    for t in session.targets:
        print(f"target {t.name}: {'known' if t.known else 'unknown'}")

    def on_step(decision, record, outcome):
        found = f" -> 발견: {record.found}" if record.found else ""
        if record.covered:
            found += f" (함께 관측: {record.covered})"
        print(
            f"[{clock():6.1f}s] step {decision.step} {decision.target:8s} "
            f"{decision.stage.value:10s} {decision.location_id:32s} "
            f"cost {decision.cost:5.2f}m  {outcome.status.value}{found}")

    command = CommandExecutor(
        VisitExecutor(robot, robot, clock), robot, "toy", on_step=on_step)
    command.start(session)

    result = None
    while result is None and clock() < 3600:
        result = command.step()
        robot.update(0.05)

    print(f"\n결과: {result.status.value} {result.reason} {result.detail}")
    for t in result.targets:
        where = f" @ {t.found_location}" if t.found_location else ""
        print(f"  {t.name}: {t.status.value}{where} {t.detail}")
    print(f"탐색 {len(result.history)} step, 이동 거리 "
          f"{robot.distance_traveled:.1f}m, 시뮬레이션 시간 {clock():.0f}s")


if __name__ == "__main__":
    main()
