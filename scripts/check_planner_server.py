"""Check an LLM Planner server (vLLM, OpenAI compatible) end to end.

    python3 scripts/check_planner_server.py --robot-config configs/robot/demo.yaml \
        --base-url http://<host>:<port> [--records out.jsonl]

The API key (planner.llm.api_key_env) is read from --env-file by this tool,
never sourced into the shell.

1. GET /v1/models: every model name of planner.llm.models is served.
2. Toy searches with the robot's own client (LLMPolicy via make_policy) and
   search loop (SearchSession, CommandExecutor, simulated robot): the real
   prompts of every stage go to the server. Each query is recorded
   (athome.search.decision_log) and summarized: calls, fallbacks (timeouts,
   HTTP errors, outputs outside the candidates), latency.

Only planner.llm of the robot config is read (the map need not exist here).
Exit code 1 if a model is missing or any query fell back to minimum cost.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from athome.execution.command import CommandExecutor  # noqa: E402
from athome.execution.fake import FakeClock  # noqa: E402
from athome.execution.visit import VisitExecutor  # noqa: E402
from athome.inference.chat import served_models  # noqa: E402
from athome.inference.factory import _env_key, make_policy  # noqa: E402
from athome.navigation import NavigationPlanner  # noqa: E402
from athome.scene_graph.query import SceneGraph  # noqa: E402
from athome.schemas import Pose2D  # noqa: E402
from athome.search import SearchSession  # noqa: E402
from athome.search.decision_log import DecisionLog  # noqa: E402
from athome.testing import toy_env  # noqa: E402
from athome.testing.sim import SimRobot  # noqa: E402

# (target, categories hidden from the graph): unknown targets reach every stage.
EPISODES = [("remote", ["remote"]), ("kettle", ["kettle"]), ("banana", [])]


def run_episode(policy, target, hidden):
    clock = FakeClock()
    robot = SimRobot(clock, toy_env.toy_world(), Pose2D(2.5, 3.0, 0.0))
    session = SearchSession(SceneGraph(toy_env.toy_graph(hidden)),
                            NavigationPlanner(toy_env.toy_grid()), [target], policy=policy)
    command = CommandExecutor(VisitExecutor(robot, robot, clock), robot, "toy")
    command.start(session)
    result = None
    while result is None and clock() < 3600:
        result = command.step()
        robot.update(0.05)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--base-url", help="planner.llm.base_url 대신 사용")
    parser.add_argument("--records", type=Path, help="질의 기록 JSONL 저장 경로")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)     # the process environment wins

    llm = dict(yaml.safe_load(args.robot_config.read_text(encoding="utf-8"))["planner"]["llm"])
    if args.base_url:
        llm["base_url"] = args.base_url
    print(f"서버: {llm['base_url']}, 모델: {llm['models']}")

    ok = True
    try:
        served = served_models(llm["base_url"], _env_key(llm.get("api_key_env")),
                               float(llm.get("timeout", 5.0)))
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"/v1/models 실패: {e}")
    missing = sorted(set(llm["models"].values()) - set(served))
    print(f"/v1/models: {served}")
    if missing:
        print(f"없는 모델: {missing}")
        ok = False

    records = []
    policy = DecisionLog(make_policy({"type": "llm", "llm": llm}), records.append)
    for target, hidden in EPISODES:
        before = len(records)
        result = run_episode(policy, target, hidden)
        found = [t.name for t in result.targets if t.status.value == "found"]
        print(f"[{target}] {result.status.value}, 발견 {found}, 탐색 {len(result.history)} step, "
              f"질의 {len(records) - before}")

    asked = [r for r in records if r["raw"] is not None]       # single candidates skip the LLM
    fallbacks = [r for r in records if r["fallback"]]
    print(f"\n질의 {len(records)}개 (LLM 호출 {len(asked)}), 대체 {len(fallbacks)}개")
    for stage in ("room", "workspace", "standalone"):
        lat = sorted(r["latency_s"] for r in asked if r["stage"] == stage)
        if lat:
            print(f"  {stage:10s} {len(lat):3d}회  지연 중앙값 {statistics.median(lat):.2f}s  "
                  f"최대 {lat[-1]:.2f}s")
    for r in fallbacks[:5]:
        print(f"  대체 [{r['stage']}] {r['error']}")
    if asked:
        print(f"\n응답 예: {asked[0]['raw']!r}")
    if args.records:
        with args.records.open("x", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("기록:", args.records)
    ok = ok and not fallbacks and bool(asked)
    print("결과:", "정상" if ok else "문제 있음")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
