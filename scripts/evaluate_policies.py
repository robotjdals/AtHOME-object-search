"""Evaluate search policies on fixed start states (proposal 6-1 Test).

Metrics per episode and in total (Anderson et al. 2018, Habitat ObjectNav):
- success: a target instance observed before the session ends
- SPL = success x l* / max(l*, travelled), l* from export_start_states.py
- visits (Search Location visits) and travelled distance [m]
- with --time-budget: modelled time and the robot's time budget
  (robot config search.*), the same stop rule as on the robot.

Policies: mincost (nearest first), random (seeded), and later the Student.
Layout: ATHOME_HM3D_LAYOUT (must match the states file).
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import yaml

from athome.execution.visit import VisitConfig
from athome.schemas import Pose2D
from athome.search import SearchSession
from athome.search.policy import MinCostPolicy, RandomPolicy, ShuffledPolicy
from athome.symbolic import SymbolicEnvironment, TimeModel

from scene_episodes import SceneProblems


def make_policy(name, seed):
    if name == "mincost":
        return MinCostPolicy()
    if name == "random":
        return RandomPolicy(seed)
    raise SystemExit(f"알 수 없는 정책: {name}")


def time_budget(layout):
    search = (yaml.safe_load(layout.robot.read_text(encoding="utf-8")).get("search") or {})
    missing = [k for k in ("time_budget_s", "nominal_speed_mps", "rotation_speed_radps") if search.get(k) is None]
    if missing:
        raise SystemExit(f"로봇 설정 search 값이 비어 있습니다(확정 필요): {missing}")
    return float(search["time_budget_s"]), TimeModel(
        float(search["nominal_speed_mps"]), float(search["rotation_speed_radps"]),
        VisitConfig().observation_window)


def run(problem, state, policy, max_steps, budget):
    start = Pose2D(*problem.grid.to_xy(tuple(state["start_row_col"])), 0.0)
    budget_s, model = budget or (None, None)
    env = SymbolicEnvironment(problem.grid, start, problem.world, problem.observer,
                              problem.surface.height_at, problem.heading_count, model)
    session = SearchSession(problem.graph, problem.navigation(), [problem.target], policy=policy,
                            max_steps=max_steps, time_budget_s=budget_s,
                            elapsed_s=env.elapsed_s if model else None,
                            coverage=problem.coverage())
    while True:
        decision = session.next_decision(env.pose)
        if decision is None:
            break
        session.report(decision, env.visit(decision))
    success = session.targets[0].status.value == "found"
    l_star = state["oracle_distance_m"]
    if not success or l_star is None:
        spl = 0.0
    else:  # found without moving when the target is in view at the start (l* = 0)
        longest = max(l_star, env.distance_m)
        spl = l_star / longest if longest > 0 else 1.0
    return {"state_id": state["state_id"], "target": problem.target, "component": problem.component,
            "success": success, "spl": spl, "visits": env.visits, "distance_m": env.distance_m,
            "oracle_distance_m": l_star, "session_status": session.status.value,
            **({"elapsed_s": env.elapsed_s()} if model else {})}


def summarize(rows):
    n = len(rows)
    return {"episodes": n,
            "success_rate": sum(r["success"] for r in rows) / n if n else None,
            "spl": sum(r["spl"] for r in rows) / n if n else None,
            "mean_visits": sum(r["visits"] for r in rows) / n if n else None,
            "mean_distance_m": sum(r["distance_m"] for r in rows) / n if n else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--policy", choices=["mincost", "random"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10000, help="기본: 사실상 제한 없음")
    parser.add_argument("--time-budget", action="store_true", help="로봇 설정의 시간 예산 적용")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    states = [json.loads(l) for l in args.states.read_text(encoding="utf-8").splitlines() if l.strip()]
    scene = SceneProblems()
    if any(s["layout"] != scene.layout.source for s in states):
        raise SystemExit("시작 상태 파일의 layout이 현재 ATHOME_HM3D_LAYOUT과 다릅니다.")
    problems = scene.by_key()
    budget = time_budget(scene.layout) if args.time_budget else None
    rows = []
    for k, state in enumerate(states):
        # Candidates in a seeded random order per query; the same seed for
        # every policy, so policies are compared on the same orders.
        policy = ShuffledPolicy(make_policy(args.policy, args.seed + k), f"order:{args.seed}:{k}")
        rows.append(run(problems[(state["component"], state["target"])], state, policy, args.max_steps, budget))
    by_target = defaultdict(list)
    for r in rows:
        by_target[r["target"]].append(r)
    result = {"policy": args.policy, "seed": args.seed, "states": str(args.states),
              "max_steps": args.max_steps, "time_budget": bool(args.time_budget),
              "summary": summarize(rows), "by_target": {t: summarize(v) for t, v in sorted(by_target.items())},
              "episodes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(args.policy, json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
