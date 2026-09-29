"""Episode-level efficiency of the Teacher trajectories against baselines.

For every Teacher episode of one scene (outputs/teacher_<v>/episodes_<scene>),
the nearest-first (MinCostPolicy) and a uniformly random policy (seeded per
episode) search from the same component, target and start cell, with the same
session settings (visit cap, room coverage, observation model, headings).
Success, distance, visits and SPL (Anderson et al. 2018; l* = shortest path to
a view of the target, athome.training.findability.oracle_distance) are
written per episode, paired, so the policies are compared on identical starts.
No API calls. Layout: ATHOME_HM3D_LAYOUT.
"""
import argparse
import json
from pathlib import Path
import zlib

from athome.schemas import Pose2D
from athome.search import SearchSession
from athome.search.policy import MinCostPolicy, RandomPolicy
from athome.symbolic.environment import SymbolicEnvironment
from athome.training.findability import oracle_distance

from scene_episodes import SceneProblems


def spl(found, l_star, distance):
    if not found or l_star is None:
        return 0.0
    longest = max(l_star, distance)
    return l_star / longest if longest > 0 else 1.0


def run_policy(problem, cell, policy, max_steps):
    start = Pose2D(*problem.grid.to_xy(cell), 0.0)
    env = SymbolicEnvironment(problem.grid, start, problem.world, problem.observer,
                              problem.surface.height_at, problem.heading_count)
    # Room coverage only feeds the LLM prompt (observed fraction); these
    # baselines ignore it, so it is not tracked (it dominates the run time).
    session = SearchSession(problem.graph, problem.navigation(), [problem.target], policy=policy,
                            max_steps=max_steps)
    while True:
        decision = session.next_decision(env.pose)
        if decision is None:
            break
        session.report(decision, env.visit(decision))
    return {"found": session.targets[0].status.value == "found",
            "distance_m": env.distance_m, "steps": session.steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-run", type=Path, required=True, help="episodes_<scene> 폴더")
    parser.add_argument("--output", type=Path, required=True, help="새 .jsonl 파일")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    data = json.loads((args.teacher_run / "episodes.json").read_text(encoding="utf-8"))
    summary = data["summary"]
    problems = SceneProblems().by_key()
    rows = []
    for ep in data["episodes"]:
        p = problems[(ep["component"], ep["target"])]
        cell = tuple(ep["start_row_col"])
        start = Pose2D(*p.grid.to_xy(cell), 0.0)
        l_star = oracle_distance(p.navigation(), p.graph.locations, start, p.observer, p.target_ids)
        teacher = {"found": ep["status"] == "found", "distance_m": ep["distance_m"], "steps": ep["steps"]}
        seed = zlib.crc32(ep["episode_id"].encode())
        row = {"episode_id": ep["episode_id"], "component": ep["component"], "target": ep["target"],
               "oracle_distance_m": l_star, "teacher": teacher,
               "mincost": run_policy(p, cell, MinCostPolicy(), summary["max_steps"]),
               "random": run_policy(p, cell, RandomPolicy(seed), summary["max_steps"])}
        for name in ("teacher", "mincost", "random"):
            r = row[name]
            r["spl"] = spl(r["found"], l_star, r["distance_m"])
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{args.teacher_run.name}: 에피소드 {len(rows)} → {args.output}")


if __name__ == "__main__":
    main()
