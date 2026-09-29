"""GRPO rollout groups on fixed start states (proposal 6-4), stand-in policy.

For each start state, --group-size rollouts with independent policy seeds;
rewards, group-relative advantages and the reason a group is left out.
The random stand-in only exercises the pipeline; the Student sampler (vLLM,
Workspace Selection Adapter) plugs into the same ``policies`` mapping.
Layout: ATHOME_HM3D_LAYOUT (must match the states file).
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from athome.schemas import Pose2D
from athome.search.policy import MinCostPolicy, RandomPolicy, Stage
from athome.symbolic import SymbolicEnvironment
from athome.training.grpo import group_advantages, rollout

from scene_episodes import SceneProblems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--step-cost", type=float, default=3.0, help="lambda_s [m]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    states = [json.loads(l) for l in args.states.read_text(encoding="utf-8").splitlines() if l.strip()]
    scene = SceneProblems()
    if any(s["layout"] != scene.layout.source for s in states):
        raise SystemExit("시작 상태 파일의 layout이 현재 ATHOME_HM3D_LAYOUT과 다릅니다.")
    problems = scene.by_key()
    reasons, groups = Counter(), []
    for k, state in enumerate(states):
        p = problems[(state["component"], state["target"])]
        start = Pose2D(*p.grid.to_xy(tuple(state["start_row_col"])), 0.0)
        group = []
        for g in range(args.group_size):
            env = SymbolicEnvironment(p.grid, start, p.world, p.observer, p.surface.height_at, p.heading_count)
            # Stand-ins: fixed stages nearest-first, trained stage sampled.
            policies = {Stage.ROOM: MinCostPolicy(), Stage.STANDALONE: MinCostPolicy(),
                        Stage.WORKSPACE: RandomPolicy(args.seed + 1000 * k + g)}
            group.append(rollout(p.graph, p.navigation(), env, p.target, policies, args.step_cost,
                                 coverage=p.coverage(), shuffle_seed=f"{args.seed}:{k}:{g}"))
        why = group_advantages(group)
        reasons[why or "usable"] += 1
        groups.append({"state_id": state["state_id"], "excluded": why,
                       "rollouts": [asdict(r) for r in group]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for row in groups:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("그룹:", dict(reasons), "→", args.output)


if __name__ == "__main__":
    main()
