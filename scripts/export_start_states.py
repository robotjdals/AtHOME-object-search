"""Fixed start states of one scene for evaluation and GRPO (proposal 6-1, 6-4).

For every component x target with a target instance in the component, up to
--starts start cells drawn uniformly (seeded) among cells from which the
target is findable (athome.training.findability), with the SPL reference
distance l* (shortest path to a view of the target). The same file is the
Test/Val episode set of evaluate_policies.py and the GRPO start states, so
Student, baselines and training use identical episodes.
Layout: ATHOME_HM3D_LAYOUT.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zlib

from athome.schemas import Pose2D
from athome.training.findability import findable_from, findable_starts, oracle_distance

from scene_episodes import SceneProblems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=int, default=5, help="component x target 당 시작 상태 수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--findable-rule", choices=["every_goal", "any_goal"], default="every_goal",
                        help="athome.training.findability 규칙 (every_goal: 방문하면 반드시 발견)")
    parser.add_argument("--output", type=Path, required=True, help="새 .jsonl 파일")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    scene = SceneProblems()
    rows, short = [], []
    for p in scene.problems():
        navigation = p.navigation()
        seed = args.seed + zlib.crc32(f"{p.component}:{p.target}".encode())
        starts = findable_starts(p.grid, args.starts, seed, lambda cell: findable_from(
            navigation, p.graph.locations, Pose2D(*p.grid.to_xy(cell), 0.0), p.observer,
            p.target_ids, args.findable_rule) is not None)
        if len(starts) < args.starts:
            short.append({"component": p.component, "target": p.target, "starts": len(starts)})
        for k, cell in enumerate(starts):
            start = Pose2D(*p.grid.to_xy(cell), 0.0)
            rows.append({
                "state_id": f"{scene.layout.scene_id}:{p.component}:{p.target}:{k}",
                "scene_id": scene.layout.scene_id, "layout": scene.layout.source,
                "component": p.component, "target": p.target, "start_row_col": list(cell),
                "oracle_distance_m": oracle_distance(navigation, p.graph.locations, start,
                                                     p.observer, p.target_ids),
                "gt_object_ids": sorted(o["object_id"] for o in p.gt), "seed": seed})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"layout": scene.layout.source, "layout_sha256": hashlib.sha256(
                    Path(scene.layout.source).read_bytes()).hexdigest() if scene.layout.source != "default" else None,
                "starts_per_problem": args.starts, "seed": args.seed, "states": len(rows),
                "problems_short_of_findable_starts": short,
                "start_policy": "uniform_free_cells_target_findable",
                "findable_rule": args.findable_rule}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    print(f"시작 상태 {len(rows)}개, 부족한 조합 {short} → {args.output}")


if __name__ == "__main__":
    main()
