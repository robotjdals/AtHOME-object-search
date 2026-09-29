"""Findability rule check on recorded Teacher starts (no API).

For every Teacher episode of one scene, whether its start is findable under
the ``any_goal`` and ``every_goal`` rules (athome.training.findability),
next to whether the Teacher and the baselines (outputs/baseline_compare_*)
found the target. Layout: ATHOME_HM3D_LAYOUT.
"""
import argparse
import json
from pathlib import Path

from athome.schemas import Pose2D
from athome.training.findability import findable_from

from scene_episodes import SceneProblems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    episodes = json.loads((args.teacher_run / "episodes.json").read_text(encoding="utf-8"))["episodes"]
    problems = SceneProblems().by_key()
    rows = []
    for ep in episodes:
        p = problems[(ep["component"], ep["target"])]
        start = Pose2D(*p.grid.to_xy(tuple(ep["start_row_col"])), 0.0)
        nav = p.navigation()
        rows.append({"episode_id": ep["episode_id"], "teacher_found": ep["status"] == "found",
                     **{rule: findable_from(nav, p.graph.locations, start, p.observer, p.target_ids, rule)
                        is not None for rule in ("any_goal", "every_goal")}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
