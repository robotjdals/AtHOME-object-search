"""Re-verify recorded Teacher selections with the current GT rules (no API).

The Teacher's trajectory and choices are kept; only meta.verification (and
passed_by_step_cost) is recomputed from the recorded state (pose, visited)
with athome.training.verify.GroundTruth. Writes a new run folder with the
same layout, so export_sft_dataset.py can use it; the source is recorded.
Layout: ATHOME_HM3D_LAYOUT (the run's layout).
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from athome.schemas import Pose2D
from athome.training.verify import GroundTruth

from scene_episodes import SceneProblems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Teacher 실행 폴더")
    parser.add_argument("--output", type=Path, required=True, help="새 폴더")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    episodes = json.loads((args.run / "episodes.json").read_text(encoding="utf-8"))
    scene = SceneProblems()
    if episodes["summary"]["layout"] != scene.layout.source:
        raise SystemExit("실행 폴더의 layout이 현재 ATHOME_HM3D_LAYOUT과 다릅니다.")
    problems = scene.by_key()
    raw = (args.run / "queries.jsonl").read_bytes()
    out, changes, passed = [], Counter(), Counter()
    truths = {}
    for line in raw.decode("utf-8").splitlines():
        record = json.loads(line)
        old = (record.get("meta") or {}).get("verification")
        if old is not None:
            key = (record["meta"]["component"], record["target"])
            if key not in truths:
                p = problems[key]
                truths[key] = GroundTruth(p.graph, p.navigation(), p.observer, p.target_ids,
                                          p.gt_workspaces, gt_rooms=p.gt_rooms)
            check = truths[key].verify(record["stage"], record["selected_id"],
                                       [c["candidate_id"] for c in record["candidates"]],
                                       Pose2D(*record["meta"]["pose"]), set(record["meta"]["visited"]))
            record["meta"]["verification"] = check
            record["meta"]["passed_by_step_cost"] = {
                k: v in check["valid_candidates"] for k, v in record["selection_by_step_cost"].items()}
            changes[f"{record['stage']}:{old['passed']}->{check['passed']}"] += 1
            passed[check["passed"]] += 1
        out.append(json.dumps(record, ensure_ascii=False))
    args.output.mkdir(parents=True)
    (args.output / "queries.jsonl").write_text("\n".join(out) + "\n", encoding="utf-8")
    summary = dict(episodes["summary"])
    summary.update({"passed": passed[True], "reverified_from": str(args.run),
                    "reverified_source_sha256": hashlib.sha256(raw).hexdigest(),
                    "verification_rules": {"room": "containment", "workspace": "observation",
                                           "standalone": "observation"}})
    episodes["summary"] = summary
    (args.output / "episodes.json").write_text(json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8")
    print("통과", passed[True], "변화", dict(changes), "→", args.output)


if __name__ == "__main__":
    main()
