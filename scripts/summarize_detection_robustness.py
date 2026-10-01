"""Robustness of the planner to missed detections (evaluate_student.py
episodes --detection-recall): the same start states evaluated at detection
recall 1 and at lower recalls.

For each recall: success rate, SPL, visits, cost and SR within k visits of
the Student and of the nearest-first and random baselines (same draws for
every policy: athome.symbolic.environment.DetectionModel), how the failed
searches ended (visit cap or every location visited), how often a target in
view was missed, and server fallbacks. The change of each policy from recall
1 is paired by episode, with 95% bootstrap intervals over episodes and over
buildings (scripts/compare_episode_evals.bootstrap).

    python scripts/summarize_detection_robustness.py outputs/eval_server/v1_server_episodes_holdout.json \
        outputs/eval_server/v1_server_episodes_holdout_r0.9.json ... [--output report.json]
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from compare_episode_evals import BUDGETS, bootstrap

POLICIES = ("student", "mincost", "random")
METRICS = [("SR", lambda s: float(s["found"])), ("SPL", lambda s: s["spl"]),
           ("visits", lambda s: float(s["visits"])), ("cost", lambda s: s["cost"])]
METRICS += [(f"SR@{k}", lambda s, k=k: float(s["found"] and s["visits"] <= k)) for k in BUDGETS]


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["summary"].get("detection_recall", 1.0), {r["state_id"]: r for r in data["rows"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path, help="검출률 1 결과")
    parser.add_argument("others", type=Path, nargs="+", help="검출률 < 1 결과들 (같은 시작 상태)")
    parser.add_argument("--subset", choices=["all", "objectnav"], default="all",
                        help="objectnav: 목표까지 1 m 이상 30 m 미만인 시작점만")
    parser.add_argument("--output", type=Path, help="요약 JSON (새 파일)")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    ref_recall, ref = load(args.reference)
    if ref_recall != 1.0:
        raise SystemExit(f"기준 파일의 검출률이 1이 아닙니다: {ref_recall}")
    runs = sorted((load(p) for p in args.others), key=lambda x: -x[0])
    ids = sorted(i for i in ref if args.subset == "all" or 1.0 <= ref[i]["oracle_distance_m"] < 30.0)
    for recall, rows in runs:
        if not set(ids) <= rows.keys():
            raise SystemExit(f"검출률 {recall}: 시작 상태가 기준과 다릅니다.")
    groups = [i.split(":")[0] for i in ids]
    report = {"reference": str(args.reference), "subset": args.subset, "episodes": len(ids),
              "buildings": len(set(groups)), "recalls": {}}
    print(f"{len(ids)} episodes, {len(set(groups))} buildings ({args.subset})")
    for recall, rows in [(1.0, ref)] + runs:
        entry = {}
        print(f"\n== detection recall {recall}")
        for pol in POLICIES:
            stats = {}
            for name, f in METRICS:
                values = [f(rows[i][pol]) for i in ids]
                stats[name] = {"mean": float(np.mean(values))}
                if recall < 1.0:
                    diffs = [v - f(ref[i][pol]) for v, i in zip(values, ids)]
                    ci_e, ci_b = bootstrap(diffs, groups)
                    stats[name].update(change=float(np.mean(diffs)), ci_episodes=ci_e, ci_buildings=ci_b)
            ends = Counter(rows[i][pol].get("end", "found" if rows[i][pol]["found"] else "unknown") for i in ids)
            missed = sum(rows[i][pol].get("target_missed_visits", 0) for i in ids)
            fallbacks = sum(rows[i][pol].get("policy_fallbacks", 0) for i in ids)
            entry[pol] = {"metrics": stats, "ends": dict(ends), "target_missed_visits": missed,
                          "policy_fallbacks": fallbacks}
            line = "  ".join(f"{n} {s['mean']:.3f}" + (f" ({s['change']:+.3f}, CI {s['ci_buildings'][0]:+.3f}~{s['ci_buildings'][1]:+.3f})"
                                                       if "change" in s and n in ("SR", "cost", "SR@5") else "")
                             for n, s in stats.items() if n in ("SR", "SPL", "visits", "cost", "SR@5"))
            print(f"{pol:8s} {line}")
            print(f"{'':8s} ends {dict(ends)}, target missed on {missed} visits, fallbacks {fallbacks}")
        report["recalls"][str(recall)] = entry
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
