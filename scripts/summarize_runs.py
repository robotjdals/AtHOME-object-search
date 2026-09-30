"""Summary of several run reports (search experiments on the robot).

    python3 scripts/summarize_runs.py outputs/runs/*/report/report.json

One row per command, then the summary over finished commands (success
rate, mean active time, distance, visits, cost; athome.run_report.summarize).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from athome.run_report import summarize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", type=Path, nargs="+", help="run_report의 report.json")
    parser.add_argument("--output", type=Path, help="요약 JSON 저장 (새 파일)")
    args = parser.parse_args()

    metrics = []
    print("| 기록 | 명령 | 목표 | 결과 | 찾음 | 시간 [s] | 이동 [m] | 방문 | 비용 [m] | 일시정지 | 대체 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        for m in report["runs"]:
            metrics.append(m)
            print(f"| {Path(report['bag']).name} | {m['run']} | "
                  f"{', '.join(t['name'] for t in m['targets'])} | {m['status']} | "
                  f"{m['found']}/{len(m['targets'])} | {m['active_s']} | {m['distance_m']} | "
                  f"{m['visits']} | {m['cost_m']} | {m['pauses']} | {m['fallbacks']} |")
    summary = summarize(metrics)
    print(f"\n명령 {summary['runs']}개 (끝남 {summary['finished']}): 성공률 {summary['success_rate']}, "
          f"평균 시간 {summary['mean_active_s']} s, 이동 {summary['mean_distance_m']} m, "
          f"방문 {summary['mean_visits']}, 비용 {summary['mean_cost_m']} m; "
          f"일시정지 {summary['pauses']}, 대체 {summary['fallbacks']}")
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            json.dump({"reports": [str(p) for p in args.reports], "summary": summary,
                       "runs": metrics}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
