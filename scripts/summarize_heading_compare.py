"""Summary of scripts/compare_observation_headings.py: per planner and
condition the success rate, modelled search time, visits and headings, and
the paired time / success difference against the current robot setting
(sector6: 6 views, early stop) with 95% CIs from resampling buildings.

    python scripts/summarize_heading_compare.py outputs/heading_compare/val.jsonl outputs/heading_compare/test.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np

BASE = "sector6"


def ci(per_building, rng, n=2000):
    keys = list(per_building)
    stats = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys))
        vals = [v for i in pick for v in per_building[keys[i]]]
        stats.append(np.mean(vals))
    return np.percentile(stats, [2.5, 97.5])


def main():
    rows = [json.loads(l) for p in sys.argv[1:] for l in open(p, encoding="utf-8") if l.strip()]
    keys = sorted({k for r in rows for k in r if "/" in k})
    planners = sorted({k.split("/")[1] for k in keys})
    conditions = [c for c in ("disk6", "sector6", "front_first6", "front3", "cover") if any(k.startswith(c + "/") for k in keys)]
    rng = np.random.default_rng(0)
    out = {"episodes": len(rows), "buildings": len({r["scene_id"] for r in rows})}
    print(f"에피소드 {out['episodes']}개, 건물 {out['buildings']}개")
    for planner in planners:
        print(f"\n== {planner}")
        print(f"{'조건':8s} {'성공률':>7s} {'시간 평균':>9s} {'중앙':>7s} {'방문':>6s} {'방향/방문':>9s} {'Δ시간 vs sector6 (95% CI)':>30s} {'Δ성공':>18s}")
        for c in conditions:
            k = f"{c}/{planner}"
            res = [r[k] for r in rows if k in r]
            sr = np.mean([x["found"] for x in res])
            t = np.array([x["time_s"] for x in res])
            hv = np.sum([x["headings"] for x in res]) / max(1, np.sum([x["visits"] for x in res]))
            d_t, d_s = defaultdict(list), defaultdict(list)
            for r in rows:
                if k in r and f"{BASE}/{planner}" in r:
                    d_t[r["scene_id"]].append(r[k]["time_s"] - r[f"{BASE}/{planner}"]["time_s"])
                    d_s[r["scene_id"]].append(float(r[k]["found"]) - float(r[f"{BASE}/{planner}"]["found"]))
            mt = np.mean([v for vs in d_t.values() for v in vs])
            ms = np.mean([v for vs in d_s.values() for v in vs])
            lo_t, hi_t = ci(d_t, rng) if c != BASE else (0.0, 0.0)
            lo_s, hi_s = ci(d_s, rng) if c != BASE else (0.0, 0.0)
            print(f"{c:8s} {sr:7.3f} {t.mean():9.1f} {np.median(t):7.1f} {np.mean([x['visits'] for x in res]):6.2f} "
                  f"{hv:9.2f} {mt:+9.1f} s ({lo_t:+.1f}, {hi_t:+.1f}) {ms:+.3f} ({lo_s:+.3f}, {hi_s:+.3f})")
            out[k] = {"success": round(sr, 4), "time_mean_s": round(float(t.mean()), 1),
                      "time_median_s": round(float(np.median(t)), 1), "headings_per_visit": round(float(hv), 2),
                      "visits_mean": round(float(np.mean([x["visits"] for x in res])), 2),
                      "dtime_vs_sector6": [round(float(mt), 1), round(float(lo_t), 1), round(float(hi_t), 1)],
                      "dsuccess_vs_sector6": [round(float(ms), 4), round(float(lo_s), 4), round(float(hi_s), 4)]}
    print(json.dumps(out, ensure_ascii=False), file=open("outputs/heading_compare/summary.json", "w"))


if __name__ == "__main__":
    main()
