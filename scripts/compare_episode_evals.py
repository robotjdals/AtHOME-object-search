"""Paired comparison of two ``evaluate_student.py episodes`` result files.

Both files must come from the same start states (episodes paired by
state_id), and the nearest-first baseline must agree on every episode, which
checks that both ran in the same environment. Metrics of each policy: success
rate (SR) and SPL (Anderson et al. 2018, "On Evaluation of Embodied Navigation
Agents"), path length, visits, the cost D + lambda N of the reward (proposal
6-4), and SR within a budget of k visits (SR@k; ObjectNav counts success
within an action budget, while here every start is findable and the cap of
60 visits is rarely reached, so SR at the cap is saturated by design).
Differences B - A are paired, with 95% percentile bootstrap intervals over
episodes and over buildings (cluster bootstrap: episodes of one building are
not independent).

Subsets: all episodes; the ObjectNav start rule (shortest path to a view of
the target at least 1 m and below 30 m, as the Habitat ObjectNav episodes of
HSSD-200, Khanna et al. 2024; below 1 m SPL is 0 whenever the agent moves at
all); seen / synonym / unseen target categories when more than one is present.

    python scripts/compare_episode_evals.py outputs/eval_student/base_episodes_holdout.json \
        outputs/eval_student/sft_v1_episodes_holdout.json --names base SFT
"""
import argparse
import json
from pathlib import Path

import numpy as np

BUDGETS = (1, 3, 5, 10)
METRICS = [("SR", lambda s: float(s["found"])), ("SPL", lambda s: s["spl"]),
           ("distance_m", lambda s: s["distance_m"]), ("visits", lambda s: float(s["visits"])),
           ("cost", lambda s: s["cost"])]
METRICS += [(f"SR@{k}", lambda s, k=k: float(s["found"] and s["visits"] <= k)) for k in BUDGETS]


def bootstrap(diffs, groups, n=10000, seed=0):
    """95% percentile intervals of the mean paired difference, resampling
    episodes and resampling buildings (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    episodes = diffs[rng.integers(0, len(diffs), size=(n, len(diffs)))].mean(axis=1)
    keys = sorted(set(groups))
    index = np.array([keys.index(g) for g in groups])
    sums = np.bincount(index, weights=diffs, minlength=len(keys))
    counts = np.bincount(index, minlength=len(keys)).astype(float)
    picked = rng.integers(0, len(keys), size=(n, len(keys)))
    clusters = sums[picked].sum(axis=1) / counts[picked].sum(axis=1)
    return (tuple(np.percentile(episodes, [2.5, 97.5])), tuple(np.percentile(clusters, [2.5, 97.5])))


def compare(a, b, ids, names):
    groups = [i.split(":")[0] for i in ids]
    metrics = []
    for metric, f in METRICS:
        va = [f(a[i]["student"]) for i in ids]
        vb = [f(b[i]["student"]) for i in ids]
        diffs = [y - x for x, y in zip(va, vb)]
        ci_episodes, ci_buildings = bootstrap(diffs, groups)
        metrics.append({"metric": metric, names[0]: float(np.mean(va)), names[1]: float(np.mean(vb)),
                        "nearest": float(np.mean([f(a[i]["mincost"]) for i in ids])),
                        "random": float(np.mean([f(a[i]["random"]) for i in ids])),
                        "diff": float(np.mean(diffs)), "ci_episodes": ci_episodes, "ci_buildings": ci_buildings})
    cheaper = sum(b[i]["student"]["cost"] < a[i]["student"]["cost"] - 1e-9 for i in ids)
    tie = sum(abs(b[i]["student"]["cost"] - a[i]["student"]["cost"]) <= 1e-9 for i in ids)
    return {"episodes": len(ids), "buildings": len(set(groups)), "metrics": metrics,
            "cost_per_episode": {f"{names[1]} cheaper": cheaper, "tie": tie,
                                 f"{names[0]} cheaper": len(ids) - cheaper - tie}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, help="기준 결과 (evaluate_student.py episodes)")
    parser.add_argument("b", type=Path, help="비교할 결과 (같은 시작 상태)")
    parser.add_argument("--names", nargs=2, default=["A", "B"], help="두 정책의 이름")
    parser.add_argument("--output", type=Path, help="요약 JSON (새 파일)")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    if args.names[0] == args.names[1] or set(args.names) & {"nearest", "random"}:
        raise SystemExit("--names는 서로 다르고 nearest/random이 아니어야 합니다.")
    a = {r["state_id"]: r for r in json.loads(args.a.read_text(encoding="utf-8"))["rows"]}
    b = {r["state_id"]: r for r in json.loads(args.b.read_text(encoding="utf-8"))["rows"]}
    if a.keys() != b.keys():
        raise SystemExit(f"시작 상태가 다릅니다: {len(a)} / {len(b)} (공통 {len(a.keys() & b.keys())})")
    ids = sorted(a)
    mismatch = [i for i in ids if abs(a[i]["mincost"]["cost"] - b[i]["mincost"]["cost"]) > 1e-6]
    if mismatch:
        raise SystemExit(f"같은 시작 상태에서 nearest-first 결과가 다릅니다 (환경 불일치): {mismatch[:3]}")
    subsets = {"all": ids,
               "ObjectNav start rule (1 m <= l* < 30 m)":
                   [i for i in ids if 1.0 <= a[i]["oracle_distance_m"] < 30.0]}
    splits = {a[i]["target_split"] for i in ids}
    for split in ("seen", "synonym", "unseen"):
        if len(splits) > 1 and split in splits:
            subsets[f"target {split}"] = [i for i in ids if a[i]["target_split"] == split]
    report = {"a": str(args.a), "b": str(args.b), "names": args.names,
              "subsets": {name: compare(a, b, sel, args.names) for name, sel in subsets.items() if sel}}
    for name, r in report["subsets"].items():
        print(f"\n== {name}: {r['episodes']} episodes, {r['buildings']} buildings ==")
        print(f"{'':11s} {args.names[0]:>9s} {args.names[1]:>9s} {'nearest':>9s} {'random':>9s} | "
              f"{'diff':>8s}   95% CI episodes | 95% CI buildings")
        for m in r["metrics"]:
            (e1, e2), (c1, c2) = m["ci_episodes"], m["ci_buildings"]
            print(f"{m['metric']:11s} {m[args.names[0]]:9.3f} {m[args.names[1]]:9.3f} {m['nearest']:9.3f} "
                  f"{m['random']:9.3f} | {m['diff']:+8.3f}  ({e1:+.3f}, {e2:+.3f}) | ({c1:+.3f}, {c2:+.3f})")
        print("cost per episode:", r["cost_per_episode"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
