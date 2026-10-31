"""Evaluation report in the form of the object-search / RL literature, from
existing ``evaluate_student.py episodes`` files (no GPU, no API).

Per policy, on the same episodes (paired by state_id):
- SR, SPL (Anderson et al. 2018), SR within k visits, and AUC-E: the area
  under the efficiency curve SR(budget) of MoMa-LLM (Honerkamp et al. 2024),
  budget in visits up to the episode cap, so a policy that finds every
  target at its first visit scores 1 and one that finds nothing 0.
- The cost D + 3N of the reward (proposal 6-4) as mean and as interquartile
  mean (IQM, 25% trimmed mean; Agarwal et al. 2021, rliable), which is robust
  to the long tail of very costly episodes.
- Intervals: 95% percentile bootstrap over buildings (episodes of one
  building are not independent), 10,000 resamples.

Pairwise, for a pre-specified family of comparisons (base -> SFT, SFT ->
GRPO, and Teacher vs SFT / GRPO on the holdout, the only buildings the
Teacher ran on):
- mean difference with the building bootstrap interval;
- probability of improvement (rliable): share of episodes where the second
  policy costs less, ties counting one half (one deterministic run per
  episode, so each episode is one "task");
- Wilcoxon signed-rank test (Pratt's zero handling: identical episodes are
  many), Holm-Bonferroni adjusted over the family of one split;
- buildings where the second policy is better (two-sided sign test);
- McNemar exact test on success within 5 visits.
Breakdowns: the ObjectNav start rule (1 m <= l* < 30 m), target categories
seen / synonym / unseen (HM3D-OVON), and distance-to-goal bins (ObjectNav
difficulty).

Not covered: training-seed variance (one SFT and one GRPO run; Henderson et
al. 2018 and rliable ask for several training runs).

    python scripts/benchmark_report.py --output outputs/eval_report/benchmark.json \
        --figures outputs/eval_report
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
STEP_COST, MAX_VISITS, BUDGETS = 3.0, 60, (1, 3, 5, 10)
STANDARD = (1.0, 30.0)
DISTANCE_BINS = ((0.0, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, 30.0), (30.0, float("inf")))
METHODS = {"base": "outputs/eval_student/base_episodes_{split}.json",
           "SFT": "outputs/eval_student/sft_v1_episodes_{split}.json",
           "GRPO": "outputs/eval_student/grpoA200_episodes_{split}.json"}
COMPARISONS = (("base", "SFT"), ("SFT", "GRPO"), ("Teacher", "SFT"), ("Teacher", "GRPO"))
N_BOOT = 10_000


def spl(found, l_star, distance):            # as evaluate_student.py
    if not found or l_star is None:
        return 0.0
    longest = max(l_star, distance)
    return l_star / longest if longest > 0 else 1.0


def load_split(split, methods):
    """Episodes of every policy, keyed by state_id: {state_id: {policy: run}}
    plus the episode's l*, building and target category."""
    files = {name: {r["state_id"]: r for r in json.loads((ROOT / path.format(split=split)).read_text())["rows"]}
             for name, path in methods.items()}
    ids = sorted(set.intersection(*(set(f) for f in files.values())))
    first = files[next(iter(files))]
    for name, rows in files.items():      # same environment: the nearest-first baseline must agree
        bad = [i for i in ids if abs(rows[i]["mincost"]["cost"] - first[i]["mincost"]["cost"]) > 1e-6]
        if bad:
            raise SystemExit(f"{split}/{name}: 가장 가까운 곳부터 결과가 {len(bad)}개 에피소드에서 다름 (다른 환경)")
    episodes = {}
    for i in ids:
        r = first[i]
        runs = {"random": r["random"], "nearest": r["mincost"]}
        runs.update({name: rows[i]["student"] for name, rows in files.items()})
        episodes[i] = {"l_star": r["oracle_distance_m"], "building": i.split(":")[0],
                       "target_split": r["target_split"], "runs": runs}
    return episodes


def add_teacher(episodes):
    """GPT-4.1 Teacher episodes (outputs/teacher_v5, label generation) matched
    to the evaluation episodes by building, component, target and start cell.
    The nearest-first run recorded with them must equal the evaluation's."""
    matched = mismatch = 0
    for building in sorted({e["building"] for e in episodes.values()}):
        teacher_file = ROOT / f"outputs/teacher_v5/episodes_{building}/episodes.json"
        states_file = ROOT / f"outputs/eval_v5_everygoal/{building}/start_states.jsonl"
        if not teacher_file.exists() or not states_file.exists():
            continue
        states = [json.loads(l) for l in states_file.read_text().splitlines() if l.strip()]
        by_start = {(s["component"], s["target"], tuple(s["start_row_col"])): s["state_id"] for s in states}
        nearest = {}
        nearest_file = ROOT / f"outputs/baseline_compare_v5/{building}.jsonl"
        if nearest_file.exists():
            for l in nearest_file.read_text().splitlines():
                b = json.loads(l)
                nearest[(b["component"], b["target"], b["episode_id"])] = b["mincost"]
        for t in json.loads(teacher_file.read_text())["episodes"]:
            sid = by_start.get((t["component"], t["target"], tuple(t["start_row_col"])))
            if sid not in episodes:
                continue
            e = episodes[sid]
            ref = nearest.get((t["component"], t["target"], t["episode_id"]))
            if ref is None or abs(ref["distance_m"] - e["runs"]["nearest"]["distance_m"]) > 1e-6 \
                    or ref["steps"] != e["runs"]["nearest"]["visits"]:
                mismatch += 1
                continue
            found = t["status"] == "found"
            e["runs"]["Teacher"] = {"found": found, "distance_m": t["distance_m"], "visits": t["steps"],
                                    "spl": spl(found, e["l_star"], t["distance_m"]),
                                    "cost": t["distance_m"] + STEP_COST * t["steps"],
                                    "policy_fallbacks": t["policy_fallbacks"]}
            matched += 1
    return matched, mismatch


def per_episode(run):
    """Per-episode values whose means are the reported metrics."""
    found, visits = bool(run["found"]), run["visits"]
    values = {"SR": float(found), "SPL": run["spl"], "cost": run["cost"], "visits": float(visits),
              "AUC-E": float(found) * max(0, MAX_VISITS - visits + 1) / MAX_VISITS}
    values.update({f"SR@{k}": float(found and visits <= k) for k in BUDGETS})
    return values


class Bootstrap:
    """Building (cluster) bootstrap with one set of resamples for every
    statistic, so intervals of different metrics use the same draws."""

    def __init__(self, buildings, n=N_BOOT, seed=0):
        keys = sorted(set(buildings))
        self.index = np.array([keys.index(b) for b in buildings])
        self.members = [np.flatnonzero(self.index == k) for k in range(len(keys))]
        self.picks = np.random.default_rng(seed).integers(0, len(keys), size=(n, len(keys)))
        self.counts = np.bincount(self.index, minlength=len(keys)).astype(float)

    def mean(self, values):
        sums = np.bincount(self.index, weights=values, minlength=len(self.members))
        boot = sums[self.picks].sum(axis=1) / self.counts[self.picks].sum(axis=1)
        return float(np.mean(values)), tuple(float(x) for x in np.percentile(boot, [2.5, 97.5]))

    def iqm(self, values):
        boot = [stats.trim_mean(values[np.concatenate([self.members[k] for k in pick])], 0.25) for pick in self.picks]
        return float(stats.trim_mean(values, 0.25)), tuple(float(x) for x in np.percentile(boot, [2.5, 97.5]))


def summarize(episodes, ids, policies):
    boot = Bootstrap([episodes[i]["building"] for i in ids])
    table = {}
    for p in policies:
        values = [per_episode(episodes[i]["runs"][p]) for i in ids]
        row = {}
        for metric in values[0]:
            mean, ci = boot.mean(np.array([v[metric] for v in values]))
            row[metric] = {"mean": mean, "ci95_buildings": ci}
        mean, ci = boot.iqm(np.array([v["cost"] for v in values]))
        row["cost_IQM"] = {"mean": mean, "ci95_buildings": ci}
        table[p] = row
    return {"episodes": len(ids), "buildings": len(boot.members), "policies": table}


def compare(episodes, ids, a, b):
    """Second policy b against first policy a on the same episodes."""
    boot = Bootstrap([episodes[i]["building"] for i in ids])
    va = [per_episode(episodes[i]["runs"][a]) for i in ids]
    vb = [per_episode(episodes[i]["runs"][b]) for i in ids]
    out = {"pair": f"{b} - {a}", "episodes": len(ids)}
    for metric in ("cost", "SPL", "SR@5", "AUC-E"):
        diff = np.array([y[metric] - x[metric] for x, y in zip(va, vb)])
        mean, ci = boot.mean(diff)
        entry = {"mean_diff": mean, "ci95_buildings": ci}
        if np.any(diff != 0):
            entry["wilcoxon_p"] = float(stats.wilcoxon(diff, zero_method="pratt").pvalue)
        out[metric] = entry
    cost_a = np.array([x["cost"] for x in va])
    cost_b = np.array([y["cost"] for y in vb])
    better = (cost_b < cost_a - 1e-9).astype(float) + 0.5 * (np.abs(cost_b - cost_a) <= 1e-9)
    mean, ci = boot.mean(better)
    out["prob_improvement"] = {"value": mean, "ci95_buildings": ci,
                               "better": int(np.sum(cost_b < cost_a - 1e-9)),
                               "tie": int(np.sum(np.abs(cost_b - cost_a) <= 1e-9)),
                               "worse": int(np.sum(cost_b > cost_a + 1e-9))}
    buildings = sorted({episodes[i]["building"] for i in ids})
    wins = losses = 0
    for building in buildings:
        sel = [k for k, i in enumerate(ids) if episodes[i]["building"] == building]
        d = cost_b[sel].mean() - cost_a[sel].mean()
        wins, losses = wins + (d < -1e-9), losses + (d > 1e-9)
    out["buildings_better"] = {"better": int(wins), "worse": int(losses), "buildings": len(buildings),
                               "sign_test_p": float(stats.binomtest(int(wins), int(wins + losses)).pvalue)
                               if wins + losses else None}
    sa = np.array([x["SR@5"] for x in va], dtype=bool)
    sb = np.array([y["SR@5"] for y in vb], dtype=bool)
    only_b, only_a = int(np.sum(sb & ~sa)), int(np.sum(sa & ~sb))
    out["mcnemar_SR@5"] = {"only_second": only_b, "only_first": only_a,
                           "p": float(stats.binomtest(only_b, only_b + only_a).pvalue) if only_a + only_b else None}
    return out


def holm(comparisons):
    """Holm-Bonferroni over every Wilcoxon test of one split."""
    tests = [(c, m) for c in comparisons for m in ("cost", "SPL", "SR@5", "AUC-E") if "wilcoxon_p" in c[m]]
    order = sorted(range(len(tests)), key=lambda k: tests[k][0][tests[k][1]]["wilcoxon_p"])
    running = 0.0
    for rank, k in enumerate(order):
        c, m = tests[k]
        running = max(running, min(1.0, (len(tests) - rank) * c[m]["wilcoxon_p"]))
        c[m]["holm_p"] = running


def breakdown(episodes, ids, policies):
    out = {}
    for split in ("seen", "synonym", "unseen"):
        sel = [i for i in ids if episodes[i]["target_split"] == split]
        if sel:
            out[split] = {"episodes": len(sel), **{p: {m: float(np.mean([per_episode(episodes[i]["runs"][p])[m] for i in sel]))
                                                       for m in ("cost", "SPL", "SR@5")} for p in policies}}
    for lo, hi in DISTANCE_BINS:
        sel = [i for i in ids if lo <= episodes[i]["l_star"] < hi]
        if sel:
            out[f"l*[{lo:g},{hi:g})"] = {"episodes": len(sel), **{p: {m: float(np.mean([per_episode(episodes[i]["runs"][p])[m] for i in sel]))
                                                                       for m in ("cost", "SPL", "SR@5")} for p in policies}}
    return out


def fmt_ci(entry, digits=2):
    lo, hi = entry["ci95_buildings"]
    return f"{entry['mean']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def print_report(split, report):
    for subset, block in report["subsets"].items():
        s = block["summary"]
        print(f"\n## {split} / {subset}: 에피소드 {s['episodes']}개, 건물 {s['buildings']}개 (평균 [건물 bootstrap 95% CI])")
        print("| 정책 | SR | SPL | SR@1 | SR@5 | AUC-E | 비용 | 비용 IQM |")
        print("|---|---|---|---|---|---|---|---|")
        for p, row in s["policies"].items():
            print(f"| {p} | {row['SR']['mean']:.3f} | {fmt_ci(row['SPL'], 3)} | {row['SR@1']['mean']:.3f} | "
                  f"{fmt_ci(row['SR@5'], 3)} | {fmt_ci(row['AUC-E'], 3)} | {fmt_ci(row['cost'])} | {fmt_ci(row['cost_IQM'])} |")
        print("\n| 비교 | 비용 차 [CI] (Holm p) | SPL 차 [CI] (Holm p) | SR@5 차 (McNemar p) | 개선 확률 [CI] | 나은 건물 (부호검정 p) |")
        print("|---|---|---|---|---|---|")
        for c in block["comparisons"]:
            def d(m, digits):
                e = c[m]
                lo, hi = e["ci95_buildings"]
                p = f"{e['holm_p']:.2g}" if "holm_p" in e else "-"
                return f"{e['mean_diff']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}] ({p})"
            pi, bb, mc = c["prob_improvement"], c["buildings_better"], c["mcnemar_SR@5"]
            mp = f"{mc['p']:.2g}" if mc["p"] is not None else "-"
            sp = f"{bb['sign_test_p']:.2g}" if bb["sign_test_p"] is not None else "-"
            print(f"| {c['pair']} (n={c['episodes']}) | {d('cost', 2)} | {d('SPL', 3)} | {c['SR@5']['mean_diff']:+.3f} ({mp}) | "
                  f"{pi['value']:.3f} [{pi['ci95_buildings'][0]:.3f}, {pi['ci95_buildings'][1]:.3f}] "
                  f"(나음 {pi['better']} / 같음 {pi['tie']} / 나쁨 {pi['worse']}) | {bb['better']}/{bb['buildings']} ({sp}) |")


def figures(out_dir, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_midterm_figures import BASE, GRID, GRPO, INK, NEAREST, RANDOM, SFT, note, setup
    setup()
    style = {"random": ("무작위", RANDOM, ":"), "nearest": ("가장 가까운 곳부터", NEAREST, "--"),
             "base": ("기본 Qwen3-4B", BASE, "-"), "SFT": ("SFT", SFT, "-"), "GRPO": ("SFT + GRPO (v1)", GRPO, "-"),
             "Teacher": ("Teacher (GPT-4.1)", INK, "--")}
    titles = {"holdout": "홀드아웃 (Teacher와 같은 에피소드)", "val": "Val (처음 보는 24개 건물)"}
    out_dir.mkdir(parents=True, exist_ok=True)
    # Efficiency curve (MoMa-LLM): success within a visit budget, up to the cap.
    fig, axes = plt.subplots(1, len(reports), figsize=(6.5 * len(reports), 5.2), sharey=True)
    for ax, (split, (episodes, ids, policies)) in zip(np.atleast_1d(axes), reports.items()):
        budgets = np.arange(1, MAX_VISITS + 1)
        for p in policies:
            visits = np.array([episodes[i]["runs"][p]["visits"] if episodes[i]["runs"][p]["found"] else np.inf for i in ids])
            curve = [(visits <= b).mean() * 100 for b in budgets]
            name, color, ls = style[p]
            auc = np.mean(curve) / 100
            ax.step(budgets, curve, where="post", color=color, linestyle=ls, linewidth=2.2, label=f"{name}  AUC-E {auc:.3f}")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 10, 20, 60], ["1", "2", "3", "5", "10", "20", "60"])
        ax.set_xlabel("방문 예산 (회, 로그 축)")
        ax.set_ylim(0, 101)
        ax.grid(color=GRID, linewidth=1)
        ax.set_title(f"{titles.get(split, split)}, {len(ids)}개", loc="left", fontsize=13)
        ax.legend(frameon=False, loc="lower right", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("예산 안에 찾은 비율 (%)")
    fig.suptitle("효율 곡선 (MoMa-LLM의 AUC-E): 같은 방문 수로 더 많이 찾을수록 좋음", x=0.01, ha="left", fontsize=15, fontweight="bold")
    note(fig, "AUC-E = 방문 1~60회 예산별 성공률의 평균. 첫 방문에 모두 찾으면 1.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out_dir / "efficiency_curve.png", dpi=200)
    plt.close(fig)
    # Performance profile (rliable): share of episodes with SPL above tau.
    fig, axes = plt.subplots(1, len(reports), figsize=(6.5 * len(reports), 5.2), sharey=True)
    taus = np.linspace(0, 1, 101)
    for ax, (split, (episodes, ids, policies)) in zip(np.atleast_1d(axes), reports.items()):
        sel = [i for i in ids if STANDARD[0] <= episodes[i]["l_star"] < STANDARD[1]]
        for p in policies:
            values = np.array([episodes[i]["runs"][p]["spl"] for i in sel])
            name, color, ls = style[p]
            ax.plot(taus, [(values > t).mean() * 100 for t in taus], color=color, linestyle=ls, linewidth=2.2, label=name)
        ax.set_xlabel("SPL 기준 τ")
        ax.grid(color=GRID, linewidth=1)
        ax.set_title(f"{titles.get(split, split)}, {len(sel)}개", loc="left", fontsize=13)
        ax.legend(frameon=False, loc="upper right", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("SPL > τ 인 에피소드 비율 (%)")
    fig.suptitle("성능 프로파일 (rliable): 곡선이 위에 있을수록 모든 수준에서 더 효율적", x=0.01, ha="left", fontsize=15, fontweight="bold")
    note(fig, "ObjectNav 시작 규칙(목표까지 1~30 m) 에피소드. 곡선 아래 넓이 = 평균 SPL.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out_dir / "performance_profile.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["holdout", "val"])
    parser.add_argument("--method", action="append", default=[], metavar="NAME=PATH",
                        help="정책 결과 파일 ({split} 자리 표시 가능). 기본: base, SFT, GRPO")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    methods = dict(METHODS)
    methods.update(dict(m.split("=", 1) for m in args.method))
    result, plots = {}, {}
    for split in args.splits:
        episodes = load_split(split, methods)
        report = {"episodes": len(episodes)}
        if split == "holdout":
            matched, mismatch = add_teacher(episodes)
            report["teacher"] = {"matched": matched, "nearest_first_mismatch": mismatch}
        policies = ["random", "nearest", *methods]
        teacher_ids = sorted(i for i, e in episodes.items() if "Teacher" in e["runs"])
        subsets = {"all": sorted(episodes),
                   "objectnav_rule": sorted(i for i, e in episodes.items() if STANDARD[0] <= e["l_star"] < STANDARD[1])}
        if teacher_ids:
            subsets["teacher_episodes"] = teacher_ids
        report["subsets"] = {}
        for subset, ids in subsets.items():
            pols = policies + (["Teacher"] if subset == "teacher_episodes" else [])
            comps = [compare(episodes, ids, a, b) for a, b in COMPARISONS
                     if all(p in pols for p in (a, b)) and ("Teacher" in (a, b)) == (subset == "teacher_episodes")]
            holm(comps)
            report["subsets"][subset] = {"summary": summarize(episodes, ids, pols), "comparisons": comps,
                                         "breakdown": breakdown(episodes, ids, pols)}
        result[split] = report
        print_report(split, report)
        if "teacher" in report:
            print(f"\nTeacher 짝 맞춤: {report['teacher']['matched']}개 (가장 가까운 곳부터 불일치로 제외 "
                  f"{report['teacher']['nearest_first_mismatch']}개)")
        plot_ids = teacher_ids if teacher_ids else sorted(episodes)
        plots[split] = (episodes, plot_ids, policies + (["Teacher"] if teacher_ids else []))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print("\nwrote", args.output)
    if args.figures:
        figures(args.figures, plots)
        print("wrote", args.figures / "efficiency_curve.png", args.figures / "performance_profile.png")


if __name__ == "__main__":
    main()
