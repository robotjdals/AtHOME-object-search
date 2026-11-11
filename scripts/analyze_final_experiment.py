"""Final experiment analysis (experiments/room_baram_final/protocol_ko.md, as
registered before the runs): trained vs base planner on 20 paired placements.

    ~/miniconda3/envs/athome-data/bin/python scripts/analyze_final_experiment.py \\
        --robot-config configs/robot/room_baram.yaml \\
        --plan experiments/room_baram_final/scenarios.csv \\
        --runs outputs/room_baram/experiments/room_baram_final

Success: found, the found photo's box on the target (operator verdict), within
the 600 s budget. SPL: shortest path from the run's actual start pose to the
placement (bedroom: to its entry goal, as every bedroom search starts there)
over the distance driven, from the motion module's logs (<runs>/motion/real_*.csv,
x/y at ~10 Hz while driving, the files started within the run). Paired: Wilcoxon signed-rank
(steps, distance, time), exact McNemar (success). Typicality: Spearman between
the placement's training share and the paired step difference (base - trained).
"""
import argparse
import csv
import glob
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from athome.config import load_robot_config  # noqa: E402
from athome.execution.experiment import check_starts, load_plan, load_runs  # noqa: E402
from athome.navigation import load_map_server  # noqa: E402
from athome.navigation.path_cost import shortest_costs  # noqa: E402

BUDGET_S = 600.0
SECONDARY_S = 300.0


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def path_length(points):
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(points, points[1:]))


def driven(motion_dir, t0, t1):
    """Distance driven between t0 and t1: motion logs real_<start>.csv (t from the file's start)."""
    total = 0.0
    for f in sorted(glob.glob(str(Path(motion_dir) / "real_*.csv"))):
        start = int(Path(f).stem.split("_")[1])
        if not t0 - 1 <= start <= t1:
            continue
        with open(f, encoding="utf-8") as fh:
            # Driving rows only: in place (rotating, stopping) the pose jitters by a few cm at
            # 10 Hz, 367 m over the experiment; driving rows match the planned paths within 10 %.
            pts = [(float(r["x"]), float(r["y"])) for r in csv.DictReader(fh)
                   if r.get("x") and r["phase"] == "driving" and t0 <= start + float(r["t"]) <= t1]
        total += path_length(pts)
    return total


def first_room_ok(run):
    visited = run.get("visited") or []
    return bool(visited) and visited[0].split(":")[1] == run["location_id"].split(":")[1]


def main():
    from scipy.stats import binomtest, spearmanr, wilcoxon

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-config", required=True)
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path, help="the experiment's record folder")
    ap.add_argument("--output", type=Path, help="default: <runs>/analysis.json")
    args = ap.parse_args()

    config = load_robot_config(args.robot_config)
    plan = {r["scenario"]: r for r in load_plan(args.plan)}
    typ = {int(r["scenario"]): float(r["share"])
           for r in csv.DictReader(open(args.plan.with_name("typicality.csv"), encoding="utf-8"))}
    entry = config.room_entry_goals
    records = load_runs(args.runs / "runs.jsonl")
    valid = [r for r in records if r.get("valid")]

    # Shortest paths from each run's actual start (the search's own grid and goal rule);
    # a room with an entry goal: to that pose, where every search of the room starts.
    rows, starts = [], {}
    for r in valid:
        name = f"run{r['run']}"
        starts[name] = r["start_pose_actual"]
        rows.append({"scenario": r["scenario"], "object": r["object"], "start": name,
                     "location_id": r["location_id"]})
    lstar = {row["start"]: row["l_star_m"] for row in check_starts(config, rows, starts)}
    grid = load_map_server(config.map_yaml, config.map_inflation, config.unknown_as_occupied)
    for r in valid:
        room = r["location_id"].split(":")[1]
        if room in entry:
            sx, sy = r["start_pose_actual"][:2]
            goal = grid.to_cell(*entry[room][:2])
            cost = shortest_costs(grid.free, grid.nearest_free(grid.to_cell(sx, sy), 0.5), [goal], grid.resolution)
            lstar[f"run{r['run']}"] = round(cost[goal], 2)

    out = {}
    for r in valid:
        found = r["target_status"] == "found" and r["duration_s"] <= BUDGET_S
        d = driven(args.runs / "motion", r["t_start"], r["t_end"])
        ls = lstar[f"run{r['run']}"]
        out[(r["scenario"], r["variant"])] = {
            "run": r["run"], "success": found, "success_300": found and r["duration_s"] <= SECONDARY_S,
            "steps": r["steps_completed"], "time_s": r["duration_s"], "distance_m": round(d, 2),
            "l_star_m": ls, "spl": (ls / max(ls, d)) if found and ls else 0.0,
            "first_room": first_room_ok(r), "fallbacks": r.get("fallbacks"), "decisions": r.get("decisions"),
            "status": r.get("status"), "reason": r.get("reason"), "video": r.get("video"),
            "used": r.get("planner_variant_used")}

    scenarios = sorted({s for s, _ in out})
    summary = {}
    for v in ("trained", "base"):
        xs = [out[(s, v)] for s in scenarios]
        k = sum(x["success"] for x in xs)
        summary[v] = {
            "n": len(xs), "success": k, "sr": k / len(xs), "sr_ci95": wilson(k, len(xs)),
            "success_300": sum(x["success_300"] for x in xs),
            "spl": statistics.mean(x["spl"] for x in xs),
            "steps_median": statistics.median(x["steps"] for x in xs),
            "time_median_s": statistics.median(x["time_s"] for x in xs),
            "distance_median_m": statistics.median(x["distance_m"] for x in xs),
            "first_room": sum(x["first_room"] for x in xs),
            "fallbacks": sum(x["fallbacks"] or 0 for x in xs), "decisions": sum(x["decisions"] or 0 for x in xs),
            "variant_used_ok": all(x["used"] == v for x in xs)}
    paired = {}
    for key in ("steps", "time_s", "distance_m"):
        diff = [out[(s, "base")][key] - out[(s, "trained")][key] for s in scenarios]
        paired[key] = {"median_diff_base_minus_trained": statistics.median(diff),
                       "trained_better": sum(d > 0 for d in diff), "base_better": sum(d < 0 for d in diff),
                       "ties": sum(d == 0 for d in diff), "wilcoxon_p": wilcoxon(diff).pvalue}
    t_only = sum(out[(s, "trained")]["success"] and not out[(s, "base")]["success"] for s in scenarios)
    b_only = sum(out[(s, "base")]["success"] and not out[(s, "trained")]["success"] for s in scenarios)
    paired["success"] = {"trained_only": t_only, "base_only": b_only,
                         "mcnemar_exact_p": binomtest(t_only, t_only + b_only).pvalue if t_only + b_only else 1.0}
    share = [typ[s] for s in scenarios]
    step_diff = [out[(s, "base")]["steps"] - out[(s, "trained")]["steps"] for s in scenarios]
    rho = spearmanr(share, step_diff)
    groups = {}
    for s, d in zip(scenarios, step_diff):
        groups.setdefault(plan[s]["typical"], []).append(d)
    result = {
        "summary": summary, "paired": paired,
        "typicality": {"spearman_rho": rho.statistic, "p": rho.pvalue,
                       "median_step_diff": {g: statistics.median(v) for g, v in groups.items()},
                       "points": [{"scenario": s, "share": typ[s], "typical": plan[s]["typical"], "step_diff": d}
                                  for s, d in zip(scenarios, step_diff)]},
        "runs": {f"{s}_{v}": x for (s, v), x in sorted(out.items())},
        "records": {"valid": len(valid), "invalid": sum(r.get("valid") is False for r in records),
                    "aborted": sum(bool(r.get("aborted")) for r in records)},
    }
    path = args.output or args.runs / "analysis.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("summary", "paired", "records")}, ensure_ascii=False, indent=1,
                     default=float))
    print("typicality", {k: v for k, v in result["typicality"].items() if k != "points"})
    figure(result["typicality"]["points"], rho.statistic, rho.pvalue, path.with_name("typicality_scatter.png"))
    print(f"저장: {path}, {path.with_name('typicality_scatter.png')}")


def figure(points, rho, p, path):
    """Typicality scatter (protocol: x the placement's training share, y paired step difference)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#6b6b6b", "#e4e3df"
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhline(0, color=muted, lw=1)
    ax.axvline(0.25, color=muted, lw=1, ls=(0, (4, 3)))
    for group, color, marker, label in (("typical", "#2a78d6", "o", "typical (share >= 0.25)"),
                                        ("atypical", "#eb6834", "^", "atypical")):
        pts = [q for q in points if q["typical"] == group]
        ax.scatter([q["share"] for q in pts], [q["step_diff"] for q in pts], s=46, c=color, marker=marker,
                   edgecolors="#fcfcfb", linewidths=1.5, label=label, zorder=3)
    ax.set_xlabel("Share of the object's training instances in the placement's room type", color=ink, fontsize=9)
    ax.set_ylabel("Search steps, base - trained", color=ink, fontsize=9)
    ax.set_title(f"Placement typicality vs. planner difference (Spearman rho = {rho:.2f}, p = {p:.2f}, n = {len(points)})",
                 color=ink, fontsize=9, loc="left")
    ax.grid(axis="y", color=grid, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
