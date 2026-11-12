"""Poster tables of the final experiment, drawn from experiments/room_baram_final/analysis.json
and sim_vs_real.json (same style as the earlier poster table):

    table_final_real.png      the real robot: untrained Qwen3-4B vs the trained planner, 20 paired placements
    table_sim_vs_real.png     the same metrics in simulation (HM3D Val) and on the real robot

    ~/miniconda3/envs/athome-data/bin/python scripts/build_poster_tables.py [--output DIR]
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/room_baram_final"
FONT = "DejaVu Serif"
W, H, DPI = 2153, 1184, 200


def draw(path, title, columns, rows, rules_after, notes, bold):
    """columns: (label, x [0-1], align), a label may have two lines; rows: lists of cells; bold: {(row, col)}."""
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.axis("off")
    t = lambda x, y, s, size, **k: ax.text(x, y, s, fontsize=size, family=FONT, va="center", **k)
    t(0.014, 0.075, title, 15)
    top, head, dy = 0.145, 0.115, 0.075
    ax.plot([0.014, 0.986], [top, top], color="black", lw=2.6)
    for label, x, ha in columns:
        t(x, top + head / 2 + 0.004, label, 13.5, ha=ha, linespacing=1.25)
    ax.plot([0.014, 0.986], [top + head, top + head], color="black", lw=1.4)
    for i, row in enumerate(rows):
        y = top + head + dy * (i + 0.5) + 0.004
        for j, ((_, x, ha), cell) in enumerate(zip(columns, row)):
            t(x, y, cell, 13.5, ha=ha, fontweight="bold" if (i, j) in bold else "normal")
        if i in rules_after:
            ax.plot([0.014, 0.986], [y + dy / 2, y + dy / 2], color="black", lw=0.8)
    yb = top + head + dy * len(rows) + 0.004
    ax.plot([0.014, 0.986], [yb, yb], color="black", lw=2.6)
    y = yb + 0.055
    for line in notes:
        for piece in textwrap.wrap(line, 112):
            t(0.014, y, piece, 11, color="#333333")
            y += 0.042
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)


def pct(x): return f"{x * 100:.0f}%"
def ci(lo, hi): return f"{lo * 100:.0f}–{hi * 100:.0f}"
def num(x): return f"{x:g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/poster")
    args = ap.parse_args()
    a = json.loads((EXP / "analysis.json").read_text(encoding="utf-8"))
    s = json.loads((EXP / "sim_vs_real.json").read_text(encoding="utf-8"))
    base, tr = a["summary"]["base"], a["summary"]["trained"]
    pv, pt = a["paired"]["steps"], a["paired"]["time_s"]
    rb, rt = s["real_room_baram"]["base"], s["real_room_baram"]["trained"]
    sb, st = s["simulation_hm3d_val"]["base"], s["simulation_hm3d_val"]["trained"]

    cols = [("Planner", 0.014, "left"), ("Runs", 0.33, "center"), ("SR↑\n(95% CI)", 0.45, "center"), ("SR@300 s↑", 0.585, "center"),
            ("SPL↑", 0.69, "center"), ("Visits↓\nmean (median)", 0.81, "center"), ("Time (s)↓\nmedian", 0.935, "center")]
    rows = [["Untrained Qwen3-4B", "20", f"{pct(base['sr'])} ({ci(*base['sr_ci95'])})", f"{base['success_300']}/20", f"{base['spl']:.3f}",
             f"{rb['visits_mean']:.1f} ({num(base['steps_median'])})", f"{base['time_median_s']:.0f}"],
            ["Trained planner (ours)", "20", f"{pct(tr['sr'])} ({ci(*tr['sr_ci95'])})", f"{tr['success_300']}/20", f"{tr['spl']:.3f}",
             f"{rt['visits_mean']:.1f} ({num(tr['steps_median'])})", f"{tr['time_median_s']:.0f}"]]
    notes = ["Real robot, room_baram (4 rooms + hall), 2026-11-11/12: 20 pre-registered placements of objects not in the graph (laptop, book, plant), "
             "each searched once by each planner from the same hallway point; 600 s per command; both planners used the same scene graph and detector.",
             "Success: found, checked on the found photo, within 600 s. SR@300 s: found within 300 s. SPL: shortest path / max(shortest path, distance driven), failures 0. "
             "Visits: Search Locations with completed move and observation. Visits (mean, median), Time (median): over all 20 runs of a planner.",
             f"Paired (untrained − trained): visits median {num(pv['median_diff_base_minus_trained'])}, Wilcoxon p = {pv['wilcoxon_p']:.3f}; "
             f"time median {pt['median_diff_base_minus_trained']:.0f} s, p = {pt['wilcoxon_p']:.3f}.",
             "Each planner has one failure, neither caused by its choice of place (system stop; missed detection in dim light)."]
    draw(args.output / "table_final_real.png", "TABLE. Object search on the real robot (room_baram): untrained vs trained planner.", cols, rows, set(),
         notes, {(1, 3), (1, 4), (1, 5), (1, 6)})

    cols2 = [("Setting", 0.014, "left"), ("Planner", 0.17, "left"), ("Episodes", 0.445, "center"), ("SR↑\n(95% CI)", 0.585, "center"),
             ("SPL↑", 0.725, "center"), ("Visits↓\nmean (median)", 0.835, "center"), ("Time (s)↓\nmedian", 0.945, "center")]
    rows2 = [["Simulation", "Untrained Qwen3-4B", "1495", f"{sb['sr'] * 100:.1f}% (99.0–99.8)", f"{sb['spl']:.3f}", f"{sb['visits_mean']:.2f} ({num(sb['visits_median'])})", "–"],
             ["", "Trained planner (ours)", "1495", f"{st['sr'] * 100:.0f}% (99.7–100)", f"{st['spl']:.3f}", f"{st['visits_mean']:.2f} ({num(st['visits_median'])})", "–"],
             ["Real robot", "Untrained Qwen3-4B", "20", f"{pct(rb['sr'])} ({ci(*base['sr_ci95'])})", f"{rb['spl']:.3f}", f"{rb['visits_mean']:.2f} ({num(rb['visits_median'])})", f"{base['time_median_s']:.0f}"],
             ["", "Trained planner (ours)", "20", f"{pct(rt['sr'])} ({ci(*tr['sr_ci95'])})", f"{rt['spl']:.3f}", f"{rt['visits_mean']:.2f} ({num(rt['visits_median'])})", f"{tr['time_median_s']:.0f}"]]
    notes2 = ["Same two models and the same metric definitions in both settings: Success = found, SPL = shortest path / max(shortest path, distance driven) (failures 0), "
              "Visits = Search Locations with completed move and observation. Paired comparisons are per episode (placement); Time (median) is measured on the robot only.",
              "Simulation: 1495 held-out HM3D Val episodes, random start, symbolic detection within 3 m. Real robot (room_baram): 20 placements, hallway start, real detector, 600 s budget.",
              f"Visits paired (trained better / untrained better / tie): simulation {s['simulation_hm3d_val']['paired_visits']['trained_better']} / {s['simulation_hm3d_val']['paired_visits']['base_better']} / {s['simulation_hm3d_val']['paired_visits']['ties']} (p < 0.001), "
              f"real {s['real_room_baram']['paired_visits']['trained_better']} / {s['real_room_baram']['paired_visits']['base_better']} / {s['real_room_baram']['paired_visits']['ties']} (p = {s['real_room_baram']['paired_visits']['wilcoxon_p']:.3f}).",
              "Both settings: about 0.6x the visits and +0.06 to +0.07 SPL at the same success."]
    draw(args.output / "table_sim_vs_real.png", "TABLE. Object search: simulation and real robot, untrained vs trained planner.", cols2, rows2, {1},
         notes2, {(1, 4), (1, 5), (3, 4), (3, 5), (3, 6)})
    print("wrote", args.output / "table_final_real.png", args.output / "table_sim_vs_real.png")


if __name__ == "__main__":
    main()
