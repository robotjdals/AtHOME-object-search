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
from PIL import Image, ImageChops

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
    trim(path)


def trim(path, margin=50):
    """Cut the empty bottom of the canvas (the notes decide the height)."""
    im = Image.open(path).convert("RGB")
    rows = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    if rows:
        im.crop((0, 0, im.width, min(im.height, rows[3] + margin))).save(path)


def pct(x): return f"{x * 100:.0f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/poster")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    a = json.loads((EXP / "analysis.json").read_text(encoding="utf-8"))
    s = json.loads((EXP / "sim_vs_real.json").read_text(encoding="utf-8"))
    base, tr = a["summary"]["base"], a["summary"]["trained"]
    pv, pt = a["paired"]["steps"], a["paired"]["time_s"]
    rb, rt = s["real_room_baram"]["base"], s["real_room_baram"]["trained"]
    sb, st = s["simulation_hm3d_val"]["base"], s["simulation_hm3d_val"]["trained"]

    sv, rv = s["simulation_hm3d_val"], s["real_room_baram"]
    NOTE = "SR: found. SPL: shortest path / path driven. Visits: places visited before finding. Time: median. Bold: significantly better, paired test, p < 0.05."
    sig = lambda p: p < 0.05
    real = [sig(rv["paired_success"]["mcnemar_p"]) and rt["sr"] > rb["sr"], sig(rv["paired_spl"]["wilcoxon_p"]), sig(rv["paired_visits"]["wilcoxon_p"]), sig(pt["wilcoxon_p"])]
    simu = [sig(sv["paired_success"]["mcnemar_p"]), sig(sv["paired_spl"]["wilcoxon_p"]), sig(sv["paired_visits"]["wilcoxon_p"])]

    cols = [("Planner", 0.014, "left"), ("SR↑", 0.45, "center"), ("SPL↑", 0.60, "center"), ("Visits↓", 0.76, "center"), ("Time (s)↓", 0.91, "center")]
    rows = [["Untrained Qwen3-4B", pct(base["sr"]), f"{base['spl']:.3f}", f"{rb['visits_mean']:.1f}", f"{base['time_median_s']:.0f}"],
            ["Trained planner", pct(tr["sr"]), f"{tr['spl']:.3f}", f"{rt['visits_mean']:.1f}", f"{tr['time_median_s']:.0f}"]]
    draw(args.output / "table_final_real.png", "TABLE. Object search on the real robot, 20 placements.", cols, rows, set(), [NOTE],
         {(1, j + 1) for j, on in enumerate(real) if on})

    cols2 = [("Setting", 0.014, "left"), ("Planner", 0.30, "left"), ("SR↑", 0.55, "center"), ("SPL↑", 0.67, "center"),
             ("Visits↓", 0.79, "center"), ("Time (s)↓", 0.915, "center")]
    rows2 = [["Simulation", "Untrained Qwen3-4B", f"{sb['sr'] * 100:.1f}%", f"{sb['spl']:.3f}", f"{sb['visits_mean']:.2f}", "–"],
             ["", "Trained planner", f"{st['sr'] * 100:.0f}%", f"{st['spl']:.3f}", f"{st['visits_mean']:.2f}", "–"],
             ["Real robot", "Untrained Qwen3-4B", pct(rb["sr"]), f"{rb['spl']:.3f}", f"{rb['visits_mean']:.2f}", f"{base['time_median_s']:.0f}"],
             ["", "Trained planner", pct(rt["sr"]), f"{rt['spl']:.3f}", f"{rt['visits_mean']:.2f}", f"{tr['time_median_s']:.0f}"]]
    bold2 = {(1, j + 2) for j, on in enumerate(simu) if on} | {(3, j + 2) for j, on in enumerate(real) if on}
    draw(args.output / "table_sim_vs_real.png", "TABLE. Object search, 1495 simulated episodes and 20 real-robot placements.", cols2, rows2, {1},
         [NOTE + " Time is measured on the robot only."], bold2)
    print("wrote", args.output / "table_final_real.png", args.output / "table_sim_vs_real.png")


if __name__ == "__main__":
    main()
