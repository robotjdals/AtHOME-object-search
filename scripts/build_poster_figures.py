"""More poster figures from the final-experiment data (same style as build_poster_tables.py):

    poster_system.png        the three modules and how a search runs through them
    poster_paired.png        visits per placement, trained vs untrained planner (the 20 pairs)
    table_structure.png      the same untrained Qwen3-4B given the home as scene graph / without workspaces / flat list
    poster_qr.png, poster_qr_ko.png   QR code of the project page (English / Korean)

    PYTHONPATH=<segno dir> ~/miniconda3/envs/athome-data/bin/python scripts/build_poster_figures.py [--output DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from build_poster_tables import draw

ROOT = Path(__file__).resolve().parents[1]
PAGE = "https://robotjdals.github.io/AtHOME-object-search/"
INK, MUTED, LINE = "#16161a", "#5c5c63", "#c9c7c2"
PERCEPTION, SEARCH, MOTION = "#1a9e5c", "#2a78d6", "#eb6834"
BASE = "#7b7b83"


def system(path):
    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=200)
    ax.set_xlim(0, 150); ax.set_ylim(62, 0); ax.axis("off")

    def box(x, y, w, h, color, tag, title, text):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc="#f3f2ef", ec=LINE, lw=1.2))
        ax.add_patch(FancyBboxPatch((x, y), w, 1.8, boxstyle="round,pad=0,rounding_size=0.9", fc=color, ec=color, lw=0))
        ax.add_patch(plt.Circle((x + 2.2, y + 5.1), 0.8, color=color))
        ax.text(x + 3.9, y + 5.1, tag, fontsize=10, color=MUTED, va="center", fontweight="bold")
        ax.text(x + 2, y + 9.6, title, fontsize=15, color=INK, va="center", fontweight="bold")
        ax.text(x + 2, y + 13.6, text, fontsize=11, color=INK, va="top", linespacing=1.35)

    def arrow(x0, y0, x1, y1, rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, lw=2, color=MUTED, connectionstyle=f"arc3,rad={rad}"))

    ax.text(1, 3, "Once per home: build the map", fontsize=14, color=MUTED, va="center", fontweight="bold")
    box(1, 7, 44, 24, PERCEPTION, "PERCEPTION", "Mapping drive", "The robot is driven through the\nhome once: camera, LiDAR, pose.")
    box(53, 7, 44, 24, PERCEPTION, "PERCEPTION", "3D object map", "Detection on keyframes, fused with\nLiDAR into objects with 3D boxes.")
    box(105, 7, 44, 24, SEARCH, "SCENE GRAPH", "Rooms → Workspaces → Objects", "Objects grouped under the furniture\nand room they are in; a viewpoint\nfor every place.")
    arrow(45.5, 19, 52.5, 19); arrow(97.5, 19, 104.5, 19)

    ax.text(1, 36, "For every request: search", fontsize=14, color=MUTED, va="center", fontweight="bold")
    xs = [1, 31.5, 62, 92.5, 123]
    steps = [(SEARCH, "SEARCH", "Choose a place", "Known object: its furniture.\nOtherwise the small LLM\npicks a room, then a place."),
             (MOTION, "MOTION", "Drive and stop", "MPPI control to a viewpoint,\nthe stop is measured."),
             (MOTION, "MOTION", "Look around", "In-place turns, camera\nstill for each photo."),
             (PERCEPTION, "PERCEPTION", "Detect", "Each photo becomes named\nobjects with 3D positions."),
             (SEARCH, "SEARCH", "Found?", "Yes: report with a photo.\nNo: choose again.")]
    for x, (color, tag, title, text) in zip(xs, steps):
        box(x, 40, 26, 20.5, color, tag, title, text)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + 26.4, 50, b - 0.4, 50)
    ax.add_patch(FancyArrowPatch((136, 61.2), (14, 61.2), arrowstyle="-|>", mutation_scale=16, lw=1.6, ls=(0, (4, 3)), color=MUTED, connectionstyle="arc3,rad=0.0"))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def paired(path, scenarios):
    mx = 12
    groups = [("Trained planner visited fewer places", lambda s: s["base"]["visits"] - s["trained"]["visits"] > 0),
              ("Same number", lambda s: s["base"]["visits"] == s["trained"]["visits"]),
              ("Trained planner visited more", lambda s: s["base"]["visits"] - s["trained"]["visits"] < 0)]
    rows = []
    for title, test in groups:
        members = sorted((s for s in scenarios if test(s)), key=lambda s: (-(s["base"]["visits"] - s["trained"]["visits"]), s["scenario"]))
        rows.append(("group", f"{title} ({len(members)})"))
        rows += [("row", s) for s in members]
    h = 0.34 * len(rows) + 1.6
    fig, ax = plt.subplots(figsize=(10.5, h), dpi=200)
    ax.set_xlim(-6.2, mx + 2.2); ax.set_ylim(len(rows) + 1.2, -1.4); ax.axis("off")
    for v in range(0, mx + 1, 2):
        ax.plot([v, v], [-0.2, len(rows) + 0.1], color="#e4e2dd", lw=0.9, zorder=0)
        ax.text(v, -0.75, str(v), ha="center", va="center", fontsize=10, color=MUTED)
    ax.text(-6.1, -1.25, "Places visited before the object was found", fontsize=11, color=MUTED, va="center", fontweight="bold")
    for i, (kind, item) in enumerate(rows):
        y = i + 0.6
        if kind == "group":
            ax.text(-6.1, y, item, fontsize=11, fontweight="bold", color=MUTED, va="center")
            continue
        s = item
        t, b = s["trained"], s["base"]
        ax.text(-6.1, y, f"{s['object']}", fontsize=11, fontweight="bold", color=INK, va="center")
        ax.text(-4.4, y, s["room"].lower(), fontsize=11, color=MUTED, va="center")
        ax.plot([min(t["visits"], b["visits"]), max(t["visits"], b["visits"])], [y, y], color=BASE, lw=1.6, alpha=0.55, zorder=1)
        for run, color, filled in ((b, BASE, False), (t, SEARCH, True)):
            if not run["success"]:
                ax.text(run["visits"], y, "×", fontsize=17, fontweight="bold", color=color, ha="center", va="center", zorder=3)
            elif filled:
                ax.plot(run["visits"], y, "o", ms=6.5, color=color, mec="white", mew=1.4, zorder=4)
            else:
                ax.plot(run["visits"], y, "o", ms=11, mfc="white", mec=color, mew=1.8, zorder=2)
        ax.text(mx + 2.1, y, f"{t['visits']} / {b['visits']}", fontsize=11, color=INK, va="center", ha="right")
    ax.plot([], [], "o", ms=6.5, color=SEARCH, ls="", label="trained planner")
    ax.plot([], [], "o", ms=11, mfc="white", mec=BASE, mew=1.8, ls="", label="untrained Qwen3-4B")
    ax.plot([], [], ls="", marker="$\\times$", ms=11, color=BASE, label="not found")
    ax.legend(loc="lower right", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(1.0, -0.02), handletextpad=0.3, columnspacing=1.2)
    fig.subplots_adjust(0.01, 0.02, 0.99, 0.99)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def structure(path, results):
    names = [("full/qwen", "Hierarchical scene graph"), ("wo_workspace/qwen", "Without the workspace layer"), ("object_list/qwen", "Flat object list")]
    rows = []
    for key, label in names:
        r = results[key]
        tokens = f"{'≥ ' if r['prompt_tokens_per_episode_lower_bound'] else ''}{r['prompt_tokens_per_episode']:,.0f}"
        rows.append([label, f"{r['success']:.3f}", f"{r['spl']:.3f}", f"{r['visits']:.1f}", tokens])
    cols = [("Home given as", 0.014, "left"), ("SR↑", 0.50, "center"), ("SPL↑", 0.63, "center"), ("Visits↓", 0.76, "center"), ("Prompt tokens↓", 0.89, "center")]
    flat = results["object_list/qwen"]
    notes = [f"Same untrained Qwen3-4B planner. Flat list: {flat['invalid_rate'] * 100:.1f}% invalid answers, token count is a lower bound. Bold: best."]
    draw(path, "TABLE. Scene graph vs flat object list, 1226 simulated episodes.", cols, rows, set(), notes, {(0, 1), (0, 2), (0, 3), (0, 4)})


def system_compact(path):
    fig, ax = plt.subplots(figsize=(15, 2.75), dpi=200)
    ax.set_xlim(0, 150); ax.set_ylim(27.5, 0); ax.axis("off")
    xs = [1, 31.5, 62, 92.5, 123]
    steps = [(SEARCH, "SEARCH", "Choose a place", "the small LLM picks\na room, then a place"),
             (MOTION, "MOTION", "Drive and stop", "MPPI control, the\nstop is measured"),
             (MOTION, "MOTION", "Look around", "in-place turns,\ncamera still"),
             (PERCEPTION, "PERCEPTION", "Detect", "photo to named objects\nwith 3D positions"),
             (SEARCH, "SEARCH", "Found?", "yes: report with photo\nno: choose again")]
    for x, (color, tag, title, text) in zip(xs, steps):
        ax.add_patch(FancyBboxPatch((x, 2), 26, 22, boxstyle="round,pad=0,rounding_size=1.6", fc="#f3f2ef", ec=LINE, lw=1.2))
        ax.add_patch(FancyBboxPatch((x, 2), 26, 1.8, boxstyle="round,pad=0,rounding_size=0.9", fc=color, ec=color, lw=0))
        ax.add_patch(plt.Circle((x + 2.2, 7.2), 0.8, color=color))
        ax.text(x + 3.9, 7.2, tag, fontsize=10, color=MUTED, va="center", fontweight="bold")
        ax.text(x + 2, 12, title, fontsize=15, color=INK, va="center", fontweight="bold")
        ax.text(x + 2, 16.2, text, fontsize=11.5, color=INK, va="top", linespacing=1.35)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((a + 26.4, 13), (b - 0.4, 13), arrowstyle="-|>", mutation_scale=18, lw=2, color=MUTED))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def modules(path):
    """Perception, scene graph and motion figures of the project page side by side, one colour bar and name each."""
    from PIL import Image, ImageDraw, ImageFont
    assets = ROOT / "docs/assets"
    panels = [(PERCEPTION, "Perception & mapping", "Donggeon Yoon", assets / "object_map.png"),
              (SEARCH, "Scene graph & search", "Seongmin Lee", assets / "scene_graph_layers.png"),
              (MOTION, "Motion", "Sanghun Moon", assets / "motion_paths.png")]
    h = 1000
    bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 44)
    reg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
    imgs = [Image.open(p[3]).convert("RGB") for p in panels]
    imgs = [im.resize((int(im.width * h / im.height), h), Image.LANCZOS) for im in imgs]
    gap, head = 50, 130
    W = sum(im.width for im in imgs) + gap * 2
    sheet = Image.new("RGB", (W, head + h), "white")
    d = ImageDraw.Draw(sheet)
    x = 0
    for (color, name, who, _), im in zip(panels, imgs):
        d.rectangle((x, 0, x + im.width, 14), fill=color)
        d.text((x, 28), name, fill=(22, 22, 26), font=bold)
        d.text((x, 84), who, fill=(92, 92, 99), font=reg)
        d.rectangle((x, head, x + im.width - 1, head + h - 1), outline=(221, 219, 214), width=2)
        sheet.paste(im, (x, head))
        x += im.width + gap
    sheet.save(path)


def qr(path, url):
    import segno
    segno.make(url, error="m").save(str(path), scale=24, border=3, dark="#16161a")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/poster")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "docs/assets/search_replay.json").read_text(encoding="utf-8"))
    results = json.loads((ROOT / "outputs/graph_compare/valtest_summary_objectnav.json").read_text(encoding="utf-8"))["results"]
    system(args.output / "poster_system.png")
    paired(args.output / "poster_paired.png", data["scenarios"])
    structure(args.output / "table_structure.png", results)
    system_compact(args.output / "poster_system_compact.png")
    modules(args.output / "poster_modules.png")
    qr(args.output / "poster_qr.png", PAGE)
    qr(args.output / "poster_qr_ko.png", PAGE + "?lang=ko")
    print("wrote", *sorted(p.name for p in args.output.glob("poster_*.png")), "table_structure.png")


if __name__ == "__main__":
    main()
