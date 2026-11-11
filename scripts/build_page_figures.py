"""Figures of the perception and motion modules for the project page (docs/assets/),
drawn from the data of the test home and the motion logs of the final experiment:

    perception_frames.jpg  detector output on three keyframes of the mapping drive
    object_map.png         the fused 3D object boxes on the navigation map
    motion_paths.png       every path driven in the final experiment
    motion_stats.json      the numbers quoted on the page (printed too)

    python3 scripts/build_page_figures.py
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Polygon
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/real_baram/real_room_baram_dataset_20261105"
SCENE = DATASET / "scene_graph_dataset/scenes/robot_room_baram"   # what the perception module delivered
VIEW = (-2.1, 6.9, -6.5, 3.5)   # x0, x1, y0, y1 [m] in the map frame: the four rooms
KEYFRAMES = (40, 27, 59)        # living room, office, kitchen
DRIVE_MIN_M = 0.3               # a goal that moves less is an in-place observation turn
INK, MUTED = "#16161a", "#5c5c63"
GREEN, ORANGE, FURNITURE = "#1a9e5c", "#eb6834", "#8b95a5"


def draw_map(ax, map_yaml, margin=0.0, fontsize=9):
    meta = yaml.safe_load(map_yaml.read_text())
    grid = np.asarray(Image.open(map_yaml.parent / meta["image"]).convert("L"))
    shade = np.full(grid.shape + (4,), (1.0, 1.0, 1.0, 0.0))    # unknown: nothing
    shade[grid > 250] = (0.955, 0.953, 0.94, 1.0)               # free
    shade[grid < 50] = (0.25, 0.25, 0.28, 1.0)                  # occupied
    x0, y0 = meta["origin"][:2]
    res = meta["resolution"]
    columns = x0 + (np.arange(grid.shape[1]) + 0.5) * res
    shade[:, (columns < VIEW[0]) | (columns > VIEW[1]), 3] = 0   # walls of other rooms: not in the margin
    ax.imshow(shade, extent=(x0, x0 + grid.shape[1] * res, y0, y0 + grid.shape[0] * res), interpolation="nearest", zorder=0)
    ax.set_xlim(VIEW[0] - margin, VIEW[1] + margin)
    ax.set_ylim(VIEW[2], VIEW[3])
    ax.set_aspect("equal")
    ax.axis("off")
    # 1 m scale bar
    bx, by = VIEW[0] - margin + 0.3, VIEW[2] + 0.35
    ax.plot([bx, bx + 1], [by, by], color=INK, lw=2, solid_capstyle="butt")
    ax.text(bx + 0.5, by + 0.12, "1 m", ha="center", va="bottom", fontsize=fontsize, color=INK)


def figure(margin=0.0):
    """Canvas for a top view of the home; margin [m] adds room for names left and right."""
    height = 7.0
    width = height * (VIEW[1] - VIEW[0] + 2 * margin) / (VIEW[3] - VIEW[2])
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def spread(ys, gap, low, high):
    """Label heights as near to ys (sorted) as a minimum gap allows, inside [low, high]."""
    out = []
    for y in ys:
        out.append(max(y, out[-1] + gap) if out else max(y, low))
    for i in range(len(out) - 1, -1, -1):
        out[i] = min(out[i], high if i == len(out) - 1 else out[i + 1] - gap)
    return out


def perception_frames(out):
    files = sorted((DATASET / "robot_recording/annotated").glob("*.jpg"))
    tiles = [Image.open(files[i]).resize((640, 360), Image.LANCZOS) for i in KEYFRAMES]
    gap = 6
    sheet = Image.new("RGB", (640 * 3 + gap * 2, 360), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * (640 + gap), 0))
    sheet.save(out, quality=86)
    return len(files)


def object_map(graph, out, map_yaml):
    """Boxes and fused LiDAR points as the perception module delivered them, names as the graph uses them."""
    boxes = {o["object_id"]: o["bbox_3d"] for o in json.loads((SCENE / "objects.json").read_text())["objects"]}
    margin = 2.6
    fig, ax = figure(margin)
    draw_map(ax, map_yaml, margin)
    centre = np.mean([o["bbox"]["center"][:2] for o in graph["objects"]], axis=0)
    labels = []
    for o in graph["objects"]:
        number = int(o["object_id"].rsplit("_", 1)[1])
        box = boxes[number]
        furniture = o["role"] == "source"
        points = np.load(SCENE / f"points/obj_{number:04d}.npy")
        ax.scatter(points[:, 0], points[:, 1], s=1.6, color=FURNITURE if furniture else GREEN, alpha=0.55, lw=0, zorder=3 if furniture else 4)
        (cx, cy), (sx, sy), yaw = box["center"][:2], box["size"][:2], box["yaw"]
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [(cx + c * dx - s * dy, cy + s * dx + c * dy) for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))]
        ax.add_patch(Polygon(corners, closed=True, zorder=5, lw=1.0 if furniture else 1.4, facecolor="none",
                             edgecolor=FURNITURE if furniture else INK))
        if not furniture:
            labels.append((o["semantic_tag"], cx, cy))
    # Names sit outside the walls on the side of their object, at its height where there is room.
    for side in (-1, 1):
        group = sorted((l for l in labels if (l[1] > centre[0]) == (side > 0)), key=lambda l: l[2])
        ys = spread([l[2] for l in group], 0.42, VIEW[2] + 1.0, VIEW[3] - 0.4)
        tx = VIEW[1] + 0.9 if side > 0 else VIEW[0] - 0.9
        for (name, x, y), ty in zip(group, ys):
            ax.annotate(name, (x, y), (tx, ty), fontsize=10, color=INK, va="center", ha="left" if side > 0 else "right", zorder=6,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=3, shrinkB=0))
    ax.scatter([], [], s=40, color=GREEN, label="object: fused LiDAR points and 3D box")
    ax.scatter([], [], s=40, color=FURNITURE, label="furniture (desk)")
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK, borderaxespad=1.0, handletextpad=0.3)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    assert len(labels) > 10, "object boxes missing"
    return len(graph["objects"])


def motion(run_dir, out, map_yaml):
    fig, ax = figure()
    draw_map(ax, map_yaml, fontsize=14)   # shown at half the page width
    drives, turns, stops, solve_ms, speeds = [], 0, [], [], []
    for path in sorted((run_dir / "motion").glob("real_*.csv")):
        rows = list(csv.DictReader(open(path)))
        if not rows:
            continue
        xy = np.array([(float(r["x"]), float(r["y"])) for r in rows if r["phase"] == "driving"])
        length = float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0
        solve_ms += [float(r["mppi_ms"]) for r in rows if r["mppi_ms"]]
        stopping = [float(r["t"]) for r in rows if r["phase"] == "stopping"]
        if stopping:
            stops.append(stopping[-1] - stopping[0])
        if length < DRIVE_MIN_M:
            turns += 1
            continue
        drives.append(length)
        speeds.append(max(float(r["v_meas"]) for r in rows if r["v_meas"]))
        ax.plot(xy[:, 0], xy[:, 1], color=ORANGE, lw=1.3, alpha=0.38, solid_capstyle="round", zorder=3)
        end = (float(rows[-1]["x"]), float(rows[-1]["y"]))
        ax.plot(*end, "o", ms=3.4, color=INK, mec="white", mew=0.5, zorder=4)
    ax.plot([], [], color=ORANGE, lw=2, label="driven path")
    ax.plot([], [], "o", ms=4.5, color=INK, mec="white", mew=0.5, ls="", label="measured stop")
    ax.legend(loc="upper right", frameon=False, fontsize=14, labelcolor=INK, borderaxespad=0.6)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    assert drives and turns and stops, "no motion logs found"
    return {"goals": len(drives) + turns, "drives": len(drives), "turns": turns,
            "driven_m": round(sum(drives), 1), "drive_median_m": round(statistics.median(drives), 2),
            "top_speed_median_mps": round(statistics.median(speeds), 2),
            "solve_ms_median": round(statistics.median(solve_ms), 1),
            "solve_ms_p99": round(float(np.percentile(solve_ms, 99)), 1),
            "stop_s_median": round(statistics.median(stops), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", type=Path, default=ROOT / "outputs/room_baram/graph.json")
    ap.add_argument("--runs", type=Path, default=ROOT / "outputs/room_baram/experiments/room_baram_final")
    ap.add_argument("--map", type=Path, default=ROOT / "data/real_baram/demo/room_baram_demo.yaml")
    ap.add_argument("--output", type=Path, default=ROOT / "docs/assets")
    args = ap.parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    stats = {"keyframes": perception_frames(args.output / "perception_frames.jpg"),
             "objects": object_map(graph, args.output / "object_map.png", args.map),
             "motion": motion(args.runs, args.output / "motion_paths.png", args.map)}
    (args.output / "motion_stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
