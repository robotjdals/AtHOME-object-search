"""Top views of search episodes for slides and the paper (the tables' serif
font): rooms filled per room with black walls, the room holding the target
highlighted, each policy's path and visits, and where the target was found.

Episodes (chosen by rule, not by look): Val start-rule episodes where Ours
(SFT + GRPO) found the target in 3-5 visits, zero-shot needed at least 3 more,
and l* is 5-20 m (59 episodes); of those whose visit reduction equals the
median (12), the first two by state id. Zero-shot trajectories were recorded
for them on the served planner (outputs/paper/base_trajectories_val_median.json,
same visits as the evaluation). Paths between visits are recomputed on the episode grid with the symbolic environment's
Dijkstra, so their length equals the logged visit costs.

    python scripts/plot_search_topview.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
import yaml  # noqa: E402

from athome.navigation.path_cost import extract_path, shortest_paths  # noqa: E402
from plot_episode_map import load_row  # noqa: E402
from train_grpo import ROOT, Scenes  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Serif"
INK, MUTED, EDGE = "#212529", "#6c757d", "#ced4da"
PATH, FOUND, TARGET = "#2a78d6", "#e34948", "#e34948"
ROOM_TINTS = ["#efede8", "#e4e1da", "#f5f4f0", "#e9e6df", "#dcd9d1"]
TARGET_ROOM, ROOM_EDGE, WALL = "#cfe8d1", "#bdb8ad", "#1f1f1f"
UNNAMED = {"unknown", "other"}
NUMBER_UP_TO = 12            # visit numbers only while they stay readable
MARGIN_M = 0.6
LABEL_CLEARANCE_M = 1.0
EPISODES = [("DYehNKdT76V:A:brush:4", "Target: brush"), ("GLAQ4DNUx5U:B:tv:3", "Target: TV")]
PANELS = [("Qwen3-4B (zero-shot)", "outputs/paper/base_trajectories_val_median.json:student"),
          ("Ours (SFT + GRPO)", "outputs/eval_replay/v1_val_trajectories.json:student")]
OUT = Path.home() / "Downloads/fig_search_topview_median.png"


def rooms(problem, gt_rooms, crop, avoid, cell=0.1):
    """Room floors from the semantic-mesh floor samples (0.2 m apart): each
    cell takes the room of the nearest sample within 0.15 m, each room closed
    and its holes (unsampled floor under furniture) filled; a line where two
    rooms meet; label anchors at the sample nearest each room's centroid
    among those at least LABEL_CLEARANCE_M from the ``avoid`` points (markers),
    else just below the room.
    Also returns the floor mask for the outer walls."""
    from scipy.ndimage import binary_closing, binary_fill_holes
    ids = sorted(problem.room_points)
    points = np.concatenate([np.asarray(problem.room_points[r])[:, :2] for r in ids])
    owner_of = np.concatenate([np.full(len(problem.room_points[r]), k) for k, r in enumerate(ids)])
    xs = np.arange(crop[0], crop[1], cell) + cell / 2
    ys = np.arange(crop[2], crop[3], cell) + cell / 2
    gx, gy = np.meshgrid(xs, ys)
    dist, nearest = cKDTree(points).query(np.column_stack([gx.ravel(), gy.ravel()]), distance_upper_bound=0.15)
    owner = np.where(dist < np.inf, owner_of[np.minimum(nearest, len(owner_of) - 1)], -1).reshape(gx.shape)
    for k in range(len(ids)):                         # fill a room's holes, never over another room
        filled = binary_fill_holes(binary_closing(owner == k, iterations=2))
        owner[filled & (owner < 0)] = k
    rgba = np.zeros((*owner.shape, 4))
    anchors = {}
    for k, room_id in enumerate(ids):
        mask = owner == k
        if mask.any():
            rgba[mask] = matplotlib.colors.to_rgba(TARGET_ROOM if room_id in gt_rooms
                                                   else ROOM_TINTS[k % len(ROOM_TINTS)])
            own = points[owner_of == k]
            to_center = ((own - own.mean(axis=0)) ** 2).sum(axis=1)
            clear = cKDTree(avoid).query(own)[0] >= LABEL_CLEARANCE_M
            if clear.any():
                anchors[room_id] = (*own[np.argmin(np.where(clear, to_center, np.inf))], "center")
            else:
                anchors[room_id] = (own[:, 0].mean(), own[:, 1].min() - 0.3, "top")
    seam = np.zeros(owner.shape, dtype=bool)          # cells where the room changes
    seam[1:] |= (owner[1:] >= 0) & (owner[:-1] >= 0) & (owner[1:] != owner[:-1])
    seam[:, 1:] |= (owner[:, 1:] >= 0) & (owner[:, :-1] >= 0) & (owner[:, 1:] != owner[:, :-1])
    rgba[seam] = matplotlib.colors.to_rgba(ROOM_EDGE)
    extent = (xs[0] - cell / 2, xs[-1] + cell / 2, ys[0] - cell / 2, ys[-1] + cell / 2)
    return rgba, extent, anchors, (xs, ys, owner >= 0)


def walls(ax, problem, crop):
    """The occluders of the observation model (walls and doors in the
    0.3-1.8 m band, configs/data/symbolic_observation.json), merged."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    window = box(crop[0], crop[2], crop[1], crop[3])
    merged = unary_union([g for g in problem.observer.walls.geoms if g.intersects(window)]).intersection(window)
    for poly in getattr(merged, "geoms", [merged]):
        if poly.geom_type == "Polygon" and not poly.is_empty:
            ax.fill(*poly.exterior.xy, color=WALL, linewidth=0.4, edgecolor=WALL, zorder=2)


def draw(ax, problem, row, run, gt_rooms, crop, avoid):
    grid = problem.grid
    rgba, extent, anchors, (xs, ys, floor) = rooms(problem, gt_rooms, crop, avoid)
    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest", zorder=1)
    ax.contour(xs, ys, floor.astype(float), levels=[0.5], colors=WALL, linewidths=1.6, zorder=2)   # outer walls
    walls(ax, problem, crop)
    halo = [pe.withStroke(linewidth=3, foreground="white")]
    for room_id, (cx, cy, va) in anchors.items():
        label = problem.graph.rooms[room_id].label if room_id in problem.graph.rooms else ""
        if label and label not in UNNAMED:
            ax.text(cx, cy, label.replace("_", " "), ha="center", va=va, fontsize=10,
                    color=INK if room_id in gt_rooms else MUTED, path_effects=halo, zorder=3)
    start = (row["start"]["x_m"], row["start"]["y_m"])
    pose = grid.to_cell(*start)
    line, visits, found_at = [np.asarray(start)], [], None
    for visit in run["trajectory"]:
        goal = grid.to_cell(visit["goal"]["x_m"], visit["goal"]["y_m"])
        _, parents = shortest_paths(grid.free, pose, [goal], grid.resolution)
        line += [np.asarray(grid.to_xy(c)) for c in extract_path(parents, pose, goal)[1:]]
        visits.append(grid.to_xy(goal))
        if visit["detected"] and found_at is None:
            found_at = len(visits) - 1
        pose = goal
    line = np.asarray(line)
    ax.plot(line[:, 0], line[:, 1], color=PATH, linewidth=2.4, alpha=0.9, solid_capstyle="round", zorder=4)
    numbered = len(visits) <= NUMBER_UP_TO
    for k, (gx, gy) in enumerate(visits):
        if k == found_at:
            continue
        ax.plot(gx, gy, marker="o", markersize=13 if numbered else 6, color=PATH, markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)
        if numbered:
            ax.text(gx, gy, str(k + 1), ha="center", va="center", fontsize=7.5, color="white",
                    fontweight="bold", zorder=6)
    for obj in problem.world:
        ax.plot(obj.centroid[0], obj.centroid[1], marker="*", markersize=19, color=TARGET,
                markeredgecolor="white", markeredgewidth=1.3, zorder=7)
    if found_at is not None:
        gx, gy = visits[found_at]
        ax.plot(gx, gy, marker="o", markersize=17, color=FOUND, markeredgecolor="white", markeredgewidth=1.5,
                zorder=8)
        ax.text(gx, gy, str(found_at + 1), ha="center", va="center", fontsize=8.5, color="white",
                fontweight="bold", zorder=9)
    ax.plot(*start, marker="s", markersize=12, color=INK, markeredgecolor="white", markeredgewidth=1.5, zorder=8)
    ax.text(start[0], start[1] - 0.9, "start", ha="center", va="top", fontsize=9.5, color=INK, path_effects=halo,
            zorder=8)
    ax.set_xlim(crop[0], crop[1])
    ax.set_ylim(crop[2], crop[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    plural = "s" if run["visits"] != 1 else ""
    outcome = f"{run['visits']} visit{plural} · {run['distance_m']:.1f} m"
    return outcome if found_at is not None else f"not found in {run['visits']} visits · {run['distance_m']:.1f} m"


def crop_of(problem, row):
    """The rooms' floors (and the start), plus a margin."""
    points = np.concatenate([np.asarray(v)[:, :2] for v in problem.room_points.values()])
    xs = list(points[:, 0]) + [row["start"]["x_m"]]
    ys = list(points[:, 1]) + [row["start"]["y_m"]]
    return min(xs) - MARGIN_M, max(xs) + MARGIN_M, min(ys) - MARGIN_M, max(ys) + MARGIN_M


def main():
    grpo_cfg = yaml.safe_load((ROOT / "configs/training/grpo.yaml").read_text(encoding="utf-8"))
    scenes = Scenes(grpo_cfg["layouts"])
    episodes = []
    for state_id, row_label in EPISODES:
        runs = [(title, *load_row(source, state_id)) for title, source in PANELS]
        row = runs[0][1]
        problem = scenes.problems(row["scene_id"])[(row["component"], row["target"])]
        episodes.append((row_label, runs, row, problem, crop_of(problem, row)))
    heights = [(c[3] - c[2]) / (c[1] - c[0]) for *_, c in episodes]       # rows keep the buildings' aspect
    panel_w = 5.6
    fig, axes = plt.subplots(len(EPISODES), len(PANELS), squeeze=False, gridspec_kw={"height_ratios": heights},
                             figsize=(panel_w * len(PANELS), sum(panel_w * h for h in heights) + 1.6))
    for r, (row_label, runs, row, problem, crop) in enumerate(episodes):
        gt_rooms = set(row.get("gt_rooms") or problem.gt_rooms)
        # Room labels stay clear of every marker of the row, at the same place in each panel.
        avoid = np.array([[v["goal"]["x_m"], v["goal"]["y_m"]] for _, _, run in runs for v in run["trajectory"]]
                         + [o.centroid[:2] for o in problem.world] + [[row["start"]["x_m"], row["start"]["y_m"]]])
        for c, (title, prow, run) in enumerate(runs):
            stats = draw(axes[r][c], problem, prow, run, gt_rooms, crop, avoid)
            heading = f"{title}\n" if r == 0 else ""
            axes[r][c].set_title(f"{heading}{stats}", fontsize=12.5, loc="left", color=INK)
        axes[r][0].set_ylabel(row_label, fontsize=13, color=INK, labelpad=8)
    handles = [Line2D([], [], marker="s", ls="", color=INK, markersize=10, label="start"),
               Line2D([], [], color=PATH, lw=2.4, marker="o", markersize=9, markeredgecolor="white",
                      label="path and visits (number = order)"),
               Line2D([], [], marker="o", ls="", color=FOUND, markersize=12, markeredgecolor="white",
                      label="visit that found the target"),
               Line2D([], [], marker="*", ls="", color=TARGET, markersize=16, markeredgecolor="white",
                      label="target object"),
               Patch(facecolor=TARGET_ROOM, edgecolor=ROOM_EDGE, label="room with the target"),
               Line2D([], [], color=WALL, lw=1.2, label="wall")]
    ncol = len(handles) if len(PANELS) > 2 else 3       # two rows when the figure is narrow
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04 if ncol == len(handles) else 0.07, 1, 1))
    fig.savefig(OUT, dpi=200, facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
