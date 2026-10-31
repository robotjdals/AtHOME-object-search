"""Qualitative search paths for the paper (English, the tables' serif font):
episodes as rows, policies as columns, drawn with the room map and path
reconstruction of scripts/plot_episode_map.py.

    python scripts/plot_qualitative_paths.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from athome.navigation.path_cost import extract_path, shortest_paths  # noqa: E402
from plot_episode_map import PATH, TARGET, load_row, room_image  # noqa: E402
from train_grpo import ROOT, Scenes  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Serif"
INK, MUTED, EDGE = "#212529", "#6c757d", "#ced4da"
# (state_id, row label): per category, the episode with the median cost reduction of SFT (ObjectNav start rule).
EPISODES = [("qyAac8rV8Zk:C:bin:0", "Seen category: bin"),
            ("cvZr5TUy5C5:B:printer:0", "Unseen category: printer")]
PANELS = [("Greedy (nearest first)", "outputs/paper/sft_trajectories_val.json:mincost"),
          ("Qwen3-4B (zero-shot)", "outputs/paper/base_trajectories_val.json:student"),
          ("Ours (SFT + GRPO)", "outputs/eval_replay/v1_val_trajectories.json:student")]
OUT = Path.home() / "Downloads/fig_qualitative_paths.png"


def draw(ax, problem, row, run, gt_rooms):
    grid = problem.grid
    rgba, centers = room_image(problem, gt_rooms)
    h, w = grid.free.shape
    x0, y0 = grid.origin
    ax.imshow(rgba, origin="lower", extent=(x0, x0 + w * grid.resolution, y0, y0 + h * grid.resolution),
              interpolation="nearest")
    for room_id, (cx, cy) in centers.items():
        label = problem.graph.rooms[room_id].label if room_id in problem.graph.rooms else room_id
        ax.text(cx, cy, label.replace("_", " "), ha="center", va="center", fontsize=8.5, color=MUTED)
    for obj in problem.world:
        ax.plot(obj.centroid[0], obj.centroid[1], marker="*", markersize=16, color=TARGET, markeredgecolor="white",
                markeredgewidth=1.0, zorder=6)
    start = (row["start"]["x_m"], row["start"]["y_m"])
    pose = grid.to_cell(*start)
    for k, visit in enumerate(run["trajectory"], 1):
        goal = grid.to_cell(visit["goal"]["x_m"], visit["goal"]["y_m"])
        _, parents = shortest_paths(grid.free, pose, [goal], grid.resolution)
        xy = np.asarray([grid.to_xy(c) for c in extract_path(parents, pose, goal)])
        if len(xy) > 1:
            ax.plot(xy[:, 0], xy[:, 1], color=PATH, linewidth=1.8, alpha=0.85, zorder=4)
        gx, gy = grid.to_xy(goal)
        found = visit["detected"]
        if found:
            ax.add_patch(plt.Circle((gx, gy), 3.0, fill=False, linestyle="--", linewidth=1.2, color=TARGET, alpha=0.7,
                                    zorder=5))
        ax.plot(gx, gy, marker="o", markersize=13, color=TARGET if found else PATH, markeredgecolor="white",
                markeredgewidth=1.2, zorder=7)
        ax.text(gx, gy, str(k), ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=8)
        pose = goal
    ax.plot(*start, marker="s", markersize=10, color=INK, markeredgecolor="white", markeredgewidth=1.2, zorder=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.set_aspect("equal")
    return f"{run['visits']} visit{'s' if run['visits'] != 1 else ''} · {run['distance_m']:.1f} m"


def main():
    grpo_cfg = yaml.safe_load((ROOT / "configs/training/grpo.yaml").read_text(encoding="utf-8"))
    scenes = Scenes(grpo_cfg["layouts"])
    fig, axes = plt.subplots(len(EPISODES), len(PANELS), figsize=(5.6 * len(PANELS), 5.0 * len(EPISODES)),
                             squeeze=False)
    for r, (state_id, row_label) in enumerate(EPISODES):
        runs = [(title, *load_row(source, state_id)) for title, source in PANELS]
        row = runs[0][1]
        problem = scenes.problems(row["scene_id"])[(row["component"], row["target"])]
        gt_rooms = set(row.get("gt_rooms") or problem.gt_rooms)
        for c, (title, prow, run) in enumerate(runs):
            stats = draw(axes[r][c], problem, prow, run, gt_rooms)
            axes[r][c].set_title(f"{title}\n{stats}" if r == 0 else stats, fontsize=12, loc="left", color=INK)
        axes[r][0].set_ylabel(row_label, fontsize=12.5, color=INK, labelpad=8)
    fig.suptitle("Search paths in unseen buildings (HM3D val).", x=0.01, ha="left", fontsize=14)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], marker="s", ls="", color=INK, markersize=9, label="start"),
               Line2D([], [], marker="o", ls="", color=PATH, markersize=10, label="visit (number = order)"),
               Line2D([], [], marker="o", ls="", color=TARGET, markersize=10, label="visit that found the target"),
               Line2D([], [], color=TARGET, ls="--", label="3 m view range"),
               Line2D([], [], marker="*", ls="", color=TARGET, markersize=14, label="target instance"),
               Patch(color="#d8ecd9", label="room containing the target")]
    fig.legend(handles=handles, loc="lower left", ncol=6, frameon=False, fontsize=10.5, bbox_to_anchor=(0.005, 0.0))
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(OUT, dpi=200, facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
