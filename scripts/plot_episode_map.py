"""Top-down map of one evaluation episode: rooms, the target instances, the
start, and each policy's visits with the shortest paths it travelled.

Inputs are evaluate_student.py episode files written with --log-trajectory.
Each panel is one policy of one file ("<file>:<policy>"). Paths between
visits are recomputed on the episode's grid with the same Dijkstra the
symbolic environment uses, so their length equals the logged visit costs.

    python scripts/plot_episode_map.py --state-id svBbv1Pavdk:A:blanket:0 \
        --panel "기본 Qwen3-4B=outputs/midterm/base_cases_val.json:student" \
        --panel "SFT + GRPO (v1)=outputs/eval_replay/v1_val_trajectories.json:student" \
        --panel "가장 가까운 곳부터=outputs/eval_replay/v1_val_trajectories.json:mincost" \
        --out outputs/midterm/figures/map_blanket.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from athome.navigation.path_cost import extract_path, shortest_paths
from plot_midterm_figures import GRID, INK, INK2, MUTED, SURFACE, setup
from train_grpo import ROOT, Scenes

PATH = "#2a78d6"
TARGET = "#e34948"
MISSED, MISSED_TEXT = "#eda100", "#8a5d00"
ROOM_TINTS = ["#ecebe7", "#e2e0da", "#f3f2ef", "#dedcd5"]
TARGET_ROOM = "#d8ecd9"


def load_row(spec, state_id):
    path, policy = spec.rsplit(":", 1)
    rows = {r["state_id"]: r for r in json.loads((ROOT / path).read_text(encoding="utf-8"))["rows"]}
    row = rows[state_id]
    if "trajectory" not in row[policy]:
        raise SystemExit(f"{path}: {policy}에 trajectory가 없습니다 (--log-trajectory로 평가한 파일이 필요)")
    return row, row[policy]


def room_image(problem, gt_rooms):
    """Every free cell takes the room of its nearest floor sample (within 0.5 m)."""
    from scipy.spatial import cKDTree
    grid = problem.grid
    rgba = np.zeros((*grid.free.shape, 4))
    rgba[grid.free] = matplotlib.colors.to_rgba("#f7f6f3")
    ids = sorted(problem.room_points)
    points = np.concatenate([np.asarray(problem.room_points[r])[:, :2] for r in ids])
    owner = np.concatenate([np.full(len(problem.room_points[r]), k) for k, r in enumerate(ids)])
    rows, cols = np.nonzero(grid.free)
    xy = np.column_stack([grid.origin[0] + grid.resolution * (cols + 0.5), grid.origin[1] + grid.resolution * (rows + 0.5)])
    dist, nearest = cKDTree(points).query(xy, distance_upper_bound=0.5)
    centers = {}
    for k, room_id in enumerate(ids):
        mask = (dist < np.inf) & (owner[np.minimum(nearest, len(owner) - 1)] == k)
        color = matplotlib.colors.to_rgba(TARGET_ROOM if room_id in gt_rooms else ROOM_TINTS[k % len(ROOM_TINTS)])
        rgba[rows[mask], cols[mask]] = color
        if mask.any():
            centers[room_id] = xy[mask].mean(axis=0)
    return rgba, centers


def draw(ax, problem, row, run, title, gt_rooms):
    grid = problem.grid
    rgba, centers = room_image(problem, gt_rooms)
    h, w = grid.free.shape
    x0, y0 = grid.origin
    extent = (x0, x0 + w * grid.resolution, y0, y0 + h * grid.resolution)
    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")
    for room_id, (cx, cy) in centers.items():
        label = problem.graph.rooms[room_id].label if room_id in problem.graph.rooms else room_id
        ax.text(cx, cy, label.replace("_", " "), ha="center", va="center", fontsize=9, color=MUTED)
    for obj in problem.world:
        ax.plot(obj.centroid[0], obj.centroid[1], marker="*", markersize=17, color=TARGET,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=6)
    start = (row["start"]["x_m"], row["start"]["y_m"])
    pose = grid.to_cell(*start)
    for k, visit in enumerate(run["trajectory"], 1):
        goal = grid.to_cell(visit["goal"]["x_m"], visit["goal"]["y_m"])
        _, parents = shortest_paths(grid.free, pose, [goal], grid.resolution)
        cells = extract_path(parents, pose, goal)
        xy = np.asarray([grid.to_xy(c) for c in cells])
        if len(xy) > 1:
            ax.plot(xy[:, 0], xy[:, 1], color=PATH, linewidth=2, alpha=0.85, zorder=4)
        gx, gy = grid.to_xy(goal)
        found = visit["detected"]
        if found:       # the 3 m observation range (360 degrees) from which the target was seen
            ax.add_patch(plt.Circle((gx, gy), 3.0, fill=False, linestyle="--", linewidth=1.5, color=TARGET, alpha=0.7, zorder=5))
        missed = visit.get("missed") and not found
        if missed:      # the target was in view but the detector missed it
            ax.annotate("놓침", (gx, gy), textcoords="offset points", xytext=(10, 8), color=MISSED_TEXT, fontsize=12,
                        fontweight="bold", zorder=9)
        ax.plot(gx, gy, marker="o", markersize=15, color=TARGET if found else MISSED if missed else PATH, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=7)
        ax.text(gx, gy, str(k), ha="center", va="center", fontsize=8, color=SURFACE, fontweight="bold", zorder=8)
        pose = goal
    ax.plot(*start, marker="s", markersize=11, color=INK, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=7)
    ax.set_title(f"{title}\n방문 {run['visits']}회 · 이동 {run['distance_m']:.1f} m · 비용 {run['cost']:.1f}",
                 fontsize=13, loc="left")
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_aspect("equal")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--panel", action="append", required=True, metavar="TITLE=FILE:POLICY")
    parser.add_argument("--title", default=None)
    parser.add_argument("--legend", default=None, help="아래쪽 범례 문구 (기본: 표준 범례)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    setup()
    grpo_cfg = yaml.safe_load((ROOT / "configs/training/grpo.yaml").read_text(encoding="utf-8"))
    panels = []
    for spec in args.panel:
        title, source = spec.split("=", 1)
        panels.append((title, *load_row(source, args.state_id)))
    row = panels[0][1]
    problem = Scenes(grpo_cfg["layouts"]).problems(row["scene_id"])[(row["component"], row["target"])]
    gt_rooms = set(row.get("gt_rooms") or problem.gt_rooms)
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 6.4))
    for ax, (title, prow, run) in zip(np.atleast_1d(axes), panels):
        draw(ax, problem, prow, run, title, gt_rooms)
    fig.suptitle(args.title or f"목표: {row['target']}", x=0.01, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.01, 0.01, args.legend or "■ 출발   ● 방문(번호 = 순서, 빨강 = 찾은 곳)   점선 원 = 관측 범위 3 m   ★ 목표 물체 실제 위치   초록 방 = 목표가 있는 방   파란 선 = 이동 경로",
             fontsize=11, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
