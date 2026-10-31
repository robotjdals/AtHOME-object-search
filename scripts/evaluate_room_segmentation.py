"""Room segmentation (proposal 4-1) against HM3DSem regions.

The simulated scene graphs take their rooms from the HM3DSem region
annotations; the robot segments its own map. This measures that
segmentation on the Val/Test buildings the way HOV-SG does on HM3DSem
(regions pooled from the annotated objects' region labels as ground truth,
precision / recall "based on the metric in Hydra"):

  precision  per estimated room, its largest overlap with one GT room / its size
  recall     per GT room, its largest overlap with one estimated room / its size

averaged over rooms, on the free cells of the 2D grid (the criterion of
Bormann et al., ICRA 2016, for 2D maps).

Input per floor, as on the robot: free space from walls only. The floor area
is the union of the floors' semantic-mesh polygons; walls are the occluders
of the training observation model (walls and doors, configs/data/
symbolic_observation.json) in the floor's wall band, as the robot's wall
band of map.pcd. Furniture does not split rooms (the robot clears it too).

Baseline: morphological segmentation (Bormann et al. 2016, ipa_room_segmentation
defaults: rooms 0.8-47 m^2 after erosion, then wavefront growth).

    python scripts/evaluate_room_segmentation.py --scenes val test \
        --output outputs/room_segmentation/val_test.json
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import yaml
from scipy import ndimage
from shapely import contains_xy
from shapely.ops import unary_union

from athome.data.hm3d.floors import component_environment, on_floor
from athome.data.hm3d.layout import load_layout
from athome.scene_graph.room_segmentation import SegmentationConfig, segment_rooms

from compare_graph_representations import holdout_states
from scene_episodes import SceneProblems
import run_symbolic_review as runner
from train_grpo import ROOT

MORPHOLOGICAL_ROOM_M2 = (0.8, 47.0)    # ipa_room_segmentation room_area_factor_*_morphological


def raster(geometry, shape, origin, resolution):
    """Cells whose centers lie in ``geometry`` (row = +y, col = +x)."""
    rows, cols = np.indices(shape)
    xs = origin[0] + (cols + 0.5) * resolution
    ys = origin[1] + (rows + 0.5) * resolution
    return contains_xy(geometry, xs.ravel(), ys.ravel()).reshape(shape)


def morphological_rooms(free, resolution, area_m2=MORPHOLOGICAL_ROOM_M2):
    """Erode free space one cell at a time; a component whose area falls in
    the room range becomes a room and leaves the map. Rooms then grow over
    the remaining free cells (wavefront: nearest room)."""
    lo, hi = (a / resolution ** 2 for a in area_m2)
    rooms = np.zeros(free.shape, np.int32)
    left, n = free.copy(), 0
    while left.any():
        left = ndimage.binary_erosion(left)
        components, count = ndimage.label(left)
        sizes = ndimage.sum(left, components, range(1, count + 1))
        for comp, size in enumerate(sizes, 1):
            if lo <= size <= hi:
                n += 1
                mask = components == comp
                rooms[mask] = n
                left[mask] = False
    if n == 0:
        return rooms
    _, (r, c) = ndimage.distance_transform_edt(rooms == 0, return_indices=True)
    grown = rooms[r, c]
    # Growth stays inside the free space it can reach (no jumps over walls).
    out = np.zeros_like(rooms)
    components, count = ndimage.label(free)
    for comp in range(1, count + 1):
        mask = components == comp
        if rooms[mask].any():
            out[mask] = grown[mask]
    return out


def precision_recall(estimated, truth, free):
    """Per-room scores on free cells. ``estimated``/``truth``: int labels, 0 = none."""
    est, gt = estimated[free], truth[free]
    pairs = defaultdict(int)
    for e, g in zip(est.tolist(), gt.tolist()):
        if e and g:
            pairs[(e, g)] += 1
    est_size = {e: int((est == e).sum()) for e in np.unique(est) if e}
    gt_size = {g: int((gt == g).sum()) for g in np.unique(gt) if g}
    precision = [max((v for (e2, _), v in pairs.items() if e2 == e), default=0) / size
                 for e, size in est_size.items()]
    recall = [max((v for (_, g2), v in pairs.items() if g2 == g), default=0) / size
              for g, size in gt_size.items()]
    return precision, recall


def floor_case(problems, layout, name, spec):
    """Free space, GT room labels and grid of one floor (navigable component)."""
    unmasked = runner.read(layout.scene_dir / "component_graphs.review" / f"{name}.workspace_graph.review.json")
    room_ids = sorted(r["room_id"] for r in unmasked["rooms"])
    env = component_environment(problems._floor_plan, problems._report[name], set(room_ids))
    with np.load(layout.grid_dir / f"{name}.grid.npz", allow_pickle=False) as saved:
        origin, resolution = tuple(saved["origin_xy_m"]), float(saved["resolution_m"].item())
        shape = saved["free"].shape
    polygons = defaultdict(list)
    for obj, polys in problems._floor_polygons.values():
        if obj["room_id"] in room_ids and on_floor(env, obj):
            polygons[obj["room_id"]].extend(polys)
    truth = np.zeros(shape, np.int32)
    names = {}
    for i, rid in enumerate(sorted(polygons), 1):
        cells = raster(unary_union(polygons[rid]), shape, origin, resolution) & (truth == 0)
        truth[cells] = i
        names[i] = rid
    geometry, _ = runner.wall_geometry(name, spec, problems._walls, layout)
    walls = (raster(unary_union(geometry.geoms).buffer(resolution / 2), shape, origin, resolution)
             if geometry.geoms else np.zeros(shape, bool))
    free = (truth > 0) & ~walls
    return free, truth, origin, resolution, names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--start-states", default="outputs/eval_v5/*/start_states.jsonl",
                        help="평가 건물 목록 (scripts/compare_graph_representations.py와 같은 건물)")
    parser.add_argument("--limit", type=int, help="건물 수 제한 (점검용)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grpo_cfg = yaml.safe_load((ROOT / "configs/training/grpo.yaml").read_text(encoding="utf-8"))
    fraction = yaml.safe_load((ROOT / "configs/training/sft_lora.yaml").read_text(
        encoding="utf-8"))["validation"]["holdout_scene_fraction"]
    buildings = []
    for split in args.scenes:
        states, _ = holdout_states(args.start_states, grpo_cfg, fraction, split)
        buildings += [(split, s) for s in sorted({x["scene_id"] for x in states})]
    if args.limit:
        buildings = buildings[:args.limit]

    floors = []
    for k, (split, scene_id) in enumerate(buildings, 1):
        layout = load_layout(ROOT / grpo_cfg["layouts"].format(scene_id=scene_id))
        problems = SceneProblems(layout=layout)
        for name in layout.component_names():
            free, truth, origin, resolution, names = floor_case(problems, layout, name, problems._spec)
            if not free.any():
                continue
            ours = segment_rooms(free, origin, resolution, SegmentationConfig()).labels
            baseline = morphological_rooms(free, resolution)
            row = {"split": split, "scene_id": scene_id, "component": name,
                   "gt_rooms": len(np.unique(truth[free])) - (0 in truth[free]),
                   "free_m2": float(free.sum()) * resolution ** 2}
            for method, labels in (("ours", ours), ("morphological", baseline)):
                p, r = precision_recall(labels, truth, free)
                row[method] = {"rooms": len(p), "precision": p, "recall": r}
            floors.append(row)
            print(f"[{k}/{len(buildings)}] {scene_id} {name}: GT {row['gt_rooms']}, "
                  f"ours {row['ours']['rooms']} (P {statistics.mean(row['ours']['precision'] or [0]):.2f} "
                  f"R {statistics.mean(row['ours']['recall'] or [0]):.2f}), "
                  f"morph {row['morphological']['rooms']} "
                  f"(P {statistics.mean(row['morphological']['precision'] or [0]):.2f} "
                  f"R {statistics.mean(row['morphological']['recall'] or [0]):.2f})", flush=True)

    summary = {}
    for method in ("ours", "morphological"):
        p = [v for f in floors for v in f[method]["precision"]]
        r = [v for f in floors for v in f[method]["recall"]]
        summary[method] = {"precision": statistics.mean(p), "recall": statistics.mean(r),
                           "rooms": sum(f[method]["rooms"] for f in floors)}
    summary["gt_rooms"] = sum(f["gt_rooms"] for f in floors)
    summary["floors"], summary["buildings"] = len(floors), len({f["scene_id"] for f in floors})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "floors": floors}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
