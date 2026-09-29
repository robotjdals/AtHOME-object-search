"""Pairwise agreement of the 2D wall line-of-sight model with rendering.

Pairs: every goal-pose candidate of every Search Location in the unmasked
component graphs x every target-category instance on the component's floor
whose footprint is within the sensor range. For each pair:
- wall_los_2d: the symbolic model (configs/data/symbolic_observation.json);
- render_oracle: Habitat semantic rendering at 12 headings (30 deg), i.e. the
  ObjectNav oracle-visibility notion (seen by turning in place) -- the
  counterpart of an isotropic sensor;
- render_4: the 4 VisitExecutor headings only (field-of-view gaps included).
Rendering uses the reference camera (height is a placeholder until measured).
``--reuse-render`` recomputes only wall_los_2d on the pairs of an earlier
report made with the same rendering settings. Agreement is also reported
without high-mounted instances (bottom more than ``HIGH_MOUNT_M`` above the
room floor, the height rule SceneGraph applies to Search Locations).
Read-only; writes one report.
"""
import argparse
import contextlib
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from athome.navigation.goal_poses import goal_candidates
from athome.navigation.grid import GridMap
from athome.symbolic.habitat_observer import CameraSpec, HabitatSemanticObserver, VisibilitySpec
from athome.symbolic.wall_los import WallLosSpec

import run_symbolic_review as runner
from athome.data.hm3d.floors import component_environment, on_floor

ORACLE_HEADINGS = 12
from athome.scene_graph.query import floor_references  # noqa: E402
HIGH_MOUNT_M = 1.5


def rect_distance(x, y, lo, hi):
    return math.hypot(max(lo[0] - x, 0.0, x - hi[0]), max(lo[1] - y, 0.0, y - hi[1]))


def kappa(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def confusion(pairs, a, b):
    c = Counter((p[a], p[b]) for p in pairs)
    tp, fp, fn, tn = c[(True, True)], c[(True, False)], c[(False, True)], c[(False, False)]
    return {"both_visible": tp, f"{a}_only": fp, f"{b}_only": fn, "both_hidden": tn,
            "agreement": (tp + tn) / max(len(pairs), 1), "cohen_kappa": kappa(tp, fp, fn, tn)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-config", type=Path,
                        default=runner.ROOT / "configs/data/symbolic_observation.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-render", type=Path, help="같은 렌더 설정의 이전 보고서")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    cfg = runner.read(args.observation_config)
    w, r = cfg["wall_los_2d"], cfg["habitat_render"]
    spec = WallLosSpec(float(w["range_m"]), tuple(w["occluder_categories"]),
                       tuple(w["band_above_floor_m"]), float(w["contact_tolerance_m"]))
    camera = CameraSpec(float(r["camera"]["height_m"]), float(r["camera"]["hfov_deg"]),
                        *map(int, r["camera"]["resolution_wh"]))
    visibility = VisibilitySpec(float(r["visibility"]["max_range_m"]),
                                int(r["visibility"]["min_visible_pixels"]))
    if visibility.max_range_m != spec.range_m:
        raise SystemExit("두 모델의 관측 거리가 달라 비교할 수 없습니다.")

    layout = runner.LAYOUT
    annotation = runner.read(layout.annotations)
    scene = Path(annotation["scene_path"])
    glb = scene.with_name(scene.name.replace(".basis.glb", ".semantic.glb"))
    walls = runner.wall_triangles(glb, glb.with_suffix(".txt"), spec.occluder_categories)
    nav = {c["component"]: c for c in runner.read(layout.navigation)["components"]}
    triangles = {c["component"]: c["triangle_ids"] for c in runner.read(layout.room_report)["components"]}
    floor_plan = runner.read(layout.floors)
    report_components = {c["component"]: c for c in runner.read(layout.room_report)["components"]}
    targets = runner.read(layout.targets)["target_categories"]
    catalog = {t["target_category"]: t["instances"] for t in runner.read(layout.catalog)["targets"]}

    previous = None
    if args.reuse_render:
        previous = runner.read(args.reuse_render)
        if previous["render_settings"] != r:
            raise SystemExit("렌더 설정이 달라 재사용할 수 없습니다.")
        cached = {(p["component"], tuple(p["pose"]), p["object_id"]): p for p in previous["pairs_detail"]}
    pairs = []
    with (contextlib.nullcontext() if previous else
          HabitatSemanticObserver(annotation["scene_path"], annotation["config_path"],
                                  camera, visibility)) as renderer:
        for name in layout.component_names():
            graph = runner.read(layout.scene_dir / "component_graphs.review" / f"{name}.workspace_graph.review.json")
            rooms = {room["room_id"] for room in graph["rooms"]}
            comp_env = component_environment(floor_plan, report_components[name], rooms)
            instances = {runner.semantic_id(o["object_id"]): (t, o) for t in targets
                         for o in catalog[t] if on_floor(comp_env, o)}
            with np.load(runner.GRID / f"{name}.grid.npz", allow_pickle=False) as saved:
                grid = GridMap(saved["free"].copy(), tuple(saved["origin_xy_m"]),
                               float(saved["resolution_m"].item()))
            surface = runner.load_surface(name, triangles[name])
            floor_z = floor_references(graph)
            observer, _ = runner.wall_observer(
                name, {sid: (o["bbox"]["min"], o["bbox"]["max"]) for sid, (_, o) in instances.items()},
                spec, walls)
            objects = {o["object_id"]: o for o in graph["objects"]}
            boxes = [objects[ws["source_object_id"]]["bbox"] for ws in graph["workspaces"]]
            boxes += [o["bbox"] for o in graph["objects"] if o["role"] == "standalone"]
            poses = {}
            for box in boxes:
                for cell, pose in goal_candidates(grid, box["min"], box["max"],
                                                  nav[name]["footprint_offset_m"],
                                                  float(nav[name].get("goal_clearance_m", 0.0)),
                                                  nav[name].get("goal_max_offset_m")):
                    poses[(cell, round(pose.yaw, 6))] = pose
            print(f"{name}: 목표 후보 자세 {len(poses)}개, 같은 층 목표 인스턴스 {len(instances)}개", flush=True)
            for pose in poses.values():
                near = {sid: v for sid, v in instances.items()
                        if rect_distance(pose.x, pose.y, v[1]["bbox"]["min"], v[1]["bbox"]["max"]) <= spec.range_m}
                if not near:
                    continue
                z = surface.height_at(pose.x, pose.y)
                seen4, seen_oracle = set(), set()
                if previous is None:
                    for k in range(ORACLE_HEADINGS):
                        yaw = pose.yaw + 2 * math.pi * k / ORACLE_HEADINGS
                        visible = set(renderer.observe(pose.x, pose.y, z, yaw)) & set(near)
                        seen_oracle |= visible
                        if k % (ORACLE_HEADINGS // 4) == 0:
                            seen4 |= visible
                seen2d = observer.observe(pose.x, pose.y, z, pose.yaw)
                for sid, (target, obj) in near.items():
                    key = [pose.x, pose.y, pose.yaw]
                    if previous is not None:
                        old = cached.get((name, tuple(key), obj["object_id"]))
                        if old is None:  # reuse only covers pairs rendered before
                            raise SystemExit("이전 보고서에 없는 쌍이 있어 렌더를 재사용할 수 없습니다.")
                        in_oracle, in_4 = old["render_oracle"], old["render_4"]
                    else:
                        in_oracle, in_4 = sid in seen_oracle, sid in seen4
                    reference = floor_z.get(obj["room_id"])
                    pairs.append({
                        "component": name, "pose": key,
                        "object_id": obj["object_id"], "target": target, "room_id": obj["room_id"],
                        "distance_m": rect_distance(pose.x, pose.y, obj["bbox"]["min"], obj["bbox"]["max"]),
                        "bottom_above_floor_m": (obj["bbox"]["min"][2] - reference) if reference is not None else None,
                        "wall_los_2d": sid in seen2d, "render_oracle": in_oracle, "render_4": in_4})
    low = [p for p in pairs if p["bottom_above_floor_m"] is None or p["bottom_above_floor_m"] <= HIGH_MOUNT_M]
    report = {
        "status": "validation_review", "camera_height_placeholder": camera.height_m,
        "render_settings": r, "wall_los_settings": w,
        "render_reused_from": str(args.reuse_render) if args.reuse_render else None,
        "oracle_headings": ORACLE_HEADINGS, "pairs": len(pairs),
        "wall_los_2d_vs_render_oracle": confusion(pairs, "wall_los_2d", "render_oracle"),
        "render_4_vs_render_oracle": confusion(pairs, "render_4", "render_oracle"),
        "wall_los_2d_vs_render_4": confusion(pairs, "wall_los_2d", "render_4"),
        "excluding_high_mounted": {"threshold_m": HIGH_MOUNT_M, "pairs": len(low),
                                   "wall_los_2d_vs_render_oracle": confusion(low, "wall_los_2d", "render_oracle")},
        "by_target": {t: confusion([p for p in pairs if p["target"] == t], "wall_los_2d", "render_oracle")
                      for t in targets if any(p["target"] == t for p in pairs)},
        "disagreements": [p for p in pairs if p["wall_los_2d"] != p["render_oracle"]],
        "pairs_detail": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    for key in ("wall_los_2d_vs_render_oracle", "render_4_vs_render_oracle", "wall_los_2d_vs_render_4"):
        print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report[key].items()})
    print("높이 1.5 m 초과 제외:", {k: (round(v, 3) if isinstance(v, float) else v)
          for k, v in report["excluding_high_mounted"]["wall_los_2d_vs_render_oracle"].items()})
    print("저장:", args.output)


if __name__ == "__main__":
    main()
