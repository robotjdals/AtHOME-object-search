"""Run A/B masked graphs with shared search/navigation and evaluator-only GT.

Observation (configs/data/symbolic_observation.json):
- wall_los_2d (default): range + 2D line of sight against walls/doors at each
  reached goal (grid sensor model of multi-object search, Wandzel 2019);
- habitat_render: Habitat semantic camera at the VisitExecutor headings
  (reference for validating wall_los_2d; needs the camera height).
Only instances on the component's floor are observable.
"""
import argparse
import contextlib
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import yaml

from athome.data.hm3d.layout import current_layout
from athome.data.hm3d.floors import component_environment, on_floor
from athome.data.hm3d.semantic_mesh import instance_aabbs
from athome.execution.visit import VisitConfig
from athome.navigation import NavigationConfig, NavigationPlanner
from athome.navigation.grid import GridMap
from athome.scene_graph.query import SceneGraph, floor_references
from athome.schemas import Pose2D
from athome.search import SearchSession
from athome.symbolic import GroundTruthObject, NavmeshSurface, SymbolicEnvironment, TimeModel
from athome.symbolic.habitat_observer import (
    CameraSpec, HabitatSemanticObserver, VisibilitySpec)
from athome.symbolic.wall_los import WallGeometry, WallLosObserver, WallLosSpec
from athome.training.findability import findable_from
from athome.data.hm3d.semantic_mesh import instance_triangles, read_color_table
from athome.data.hm3d.target_masking import normalize_tag

# Reuse the existing provenance, graph-reference and grid validation gates.
from build_component_masked_graphs import main as validate_masked
from check_component_navigation import main as validate_navigation, read, sha, require

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = current_layout()
BASE = LAYOUT.scene_dir
GRID = LAYOUT.grid_dir
ANNOTATIONS = LAYOUT.annotations
COMPONENT_REPORT = LAYOUT.room_report


def semantic_id(object_id):
    return int(object_id.rsplit("_", 1)[1])


def run_episode(graph, grid, start, world, target, offset, max_steps,
                observer, floor_z, heading_count, coverage=False, goal_clearance=0.0,
                goal_max_offset=None, time_budget=None, commit_to_room=False):
    navigation = NavigationPlanner(grid, NavigationConfig(
        goal_offset=offset, max_goals_per_location=1, start_snap_distance=0.0,
        goal_clearance=goal_clearance, goal_max_offset=goal_max_offset))
    budget_s, time_model = time_budget or (None, None)
    env = SymbolicEnvironment(grid, start, world, observer, floor_z, heading_count, time_model)
    session = SearchSession(graph, navigation, [target], max_steps=max_steps,
                            observed_object_ids=env.id_map.get if coverage else None,
                            time_budget_s=budget_s,
                            elapsed_s=env.elapsed_s if time_model else None,
                            commit_to_room=commit_to_room)
    # Findable start (athome.training.findability): the rollout assumption of
    # the proposal reward holds only for these.
    findable_via = findable_from(navigation, graph.locations, start, observer,
                                 [o.semantic_id for o in world if o.category == target])
    steps = []
    no_candidates = sorted(lid for lid in graph.locations
                           if not navigation.has_candidates(lid))
    while True:
        decision = session.next_decision(env.pose)
        if decision is None:
            break
        outcome = env.visit(decision)
        record = session.report(decision, outcome)
        steps.append({
            "step": decision.step, "stage": decision.stage.value,
            "room_id": decision.room_id, "location_id": decision.location_id,
            "path_cost_m": decision.cost, "goal": asdict(env.pose),
            "path_row_col": [list(c) for c in env.last_path],
            "evaluator_detections": env.last_detections,
            "found_targets": record.found, "visited": sorted(session.visited),
            **({"covered_locations": record.covered} if coverage else {}),
            "policy_fallback": record.policy_fallback,
        })
    result = session.targets[0]
    return {
        "target_status": result.status.value, "detail": result.detail,
        "findable": findable_via is not None, "findable_via": findable_via,
        "session_status": session.status.value,
        "steps": steps, "distance_m": env.distance_m,
        **({"elapsed_s": env.elapsed_s()} if time_model else {}),
        "search_location_count": len(graph.locations),
        "no_candidate_location_ids": no_candidates,
        "visited_count": len(session.visited),
        "found_object_id": (env.id_map[result.found_object.object_id]
                            if result.found_object else None),
        "evaluator_observation_id_map": env.id_map,
    }


def load_surface(name, component_triangles, layout=None):
    """``layout``: scene layout (default: ATHOME_HM3D_LAYOUT)."""
    grid = (layout or LAYOUT).grid_dir
    meta = read(grid / f"{name}.metadata.json")
    require(meta["coordinate_frame"] == "athome_z_up", f"{name}: mesh frame")
    require(meta["source_report_sha256"] == sha((layout or LAYOUT).room_report), f"{name}: stale mesh")
    with np.load(grid / f"{name}.mesh.npz", allow_pickle=False) as saved:
        require(sorted(saved["original_triangle_ids"].tolist()) == sorted(component_triangles),
                f"{name}: component triangle mismatch")
        vertices, faces = saved["vertices"].copy(), saved["faces"].copy()
    z = vertices[faces.astype(int)][..., 2]
    require(np.allclose([z.min(), z.max()], meta["navmesh_z_range_m"], atol=1e-6),
            f"{name}: mesh Z range differs from metadata")
    return NavmeshSurface(vertices, faces)


def wall_triangles(semantic_glb, semantic_txt, categories):
    """All semantic-mesh triangles of the occluder categories."""
    wanted = {normalize_tag(c) for c in categories}
    triangles, _ = instance_triangles(semantic_glb, semantic_txt)
    walls = {sid for sid, cat, _ in read_color_table(semantic_txt).values()
             if normalize_tag(cat) in wanted}
    return [t for sid in sorted(walls & set(triangles)) for t in triangles[sid]]


def wall_geometry(name, spec, walls, layout=None):
    """Occluders of one component: the walls in its height band. The same for
    every target of the component, so callers may build it once."""
    meta = read((layout or LAYOUT).grid_dir / f"{name}.metadata.json")
    z_lo, z_hi = meta["navmesh_z_range_m"]
    band = (z_lo + spec.band_above_floor_m[0], z_hi + spec.band_above_floor_m[1])
    return WallGeometry(walls, band), band


def wall_observer(name, instances, spec, walls, layout=None, geometry=None):
    """Wall geometry in the component's height band + observer.
    ``geometry``: a prebuilt ``wall_geometry`` result of this component."""
    geometry, band = geometry or wall_geometry(name, spec, walls, layout)
    return WallLosObserver(geometry, instances, spec), {
        "z_band_m": band, "wall_pieces": len(geometry.geoms)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-config", type=Path,
                        default=ROOT / "configs/data/symbolic_observation.json")
    parser.add_argument("--observation-model", choices=["wall_los_2d", "habitat_render"],
                        help="기본값은 설정 파일의 default_model")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--include-absent", action="store_true",
                        help="영역 내 GT가 없는 음성 에피소드도 실행")
    parser.add_argument("--workspace-locations-only", action="store_true",
                        help="Workspace만 Search Location으로 사용(제안서 4장 문구)")
    parser.add_argument("--commit-to-room", action="store_true",
                        help="이전 방식: 고른 방의 위치를 다 돌 때까지 방을 유지(비교용)")
    parser.add_argument("--time-budget", action="store_true",
                        help="로봇 설정(search.*)의 시간 예산과 속도로 평가(실제 로봇과 같은 종료 조건)")
    parser.add_argument("--coverage", action="store_true",
                        help="방문 중 관측된 물체가 정의하는 Search Location을 Visited로 처리")
    parser.add_argument("--output", type=Path, required=True,
                        help="새 결과 JSON 경로; 기존 파일 덮어쓰기 금지")
    args = parser.parse_args()
    require(args.max_steps > 0, "max-steps는 양수여야 합니다.")
    require(not args.output.exists(), f"이미 존재하는 출력: {args.output}")
    require((BASE / "component_masked_graphs.review/manifest.review.json").is_file(),
            "영역별 masked graph를 먼저 생성하세요.")
    validate_masked()  # Existing artifacts must exactly match their sources.
    validate_navigation()  # Recorded start, offsets and candidates must agree.

    time_budget = None
    if args.time_budget:
        require(LAYOUT.robot is not None, "시간 예산 평가에는 layout의 robot 설정이 필요합니다.")
        search_cfg = (yaml.safe_load(LAYOUT.robot.read_text(encoding="utf-8")).get("search") or {})
        missing = [k for k in ("time_budget_s", "nominal_speed_mps", "rotation_speed_radps")
                   if search_cfg.get(k) is None]
        require(not missing, f"로봇 설정 search 값이 비어 있습니다(확정 필요): {missing}")
        time_budget = (float(search_cfg["time_budget_s"]), TimeModel(
            float(search_cfg["nominal_speed_mps"]), float(search_cfg["rotation_speed_radps"]),
            VisitConfig().observation_window))
    obs_cfg = read(args.observation_config)
    model = args.observation_model or obs_cfg["default_model"]
    heading_count = VisitConfig().heading_count  # same as the robot executor

    nav_path = BASE / "component_navigation.review.json"
    nav = read(nav_path)
    components = {item["component"]: item for item in nav["components"]}
    triangles = {c["component"]: c["triangle_ids"] for c in read(COMPONENT_REPORT)["components"]}
    config_path = LAYOUT.targets
    catalog_path = LAYOUT.catalog
    targets = read(config_path)["target_categories"]
    catalog = {item["target_category"]: item["instances"]
               for item in read(catalog_path)["targets"]}
    annotation = read(ANNOTATIONS)
    scene_path = Path(annotation["scene_path"])
    semantic_glb = scene_path.with_name(scene_path.name.replace(".basis.glb", ".semantic.glb"))
    semantic_txt = semantic_glb.with_suffix(".txt")
    mesh_boxes, mesh_stats = instance_aabbs(semantic_glb, semantic_txt)
    require(mesh_stats["mixed_color"] == 0, "semantic mesh에 여러 ID가 섞인 삼각형이 있습니다.")
    floor_plan = read(LAYOUT.floors)
    report_components = {c["component"]: c for c in read(COMPONENT_REPORT)["components"]}
    report = {
        "status": "development_symbolic_review_only", "policy": "min_cost",
        "observation_model": model,
        "observation_config": obs_cfg[model],
        "heading_count": heading_count,
        "observation_during_navigation": False,
        "observation_time_unit": "simulation_tick_not_seconds",
        "occlusion_checked": "walls_2d" if model == "wall_los_2d" else "rendered_mesh",
        "vertical_visibility_checked": model == "habitat_render",
        "field_of_view_checked": model == "habitat_render", "detector_error_modeled": False,
        "physical_robot_feasibility_verified": False,
        "world_policy": "target_category_instances_on_component_floor",
        "floor_reference_policy": "fixed_unmasked_component_semantic_floor_median",
        "camera_floor_z_policy": "component_navmesh_barycentric_height_at_goal",
        "algorithm": "dijkstra_8_connected_no_corner_cutting",
        "max_steps": args.max_steps, "include_absent": args.include_absent,
        "sft_samples_generated": False, "episodes": [],
        "source_sha256": {
            "navigation": sha(nav_path), "catalog": sha(catalog_path),
            "config": sha(config_path), "observation_config": sha(args.observation_config),
            "masked_manifest": sha(BASE / "component_masked_graphs.review/manifest.review.json"),
            "semantic_glb": sha(semantic_glb), "semantic_txt": sha(semantic_txt),
        },
        "semantic_mesh_stats": mesh_stats,
    }
    with contextlib.ExitStack() as stack:
        renderer = walls = None
        if model == "habitat_render":
            cam, vis = obs_cfg[model]["camera"], obs_cfg[model]["visibility"]
            renderer = stack.enter_context(HabitatSemanticObserver(
                annotation["scene_path"], annotation["config_path"],
                CameraSpec(float(cam["height_m"]), float(cam["hfov_deg"]),
                           int(cam["resolution_wh"][0]), int(cam["resolution_wh"][1])),
                VisibilitySpec(float(vis["max_range_m"]), int(vis["min_visible_pixels"]))))
        else:
            c = obs_cfg[model]
            wall_spec = WallLosSpec(float(c["range_m"]), tuple(c["occluder_categories"]),
                                    tuple(c["band_above_floor_m"]), float(c["contact_tolerance_m"]))
            walls = wall_triangles(semantic_glb, semantic_txt, wall_spec.occluder_categories)
            report["wall_triangles"] = len(walls)
        for name in LAYOUT.component_names():
            cfg = components[name]
            unmasked_path = BASE / "component_graphs.review" / f"{name}.workspace_graph.review.json"
            unmasked = read(unmasked_path)
            scope = {obj["object_id"]: obj for obj in unmasked["objects"]}
            floors = floor_references(unmasked)
            require(set(floors) == {r["room_id"] for r in unmasked["rooms"]},
                    f"{name}: 바닥 높이 근거가 없는 Room이 있습니다.")
            with np.load(GRID / f"{name}.grid.npz", allow_pickle=False) as saved:
                grid = GridMap(saved["free"].copy(), tuple(saved["origin_xy_m"]),
                               float(saved["resolution_m"].item()))
            surface = load_surface(name, triangles[name])
            start = Pose2D(*grid.to_xy(tuple(cfg["start_row_col"])), 0.0)

            comp_env = component_environment(floor_plan, report_components[name],
                                             {r["room_id"] for r in unmasked["rooms"]})
            floor_instances = {t: [o for o in catalog[t] if on_floor(comp_env, o)]
                               for t in targets}
            if renderer is not None:
                observer = renderer
                # Validate rendering against the semantic mesh itself at the start cell.
                sz = surface.height_at(start.x, start.y)
                fraction = observer.validate_projection(
                    [(start.x, start.y, sz, k * math.pi / 2) for k in range(4)], mesh_boxes)
                report.setdefault("projection_check", {})[name] = {
                    "poses": "start_cell_4_headings", "reference": "semantic_mesh_instance_aabb",
                    "labeled_pixels_in_own_bbox": fraction, "bbox_tolerance_m": 0.05}
            else:
                boxes = {semantic_id(o["object_id"]): (o["bbox"]["min"], o["bbox"]["max"])
                         for t in targets for o in floor_instances[t]}
                if args.coverage:  # objects that define Search Locations are observable too
                    boxes.update({semantic_id(o["object_id"]): (o["bbox"]["min"], o["bbox"]["max"])
                                  for o in scope.values()})
                observer, info = wall_observer(name, boxes, wall_spec, walls)
                report.setdefault("wall_grid", {})[name] = info

            for target in targets:
                instances = floor_instances[target]
                if renderer is not None:
                    for obj in instances:
                        require(renderer.instance_ids.get(semantic_id(obj["object_id"]))
                                == obj["object_id"], f"{obj['object_id']}: semantic ID 불일치")
                gt = [obj for obj in instances if obj["object_id"] in scope]
                for obj in gt:
                    source = scope[obj["object_id"]]
                    require(obj["bbox"] == source["bbox"] and obj["room_id"] == source["room_id"],
                            f"{target}: GT 기하 또는 Room 불일치")
                graph_path = BASE / "component_masked_graphs.review" / f"{name}.{target}.workspace_graph.json"
                masked = read(graph_path)
                require(not {obj["object_id"] for obj in gt}
                        & {obj["object_id"] for obj in masked["objects"]}, "GT 누출")
                graph = SceneGraph(masked, room_floor_z={
                    r["room_id"]: floors[r["room_id"]] for r in masked["rooms"]},
                    standalone_locations=not args.workspace_locations_only,
                    excluded_categories=LAYOUT.location_exclusions())
                require(not graph.known_locations(target), "Masked target이 Known으로 남아 있습니다.")
                episode = {
                    "component": name, "target": target,
                    "target_present": bool(gt), "gt_count": len(gt),
                    "planner_graph_sha256": sha(graph_path),
                    "unmasked_graph_sha256": sha(unmasked_path),
                    "grid_sha256": cfg["grid_sha256"],
                    "floor_reference_z_m": graph.room_floor_z,
                    "start_row_col": cfg["start_row_col"],
                    "goal_offset_m": cfg["footprint_offset_m"],
                }
                if not gt and not args.include_absent:
                    episode["status"] = "skipped_absent_target"
                else:
                    # The world holds every scene instance of the category (a robot
                    # would detect out-of-component ones too); nothing here reaches
                    # SearchSession, which sees only the masked graph.
                    world = [GroundTruthObject(o["object_id"], target, semantic_id(o["object_id"]),
                                               tuple(o["bbox"]["center"])) for o in instances]
                    if args.coverage:
                        # Location objects of the masked graph (never the target category).
                        masked_objects = {o["object_id"]: o for o in masked["objects"]}
                        defining = sorted({loc.object_id for loc in graph.locations.values()})
                        world += [GroundTruthObject(
                            oid, normalize_tag(masked_objects[oid]["semantic_tag"]), semantic_id(oid),
                            tuple(masked_objects[oid]["bbox"]["center"])) for oid in defining]
                    episode["status"] = "executed"
                    episode.update(run_episode(
                        graph, grid, start, world, target, cfg["footprint_offset_m"],
                        args.max_steps, observer, surface.height_at, heading_count,
                        coverage=args.coverage,
                        goal_clearance=float(cfg.get("goal_clearance_m", 0.0)),
                        goal_max_offset=cfg.get("goal_max_offset_m"), time_budget=time_budget,
                        commit_to_room=args.commit_to_room))
                    found = episode["found_object_id"]
                    episode["found_object_in_component"] = (found in scope) if found else None
                report["episodes"].append(episode)
                print(f"{name} / {target}: {episode.get('target_status', episode['status'])}")
    report["room_selection"] = "commit_to_room" if args.commit_to_room else "replan_each_visit"
    if LAYOUT.search_locations is not None:
        report["search_location_policy"] = {"path": str(LAYOUT.search_locations),
                                            "sha256": sha(LAYOUT.search_locations)}
    if args.workspace_locations_only or args.coverage:  # absent for the default policy
        report["location_policy"] = {"standalone_locations": not args.workspace_locations_only,
                                     "coverage_update": args.coverage}
    report["summary"] = {
        "executed": sum(e["status"] == "executed" for e in report["episodes"]),
        "skipped_absent": sum(e["status"] == "skipped_absent_target" for e in report["episodes"]),
        "found": sum(e.get("target_status") == "found" for e in report["episodes"]),
        "findable": sum(bool(e.get("findable")) for e in report["episodes"]),
        "found_of_findable": sum(bool(e.get("findable")) and e.get("target_status") == "found"
                                 for e in report["episodes"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print("관측 모델:", model, report.get("projection_check") or report.get("wall_grid"))
    print("요약:", report["summary"])
    print("저장:", args.output.resolve())
    print("개발용 결과입니다. 학습 정답이나 실제 로봇 검증 결과가 아닙니다.")


if __name__ == "__main__":
    main()
