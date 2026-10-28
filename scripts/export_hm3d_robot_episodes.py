"""HM3D evaluation episodes as robot setups, to run the ROS search stack on
the problems of the symbolic evaluation (scripts/run_hm3d_episodes.py).

    conda run -n athome-data python scripts/export_hm3d_robot_episodes.py \
        --scene-id GPyDUnjwZQy --limit 5 --out outputs/hm3d_robot

One directory per problem (component x target), <out>/<scene>/<component>.<target>/:
  map.pgm, map.yaml   the problem's grid as a map image: free = NavMesh cells
                      of the robot's inscribed radius (map.inflated_m)
  graph.json          masked planner graph (targets hidden) with room floor heights
  robot.yaml          dataset rules: costmap radius = inscribed radius, goal
                      clearance for rotation (must reproduce the symbolic run)
  robot_r037.yaml     the robot's costmap radius 0.37 (motion module), no clearance
  visibility.json     target instances visible from each goal candidate cell of
                      both setups (training observation model: wall line of sight)
  episodes.jsonl      per start state: start pose and the symbolic nearest-first
                      (MinCost) run under each setup

The robot setups are derived from the scene layout's robot config and loaded
back with athome.config, so the grid and goal rules checked here are the ones
the search server builds.
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt

from athome.config import load_robot_config, robot_geometry
from athome.data.hm3d.layout import load_layout
from athome.navigation import NavigationConfig, NavigationPlanner, load_map_server
from athome.navigation.grid import GridMap
from athome.schemas import Pose2D
from athome.search import SearchSession
from athome.search.policy import MinCostPolicy, ShuffledPolicy
from athome.symbolic.environment import SymbolicEnvironment

from scene_episodes import SceneProblems

ROOT = Path(__file__).resolve().parents[1]
SETUPS = {"robot": None, "robot_r037": 0.37}       # file stem -> navigation.costmap_robot_radius


def write_map(path: Path, grid: GridMap) -> Path:
    """Free cells white (254), the rest black (0), row 0 at the top (map_server)."""
    image = np.where(grid.free, 254, 0).astype(np.uint8)[::-1]
    Image.fromarray(image).save(path.with_suffix(".pgm"))
    path.with_suffix(".yaml").write_text(yaml.safe_dump({
        "image": path.with_suffix(".pgm").name, "mode": "trinary",
        "resolution": float(grid.resolution), "origin": [float(grid.origin[0]), float(grid.origin[1]), 0.0],
        "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.25}, sort_keys=False))
    return path.with_suffix(".yaml")


def robot_config(base: dict, layout, map_yaml: Path, graph: Path, radius) -> dict:
    """The layout's robot config on this problem: its footprint and goal rules,
    the exported map and graph, the internal planner and the rule baseline."""
    cfg = json.loads(json.dumps(base))
    inscribed = robot_geometry(base["robot"])[1]
    cfg["map"] = {"yaml": str(map_yaml), "unknown_as_occupied": True, "inflated_m": inscribed}
    nav = cfg["navigation"]
    for key in ("costmap_robot_radius", "costmap_inflation_radius", "costmap_cost_scaling_factor"):
        nav.pop(key, None)
    if radius is not None:
        nav["costmap_robot_radius"] = radius
    nav["path_planner"] = "none"        # Nav2 inflates from obstacles, not from this map
    scene_graph = cfg["scene_graph"]
    scene_graph.pop("wall_band_above_floor_m", None)
    scene_graph["path"] = str(graph)
    scene_graph["search_locations"] = (str(ROOT / layout.search_locations)
                                       if layout.search_locations else None)
    for key in ("scene_scope", "target_categories"):
        scene_graph[key] = str((ROOT / layout.robot).parent.joinpath(scene_graph[key]).resolve())
    cfg["planner"] = {"type": "min_cost"}
    # One frame per target: the symbolic run counts the first detection.
    cfg["matcher"] = {"type": "label", "min_confidence": 0.4, "confirmations": 1}
    cfg.pop("command_parser", None)
    cfg["search"]["time_budget_s"] = None
    return cfg


def snap_start(grid: GridMap, start: Pose2D, distance: float):
    """Nearest free cell centre within ``distance`` (the robot's start snap), or None."""
    cell = grid.to_cell(start.x, start.y)
    if grid.is_free(cell):
        return start
    dist, (rows, cols) = distance_transform_edt(~grid.free, return_indices=True)
    if dist[cell] * grid.resolution > distance:
        return None
    return Pose2D(*grid.to_xy((int(rows[cell]), int(cols[cell]))), start.yaw)


def symbolic_run(problem, grid, navigation, start, state_id, max_steps):
    """evaluate_student.py's MinCost episode on this grid and planner."""
    env = SymbolicEnvironment(grid, start, problem.world, problem.observer, problem.surface.height_at,
                              problem.heading_count, verify_path=False)
    policy = ShuffledPolicy(MinCostPolicy(), f"order:{state_id}")
    session = SearchSession(problem.graph, navigation, [problem.target], policy=policy, max_steps=max_steps)
    targets = {o.object_id for o in problem.world}
    visits = []
    while True:
        decision = session.next_decision(env.pose)
        if decision is None:
            break
        outcome = env.visit(decision)
        session.report(decision, outcome)
        goal = decision.goals[0]
        visits.append({"stage": decision.stage.value, "room_id": decision.room_id,
                       "location_id": decision.location_id, "cost_m": decision.cost,
                       "goal": [goal.x, goal.y, goal.yaw],
                       "detected": sorted({d["object_id"] for d in env.last_detections} & targets)})
    return {"found": session.targets[0].status.value == "found", "status": session.status.value,
            "distance_m": env.distance_m, "visits": env.visits, "trajectory": visits}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--layouts", default="outputs/hm3d_v5/{scene_id}.layout.json")
    parser.add_argument("--start-states", default="outputs/eval_v5_everygoal/{scene_id}/start_states.jsonl")
    parser.add_argument("--states", nargs="*", default=[], help="state_id (기본: 전부)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=60, help="evaluate_student.py 기본값")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/hm3d_robot")
    args = parser.parse_args()

    layout = load_layout(ROOT / args.layouts.format(scene_id=args.scene_id))
    base = yaml.safe_load((ROOT / layout.robot).read_text())
    states = [json.loads(l) for p in sorted(glob.glob(str(ROOT / args.start_states.format(scene_id=args.scene_id))))
              for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    states = [s for s in states if s["oracle_distance_m"] is not None]
    if args.states:
        states = [s for s in states if s["state_id"] in set(args.states)]
    if args.limit:
        states = states[:args.limit]
    by_problem = {}
    for s in states:
        by_problem.setdefault((s["component"], s["target"]), []).append(s)
    problems = SceneProblems(layout=layout).by_key()

    for (component, target), group in by_problem.items():
        p = problems[(component, target)]
        out = args.out / args.scene_id / f"{component}.{target}"
        out.mkdir(parents=True, exist_ok=True)
        map_yaml = write_map(out / "map", p.grid)
        graph = json.loads((layout.scene_dir / "component_masked_graphs.review"
                            / f"{component}.{target}.workspace_graph.json").read_text(encoding="utf-8"))
        for room in graph["rooms"]:
            room["floor_z_m"] = p.graph.room_floor_z[room["room_id"]]
        graph["provenance"] = {**graph.get("provenance", {}), "map_version": None,
                               "source": f"HM3D {args.scene_id} {component} {target} (masked)"}
        setups = {}
        for stem, radius in SETUPS.items():
            path = out / f"{stem}.yaml"
            path.write_text(yaml.safe_dump(robot_config(base, layout, map_yaml, out / "graph.json", radius),
                                           sort_keys=False, allow_unicode=True))
            config = load_robot_config(path)
            graph["provenance"]["map_version"] = config.map_version
            grid = load_map_server(config.map_yaml, config.map_inflation, config.unknown_as_occupied)
            nav = NavigationConfig(goal_offset=config.goal_offset, goal_clearance=config.goal_clearance,
                                   goal_max_offset=config.goal_max_offset, max_goals_per_location=1,
                                   start_snap_distance=0.0)
            setups[stem] = (config, grid, nav)
        (out / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8")

        # Dataset rules must give the symbolic problem's grid and goal rules exactly.
        config, grid, nav = setups["robot"]
        if not (grid.shape == p.grid.shape and np.array_equal(grid.free, p.grid.free)
                and np.allclose(grid.origin, p.grid.origin) and math.isclose(grid.resolution, p.grid.resolution)):
            raise SystemExit(f"{out}: 내보낸 지도에서 만든 격자가 학습 격자와 다름")
        for key in ("goal_offset", "goal_clearance", "goal_max_offset"):
            if not math.isclose(getattr(nav, key), getattr(p.nav_config, key), abs_tol=1e-9):
                raise SystemExit(f"{out}: {key} {getattr(nav, key)} != 학습 {getattr(p.nav_config, key)}")
        planners = {"robot": p.navigation()}
        _, grid37, nav37 = setups["robot_r037"]
        planners["robot_r037"] = NavigationPlanner(grid37, nav37)
        for lid, loc in p.graph.locations.items():
            planners["robot_r037"].add_location(lid, loc.bbox_min, loc.bbox_max)

        # Visible targets from every goal candidate cell (the robot stands only there).
        cells = {}
        for stem, planner in planners.items():
            for candidates in planner._candidates.values():
                for _, pose in candidates:
                    cells[(round(pose.x, 4), round(pose.y, 4))] = None
        ids = {o.semantic_id: o.object_id for o in p.world}
        for x, y in cells:
            evidence = p.observer.observe(x, y, p.surface.height_at(x, y), 0.0)
            cells[(x, y)] = sorted(ids[i] for i in evidence if i in ids)
        (out / "visibility.json").write_text(json.dumps({
            "observation": p.observer.__class__.__name__,
            "objects": {o.object_id: {"label": target, "centroid": list(o.centroid)} for o in p.world},
            "cells": [{"x": x, "y": y, "visible": v} for (x, y), v in cells.items()],
        }), encoding="utf-8")

        with (out / "episodes.jsonl").open("w", encoding="utf-8") as f:
            for s in group:
                start = Pose2D(*p.grid.to_xy(tuple(s["start_row_col"])), 0.0)
                row = {"state_id": s["state_id"], "target": target, "start": [start.x, start.y, start.yaw]}
                for stem, planner in planners.items():
                    begin = snap_start(planner.grid, start, NavigationConfig().start_snap_distance)
                    if begin is None:
                        row[stem] = {"start_not_free": True}
                        continue
                    row[stem] = {"start": [begin.x, begin.y, begin.yaw],
                                 **symbolic_run(p, planner.grid, planner, begin, s["state_id"], args.max_steps)}
                f.write(json.dumps(row) + "\n")
                print(f"{s['state_id']}: " + ", ".join(
                    f"{stem} 방문 {row[stem].get('visits')} 거리 {row[stem].get('distance_m', 0):.2f} "
                    f"찾음 {row[stem].get('found')}" for stem in planners))
        print("저장:", out, f"(가시성 셀 {len(cells)})")


if __name__ == "__main__":
    main()
