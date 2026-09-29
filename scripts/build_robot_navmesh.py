"""Recompute a scene's NavMesh for the real robot (proposal 5-2: r_robot).

The HM3D NavMesh is built for a 0.1 m agent. The search system must plan for
the robot that runs it, so the NavMesh is recomputed with the robot's
clearance radius, the value the real robot's A*/Nav2/MPPI use
(``robot.inflation_radius`` of the robot config, Nav2 robot_radius). All
other NavMesh settings (height, step, slope, cell size) are kept from the
dataset NavMesh. This is the standard way Habitat-based robot work models a
specific platform (e.g. HomeRobot/OVMM sets the NavMesh agent radius of
Stretch).
"""
import argparse
import hashlib
import json
from pathlib import Path

import habitat_sim
import yaml

from athome.config import robot_geometry, traversability

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ["agent_height", "agent_max_climb", "agent_max_slope", "cell_height", "cell_size",
            "detail_sample_dist", "detail_sample_max_error", "edge_max_error", "edge_max_len",
            "filter_ledge_spans", "filter_low_hanging_obstacles", "filter_walkable_low_height_spans",
            "region_merge_size", "region_min_size", "verts_per_poly"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="<scene>.basis.glb")
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--navmesh", type=Path, required=True, help="원본 NavMesh(설정 기준)")
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="<name>.navmesh; 옆에 .json 기록")
    args = parser.parse_args()
    record_path = args.output.with_suffix(".json")
    if args.output.exists() or record_path.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    robot = yaml.safe_load(args.robot_config.read_text(encoding="utf-8"))["robot"]
    radius = robot_geometry(robot)[1]  # inscribed radius of the footprint
    limits = traversability(robot)

    source = habitat_sim.PathFinder()
    if not source.load_nav_mesh(str(args.navmesh)):
        raise SystemExit("원본 NavMesh 로딩 실패")
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    for name in SETTINGS:
        setattr(settings, name, getattr(source.nav_mesh_settings, name))
    settings.agent_radius = radius
    if limits is not None:
        # The robot's step and slope limits; the height grid must be fine
        # enough to represent the step (Recast climb = floor(step / cell_height)).
        settings.agent_max_climb, settings.agent_max_slope = limits
        settings.cell_height = min(settings.cell_height, limits[0] / 2)
    settings.include_static_objects = False

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(args.dataset_config)
    sim_cfg.scene_id = str(args.scene)
    sim_cfg.load_semantic_mesh = False
    sim_cfg.create_renderer = False
    agent = habitat_sim.agent.AgentConfiguration(sensor_specifications=[])
    with habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent])) as sim:
        if not sim.recompute_navmesh(sim.pathfinder, settings):
            raise SystemExit("NavMesh 재계산 실패")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not sim.pathfinder.save_nav_mesh(str(args.output)):
            raise SystemExit("NavMesh 저장 실패")
        islands = sim.pathfinder.num_islands
        area = sim.pathfinder.navigable_area

    record = {
        "schema_version": "0.1", "purpose": "robot_navmesh",
        "robot_config": str(args.robot_config), "robot_config_sha256": sha(args.robot_config),
        "agent_radius_m": radius, "agent_radius_source": "inscribed radius of robot.footprint_m (Nav2 INSCRIBED; robot.inflation_radius)",
        "source_navmesh": str(args.navmesh), "source_navmesh_sha256": sha(args.navmesh),
        "settings": {n: getattr(settings, n) for n in SETTINGS},
        "robot_traversability": None if limits is None else {"max_step_m": limits[0], "max_slope_deg": limits[1]},
        "navmesh_sha256": sha(args.output), "islands": islands, "navigable_area_m2": area,
        "habitat_sim_version": habitat_sim.__version__,
    }
    with record_path.open("x", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"반경 {radius} m NavMesh: 섬 {islands}개, 이동 가능 면적 {area:.1f} m² → {args.output}")


if __name__ == "__main__":
    main()
