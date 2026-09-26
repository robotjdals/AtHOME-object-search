"""Load HM3DSem annotations in Habitat's native coordinate frame."""

from pathlib import Path

import habitat_sim
import numpy as np


def _read_bbox(obj):
    """Compute world-axis AABB from the source OBB."""
    obb = obj.obb
    center = np.asarray(obb.center, dtype=float)
    size = np.asarray(obb.sizes, dtype=float)
    quat = np.asarray(obb.rotation, dtype=float)  # x, y, z, w

    if not (
        np.isfinite(center).all()
        and np.isfinite(size).all()
        and np.isfinite(quat).all()
        and (size >= 0).all()
        and (size > 0).any()
    ):
        return None

    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return None

    x, y, z, w = quat / norm
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])

    half_extent = np.abs(rotation) @ (size / 2)
    return {
        "center": center.tolist(),
        "size": (2 * half_extent).tolist(),
        "min": (center - half_extent).tolist(),
        "max": (center + half_extent).tolist(),
    }


def load_scene(scene_path, config_path):
    """Return scene annotations as JSON-compatible Python data."""
    scene_path = Path(scene_path).resolve()
    config_path = Path(config_path).resolve()

    for path in (scene_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_path)
    sim_cfg.scene_dataset_config_file = str(config_path)
    sim_cfg.enable_physics = False
    sim_cfg.load_semantic_mesh = True
    sim_cfg.use_semantic_textures = False

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "semantic"
    sensor.sensor_type = habitat_sim.SensorType.SEMANTIC
    sensor.resolution = [128, 128]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    with habitat_sim.Simulator(cfg) as sim:
        if not sim.pathfinder.is_loaded:
            raise RuntimeError("NavMesh 로딩 실패")

        sim.build_semantic_CC_objects()

        objects = []
        for obj in sim.semantic_scene.objects:
            if obj is None:
                continue
            objects.append({
                "object_id": obj.id,
                "category": (
                    obj.category.name() if obj.category is not None else None
                ),
                "region_id": (
                    obj.region.id if obj.region is not None else None
                ),
                "bbox": _read_bbox(obj),
            })

        regions = [
            {
                "region_id": region.id,
                "object_ids": [
                    obj.id for obj in region.objects if obj is not None
                ],
            }
            for region in sim.semantic_scene.regions
            if region is not None
        ]

        if not objects or not regions:
            raise RuntimeError("객체 또는 Region Annotation 로딩 실패")

        return {
            "schema_version": "0.1",
            "habitat_sim_version": habitat_sim.__version__,
            "scene_path": str(scene_path),
            "config_path": str(config_path),
            "coordinate_frame": "habitat_native",
            "length_unit": "meter",
            "navmesh_loaded": True,
            "objects": objects,
            "regions": regions,
        }
