"""Habitat-Sim semantic camera for evaluator-side visibility.

An instance counts as visible when enough of its semantic pixels lie within
the sensor range. Occlusion, field of view and camera height follow from
rendering the scene mesh. Poses are athome_z_up; yaw 0 faces +X.

HM3DSem v0.2 semantic GLBs encode instance IDs in vertex colors, so semantic
textures are disabled. ``validate_projection`` checks the result against the
annotation bboxes before use.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    height_m: float
    hfov_deg: float
    width: int
    height: int

    def __post_init__(self):
        if not (self.height_m > 0 and 0 < self.hfov_deg < 180
                and self.width > 0 and self.height > 0):
            raise ValueError("잘못된 카메라 사양")


@dataclass(frozen=True)
class VisibilitySpec:
    max_range_m: float
    min_visible_pixels: int

    def __post_init__(self):
        if not (math.isfinite(self.max_range_m) and self.max_range_m > 0
                and self.min_visible_pixels > 0):
            raise ValueError("잘못된 가시성 기준")


class HabitatSemanticObserver:
    def __init__(self, scene_path, dataset_config_path,
                 camera: CameraSpec, visibility: VisibilitySpec):
        import habitat_sim
        import magnum as mn

        self.camera, self.visibility = camera, visibility
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(Path(scene_path).resolve(strict=True))
        sim_cfg.scene_dataset_config_file = str(Path(dataset_config_path).resolve(strict=True))
        sim_cfg.enable_physics = False
        sim_cfg.load_semantic_mesh = True
        sim_cfg.use_semantic_textures = False
        sensors = []
        for uuid, kind in (("semantic", habitat_sim.SensorType.SEMANTIC),
                           ("depth", habitat_sim.SensorType.DEPTH)):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid, spec.sensor_type = uuid, kind
            spec.resolution = [camera.height, camera.width]
            spec.hfov = mn.Deg(camera.hfov_deg)
            spec.position = mn.Vector3(0.0, camera.height_m, 0.0)
            sensors.append(spec)
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = sensors
        self._habitat = habitat_sim
        self._sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent]))
        # semantic_id -> annotation ID ("pillow_311")
        self.instance_ids: Dict[int, str] = {
            obj.semantic_id: obj.id for obj in self._sim.semantic_scene.objects
            if obj is not None}

    def close(self):
        self._sim.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def render(self, x: float, y: float, floor_z: float, yaw: float):
        state = self._habitat.AgentState()
        state.position = np.array([x, floor_z, -y], dtype=np.float32)  # athome -> habitat
        # Habitat cameras face -Z_h = +Y_athome (yaw pi/2); rotate about the up axis.
        half = (yaw - math.pi / 2) / 2
        state.rotation = np.quaternion(math.cos(half), 0.0, math.sin(half), 0.0)
        self._sim.get_agent(0).set_state(state, infer_sensor_states=False)
        obs = self._sim.get_sensor_observations()
        return np.asarray(obs["semantic"]), np.asarray(obs["depth"])

    def observe(self, x: float, y: float, floor_z: float, yaw: float) -> Dict[int, int]:
        """Visible semantic IDs -> in-range pixel count (threshold applied)."""
        semantic, depth = self.render(x, y, floor_z, yaw)
        in_range = (depth > 0) & (depth <= self.visibility.max_range_m)
        ids, counts = np.unique(semantic[in_range], return_counts=True)
        return {int(i): int(n) for i, n in zip(ids, counts)
                if n >= self.visibility.min_visible_pixels}

    def backproject(self, depth, x, y, floor_z, yaw):
        """Pixel centers to athome_z_up points (pinhole, square pixels)."""
        cam = self.camera
        f = (cam.width / 2) / math.tan(math.radians(cam.hfov_deg) / 2)
        v, u = np.mgrid[0:cam.height, 0:cam.width]
        right = (u + 0.5 - cam.width / 2) / f * depth
        up = -(v + 0.5 - cam.height / 2) / f * depth
        c, s = math.cos(yaw), math.sin(yaw)
        return np.stack([x + depth * c + right * s, y + depth * s - right * c,
                         floor_z + cam.height_m + up], axis=-1)

    def validate_projection(self, poses, bboxes: Mapping[int, Tuple], tol_m=0.05,
                            min_fraction=0.999) -> float:
        """Fraction of labeled pixels whose 3D point lies in its own AABB.

        Checks semantic ID mapping, coordinate transform, yaw and intrinsics
        together. ``bboxes``: semantic_id -> (min_xyz, max_xyz)."""
        inside = total = 0
        for x, y, z, yaw in poses:
            semantic, depth = self.render(x, y, z, yaw)
            points = self.backproject(depth, x, y, z, yaw)
            for sid in np.unique(semantic[depth > 0]):
                if int(sid) not in bboxes:
                    continue
                mask = (semantic == sid) & (depth > 0)
                lo, hi = (np.asarray(b, float) for b in bboxes[int(sid)])
                p = points[mask]
                inside += int(np.all((p >= lo - tol_m) & (p <= hi + tol_m), axis=1).sum())
                total += int(mask.sum())
        if total == 0:
            raise RuntimeError("투영 검증에 쓸 semantic 픽셀이 없습니다.")
        fraction = inside / total
        if fraction < min_fraction:
            raise RuntimeError(f"semantic 투영 불일치: bbox 내부 비율 {fraction:.4f}")
        return fraction
