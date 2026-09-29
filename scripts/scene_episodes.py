"""Search problems of one scene layout (ATHOME_HM3D_LAYOUT), shared by
start-state export, policy evaluation and GRPO rollouts.

A problem is one navigable component x one target category with a target
instance in the component: the masked planner graph, the grid and NavMesh
surface, the observation model with every instance of the category on the
floor, and the GT used only for verification and metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np

from shapely.geometry import Polygon

from athome.data.hm3d.floors import component_environment, on_floor
from athome.data.hm3d.layout import repo_path
from athome.data.hm3d.semantic_mesh import instance_triangles
from athome.data.hm3d.target_masking import normalize_tag
from athome.search.coverage import RoomCoverage, floor_points, line_of_sight
from athome.execution.visit import VisitConfig
from athome.navigation import NavigationConfig, NavigationPlanner
from athome.navigation.grid import GridMap
from athome.scene_graph.query import SceneGraph, floor_references
from athome.symbolic import GroundTruthObject
from athome.symbolic.wall_los import WallLosSpec

import run_symbolic_review as runner


@dataclass
class Problem:
    component: str
    target: str
    graph: SceneGraph                 # masked: the planner's input
    grid: GridMap
    surface: object
    observer: object
    world: List[GroundTruthObject]    # every instance of the category on the floor
    target_ids: List[int]
    gt: List[dict]                    # catalog instances inside the component
    gt_workspaces: List[str]
    gt_rooms: List[str]
    nav_config: NavigationConfig
    heading_count: int
    room_points: Dict[str, np.ndarray]   # floor samples per room (observed-fraction tracking)
    sees: object

    def coverage(self) -> RoomCoverage:
        """Fresh observed-fraction tracker for one episode."""
        return RoomCoverage(self.room_points, self.sees)

    def navigation(self) -> NavigationPlanner:
        """Planner with every Search Location's goal candidates."""
        navigation = NavigationPlanner(self.grid, self.nav_config)
        for lid, loc in self.graph.locations.items():
            navigation.add_location(lid, loc.bbox_min, loc.bbox_max)
        return navigation


class SceneProblems:
    """``layout``: a scene layout (athome.data.hm3d.layout.load_layout); the
    default is ATHOME_HM3D_LAYOUT, so several scenes can be used in one process."""

    def __init__(self, targets: Optional[List[str]] = None, layout=None):
        layout = layout or runner.LAYOUT
        self.layout = layout
        obs = runner.read(runner.ROOT / "configs/data/symbolic_observation.json")["wall_los_2d"]
        self.observation_config = obs
        self._spec = WallLosSpec(float(obs["range_m"]), tuple(obs["occluder_categories"]),
                                 tuple(obs["band_above_floor_m"]), float(obs["contact_tolerance_m"]))
        scene = repo_path(runner.read(layout.annotations)["scene_path"])
        glb = scene.with_name(scene.name.replace(".basis.glb", ".semantic.glb"))
        self._walls = runner.wall_triangles(glb, glb.with_suffix(".txt"), self._spec.occluder_categories)
        # Room floors from the semantic mesh floor instances (exact floor shape).
        triangles, _ = instance_triangles(glb, glb.with_suffix(".txt"))
        self._floor_polygons = {}
        for obj in runner.read(layout.graph)["objects"]:
            if normalize_tag(obj["semantic_tag"]) != "floor":
                continue
            tris = triangles.get(runner.semantic_id(obj["object_id"]), [])
            polys = [Polygon(t[:, :2]) for t in tris]
            self._floor_polygons[obj["object_id"]] = (obj, [p for p in polys if p.is_valid and p.area > 1e-6])
        self._nav = {c["component"]: c for c in runner.read(layout.navigation)["components"]}
        self._report = {c["component"]: c for c in runner.read(layout.room_report)["components"]}
        self._floor_plan = runner.read(layout.floors)
        self.targets = targets or runner.read(layout.targets)["target_categories"]
        self._catalog = {t["target_category"]: t["instances"] for t in runner.read(layout.catalog)["targets"]}

    def problems(self) -> Iterator[Problem]:
        layout, heading_count = self.layout, VisitConfig().heading_count
        for name in layout.component_names():
            cfg = self._nav[name]
            unmasked = runner.read(layout.scene_dir / "component_graphs.review" / f"{name}.workspace_graph.review.json")
            scope = {o["object_id"] for o in unmasked["objects"]}
            floor_z = floor_references(unmasked)
            env = component_environment(self._floor_plan, self._report[name],
                                        {r["room_id"] for r in unmasked["rooms"]})
            with np.load(layout.grid_dir / f"{name}.grid.npz", allow_pickle=False) as saved:
                grid = GridMap(saved["free"].copy(), tuple(saved["origin_xy_m"]),
                               float(saved["resolution_m"].item()))
            surface = runner.load_surface(name, self._report[name]["triangle_ids"], layout)
            nav_config = NavigationConfig(
                goal_offset=cfg["footprint_offset_m"], max_goals_per_location=1,
                goal_clearance=float(cfg.get("goal_clearance_m", 0.0)),
                goal_max_offset=cfg.get("goal_max_offset_m"), start_snap_distance=0.0)
            for target in self.targets:
                gt = [o for o in self._catalog[target] if o["object_id"] in scope]
                if not gt:
                    continue
                world_objs = [o for o in self._catalog[target] if on_floor(env, o)]
                boxes = {runner.semantic_id(o["object_id"]): (o["bbox"]["min"], o["bbox"]["max"])
                         for o in world_objs}
                observer, _ = runner.wall_observer(name, boxes, self._spec, self._walls, layout)
                room_points = self._room_points(env, {r["room_id"] for r in unmasked["rooms"]})
                masked = runner.read(layout.scene_dir / "component_masked_graphs.review"
                                     / f"{name}.{target}.workspace_graph.json")
                graph = SceneGraph(masked, room_floor_z={r["room_id"]: floor_z[r["room_id"]]
                                                         for r in masked["rooms"]},
                                   excluded_categories=layout.location_exclusions())
                yield Problem(
                    component=name, target=target, graph=graph, grid=grid, surface=surface,
                    observer=observer,
                    world=[GroundTruthObject(o["object_id"], target, runner.semantic_id(o["object_id"]),
                                             tuple(o["bbox"]["center"])) for o in world_objs],
                    target_ids=list(boxes), gt=gt,
                    gt_workspaces=[o["unmasked_graph_anchor"]["id"] for o in gt
                                   if o["unmasked_graph_anchor"]["type"] == "workspace"],
                    gt_rooms=sorted({o["room_id"] for o in gt}),
                    nav_config=nav_config, heading_count=heading_count,
                    room_points=room_points, sees=line_of_sight(observer.walls, self._spec.range_m))

    def _room_points(self, env, room_ids) -> Dict[str, np.ndarray]:
        """Floor samples of each room on this floor level (partial rooms: this level's floors)."""
        polygons = {}
        for obj, polys in self._floor_polygons.values():
            if obj["room_id"] in room_ids and on_floor(env, obj):
                polygons.setdefault(obj["room_id"], []).extend(polys)
        return {rid: floor_points(polys) for rid, polys in polygons.items()}

    def by_key(self) -> Dict[tuple, Problem]:
        return {(p.component, p.target): p for p in self.problems()}
