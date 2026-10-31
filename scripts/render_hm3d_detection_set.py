"""Detection test set from HM3D-Sem Val buildings: robot-camera views with
per-instance ground truth, for choosing the real-time detector and its
confidence threshold on many homes (scripts/evaluate_detectors_hm3d.py);
the 412 frames then only confirm the choice on real images.

    python scripts/render_hm3d_detection_set.py --scenes 12 --views 25 --out outputs/perception_bench/hm3d_set

Camera: the robot's D435 colour camera (1280 x 720, HFOV 70.6 deg,
configs/robot/room412_v2.yaml; 0.77 m above the floor, from the calibration
T_cam_body and T_body_base_link). Views: random navigable points of the
building's NavMesh with random yaw, kept when a target-category instance is
in view (the robot looks at Search Locations). Ground truth per view: every
instance with pixels in view; an instance is a positive when it has at least
1024 px (COCO's 32 x 32 small-object boundary, the training observation
model's habitat_render threshold) within 3 m (D435 recommended range,
symbolic_observation.json), otherwise it is "ignore" (detections on it count
neither way). Each view also carries a command: 3 target categories present
in the building (the task's targets), for per-command vocabularies.
"""
import argparse
import glob
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from athome.scene_graph.query import normalize_category

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, HFOV, CAM_H = 1280, 720, 70.6, 0.77
RANGE_M, MIN_PX = 3.0, 1024


def target_index():
    cats = json.loads((ROOT / "configs/data/target_categories.v5.json").read_text(encoding="utf-8"))["categories"]
    return {normalize_category(raw): c for c, info in cats.items() for raw in info["raw_names"]}


def make_sim(scene_path, config_path):
    import habitat_sim
    import magnum as mn
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.scene_dataset_config_file = config_path
    sim_cfg.enable_physics = False
    sim_cfg.load_semantic_mesh = True
    sim_cfg.use_semantic_textures = False          # HM3DSem v0.2: instance IDs in vertex colours
    sensors = []
    for uuid, kind in (("color", habitat_sim.SensorType.COLOR), ("semantic", habitat_sim.SensorType.SEMANTIC),
                       ("depth", habitat_sim.SensorType.DEPTH)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = uuid, kind
        spec.resolution = [HEIGHT, WIDTH]
        spec.hfov = mn.Deg(HFOV)
        spec.position = mn.Vector3(0.0, CAM_H, 0.0)
        sensors.append(spec)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = sensors
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", type=int, default=12)
    p.add_argument("--views", type=int, default=25)
    p.add_argument("--targets-per-command", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    import habitat_sim

    rng = random.Random(args.seed)
    index = target_index()
    val = sorted(a for a in glob.glob(str(ROOT / "outputs/hm3d_v5/*.annotations.json"))
                 if "/val/" in json.loads(Path(a).read_text())["scene_path"])
    chosen = sorted(rng.sample(val, min(args.scenes, len(val))))
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for apath in chosen:
        ann = json.loads(Path(apath).read_text())
        scene = Path(apath).name.split(".")[0]
        sim = make_sim(ann["scene_path"], ann["config_path"])
        try:
            names = {o.semantic_id: (o.id, normalize_category(o.category.name()))
                     for o in sim.semantic_scene.objects if o is not None}
            present = sorted({index[raw] for _, raw in names.values() if raw in index})
            sim.pathfinder.seed(rng.randrange(2**31))
            kept = tries = 0
            while kept < args.views and tries < args.views * 40:
                tries += 1
                pos = sim.pathfinder.get_random_navigable_point()
                yaw = rng.uniform(-math.pi, math.pi)
                state = habitat_sim.AgentState()
                state.position = pos
                state.rotation = np.quaternion(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
                sim.get_agent(0).set_state(state, infer_sensor_states=False)
                obs = sim.get_sensor_observations()
                sem, depth = np.asarray(obs["semantic"]), np.asarray(obs["depth"])
                instances = []
                for sid in np.unique(sem):
                    if int(sid) not in names:
                        continue
                    m = sem == sid
                    oid, raw = names[int(sid)]
                    in_range = int((m & (depth > 0) & (depth <= RANGE_M)).sum())
                    ys, xs = np.nonzero(m)
                    instances.append(dict(object_id=oid, raw=raw, category=index.get(raw),
                                          positive=bool(index.get(raw)) and in_range >= MIN_PX,
                                          pixels=int(m.sum()), pixels_in_range=in_range,
                                          box=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                                          depth_m=round(float(np.median(depth[m])), 2)))
                if not any(i["positive"] for i in instances):
                    continue
                name = f"{scene}_{kept:03d}"
                Image.fromarray(np.asarray(obs["color"])[..., :3]).save(args.out / "images" / f"{name}.jpg", quality=92)
                command = rng.sample(present, min(args.targets_per_command, len(present)))
                records.append(dict(frame=name, scene=scene, position=[float(v) for v in pos], yaw=yaw,
                                    command=command, instances=instances))
                kept += 1
            print(f"{scene}: {kept} views ({tries} tried), {len(present)} target categories in the building")
        finally:
            sim.close()
    meta = dict(camera=dict(width=WIDTH, height=HEIGHT, hfov_deg=HFOV, height_m=CAM_H), range_m=RANGE_M,
                min_pixels=MIN_PX, seed=args.seed, scenes=[Path(a).name.split(".")[0] for a in chosen])
    (args.out / "ground_truth.json").write_text(json.dumps(dict(meta=meta, views=records), ensure_ascii=False),
                                                encoding="utf-8")
    print(f"{len(records)} views -> {args.out}")


if __name__ == "__main__":
    main()
