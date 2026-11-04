"""Second look at the visited locations from another place, and 6 vs 8
headings, with the detections measured on the robot (ViewpointDetectionModel,
configs/perception/detection_412.json: a miss is fixed per viewpoint, false
detections happen), on the held-out start states in the symbolic environment.

A false detection ends the search as a failure. Another goal pose of a
location counts as another viewpoint (an assumption: the 412 viewpoints were
separate stops of the robot). The turn to centre a point (VisitConfig.relook_max)
is not modelled: the observer has no off-centre candidates.

    python scripts/compare_second_look.py --scenes val \
        --start-states 'outputs/eval_v5/*/start_states.jsonl' --planners greedy cooccurrence \
        --output outputs/second_look/val.jsonl
"""
import argparse
import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from athome.execution import headings as H
from athome.schemas import Pose2D
from athome.search import SearchSession
from athome.search.baselines import CooccurrencePolicy
from athome.search.policy import MinCostPolicy, ShuffledPolicy, Stage
from athome.symbolic.environment import FALSE_DETECTION, SymbolicEnvironment, ViewpointDetectionModel
from athome.symbolic.wall_los import SectorWallLosObserver

from compare_graph_representations import holdout_states, placement_stats
from compare_observation_headings import HFOV, modelled_time
from train_grpo import ROOT, Scenes

CONDITIONS = {"front6": (6, 0.0), "front8": (8, 0.0), "front6+second": (6, 1.0), "front8+second": (8, 1.0)}


def run(p, start, graph, navigation, policy, condition, coverage, max_steps, detection_cfg, scene_id):
    count, separation = CONDITIONS[condition]
    detection = ViewpointDetectionModel(detection_cfg["mean"], detection_cfg["rho"],
                                        tuple(detection_cfg["false_positive_rates"]), scene_id,
                                        false_labels=(p.target,))
    observer = SectorWallLosObserver(p.observer.walls, p.observer.instances, p.observer.spec, HFOV)
    env = SymbolicEnvironment(p.grid, start, p.world, observer, p.surface.height_at, p.heading_count,
                              verify_path=False, detection=detection)
    session = SearchSession(graph, navigation, [p.target], policy=policy, max_steps=max_steps, coverage=coverage,
                            recheck_separation_m=separation)
    second = 0
    found_step = None
    while True:
        decision = session.next_decision(env.pose)
        if decision is None:
            break
        second += decision.stage == Stage.RECHECK
        outcome = env.visit(decision, headings=H.front_first(decision.goals[0], count), stop=session.all_found_in)
        record = session.report(decision, outcome)
        if record.found:
            found_step = decision.stage.value
    target = session.targets[0]
    false_found = (target.status.value == "found"
                   and env.id_map[target.found_object.object_id].startswith(FALSE_DETECTION))
    found = target.status.value == "found" and not false_found
    return {"found": found, "false_found": false_found, "at_second": found_step == Stage.RECHECK.value,
            "second_visits": second, "visits": env.visits, "distance_m": round(env.distance_m, 3),
            "time_s": round(modelled_time(env.distance_m, env.rotations, env.turn_rad, env.headings_observed), 1)}


_STATE = {}


def _init(pattern, stats, detection_cfg):
    _STATE.update(scenes=Scenes(pattern), stats=stats, detection=detection_cfg)


def episode(args, state):
    p = _STATE["scenes"].problems(state["scene_id"])[(state["component"], state["target"])]
    start = Pose2D(*p.grid.to_xy(tuple(state["start_row_col"])), 0.0)
    row = {"state_id": state["state_id"], "scene_id": state["scene_id"], "target": state["target"]}
    for name in args.planners:
        for condition in args.conditions:
            inner = (MinCostPolicy() if name == "greedy"
                     else CooccurrencePolicy(_STATE["stats"], step_cost=args.step_cost))
            policy = ShuffledPolicy(inner, f"order:{state['state_id']}")
            coverage = p.coverage() if name == "cooccurrence" else None
            row[f"{condition}/{name}"] = run(p, start, p.graph, p.navigation(), policy, condition, coverage,
                                             args.max_steps, _STATE["detection"], state["scene_id"])
    return row


def _scene(args, states):
    return [episode(args, s) for s in states]


def summarize(rows, base="front6"):
    """Means and the difference from ``base`` with a 95% interval, resampling buildings."""
    out = {}
    keys = sorted({k for r in rows for k in r if "/" in k})
    scenes = sorted({r["scene_id"] for r in rows})
    index = {s: [i for i, r in enumerate(rows) if r["scene_id"] == s] for s in scenes}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([index[scenes[j]] for j in rng.integers(0, len(scenes), len(scenes))]).astype(int)
             for _ in range(2000)]
    for key in keys:
        condition, planner = key.split("/")
        col = {m: np.array([float(r[key][m]) for r in rows])
               for m in ("found", "false_found", "at_second", "second_visits", "visits", "time_s")}
        entry = {m: round(float(v.mean()), 4) for m, v in col.items()}
        missed = ~(col["found"].astype(bool) | col["false_found"].astype(bool))
        entry["not_found"] = round(float(missed.mean()), 4)
        ref = f"{base}/{planner}"
        if key != ref:
            for m in ("found", "false_found", "time_s"):
                d = col[m] - np.array([float(r[ref][m]) for r in rows])
                lo, hi = np.percentile([d[i].mean() for i in draws], [2.5, 97.5])
                entry[f"d_{m}"] = [round(float(d.mean()), 4), round(float(lo), 4), round(float(hi), 4)]
        out[key] = entry
    return {"episodes": len(rows), "buildings": len(scenes), "conditions": out}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--planners", nargs="+", choices=["greedy", "cooccurrence"], default=["cooccurrence"])
    parser.add_argument("--start-states", default="outputs/eval_v5/*/start_states.jsonl")
    parser.add_argument("--scenes", choices=["holdout", "val", "test"], default="holdout")
    parser.add_argument("--detection", type=Path, default=ROOT / "configs/perception/detection_412.json")
    parser.add_argument("--step-cost", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=120)   # room for the second look
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grpo_cfg = yaml.safe_load((ROOT / "configs/training/grpo.yaml").read_text(encoding="utf-8"))
    sft_cfg = yaml.safe_load((ROOT / "configs/training/sft_lora.yaml").read_text(encoding="utf-8"))
    fraction = sft_cfg["validation"]["holdout_scene_fraction"]
    states, split_of = holdout_states(args.start_states, grpo_cfg, fraction, args.scenes)
    if args.limit:
        states = sorted(states, key=lambda s: zlib.crc32(s["state_id"].encode()))[:args.limit]
    stats = placement_stats(grpo_cfg, fraction, split_of) if "cooccurrence" in args.planners else None
    detection_cfg = json.loads(args.detection.read_text(encoding="utf-8"))
    rows = []
    if args.output.exists():
        rows = [json.loads(l) for l in args.output.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = {r["state_id"] for r in rows}
    by_scene = {}
    for s in states:
        if s["state_id"] not in done:
            by_scene.setdefault(s["scene_id"], []).append(s)
    print(f"에피소드 {len(states)}개 (남은 {sum(map(len, by_scene.values()))}), 건물 {len(by_scene)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as f, ProcessPoolExecutor(
            args.workers, initializer=_init,
            initargs=(grpo_cfg["layouts"], stats, detection_cfg)) as pool:
        for k, result in enumerate(pool.map(_scene, [args] * len(by_scene), by_scene.values())):
            for row in result:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            f.flush()
            print(f"건물 {k + 1}/{len(by_scene)}", flush=True)
    summary = summarize(rows)
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
