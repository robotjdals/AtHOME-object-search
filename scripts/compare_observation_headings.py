"""Observation headings at a search location, compared on the held-out
start states (Val/Test buildings) in the symbolic environment.

  disk6    the training observation model: everything within range and
           line of sight is seen (6 headings, 360 degrees), no early stop
  sector6  6 camera views (HFOV each) from the goal yaw
  front_first6  the same 6 views, the 3 facing the location first (goal
           yaw, right, left, then on round the left: athome.execution.headings.front_first)
  front3   3 views: the goal yaw (facing the location) and +-60 degrees
  cover    the fewest views 60 degrees apart that cover the location's
           footprint from the goal pose (athome.execution.headings.cover)

The sector conditions stop a visit after the heading that confirms the
target (VisitRequest.satisfied, as on the robot). Time model measured on
the Nav2Tidy round trip (team_real.yaml MPPI, kinematic base, 2026-11-04):
driving 0.35 m/s; an in-place rotation goal 2.0 s + angle / 0.35 rad/s
(60 deg: 5.0 s, measured 4.5-5.0 s); 1.0 s observation per heading
(perception ~0.35-0.6 s + the 0.5 s window).

    python scripts/compare_observation_headings.py --scenes val \
        --start-states 'outputs/eval_v5/*/start_states.jsonl' --planners greedy cooccurrence \
        --output outputs/heading_compare/val.jsonl
"""
import argparse
import json
import math
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from athome.execution import headings as H
from athome.schemas import Pose2D
from athome.search import SearchSession
from athome.search.baselines import CooccurrencePolicy
from athome.search.policy import MinCostPolicy, ShuffledPolicy
from athome.symbolic.environment import SymbolicEnvironment
from athome.symbolic.wall_los import SectorWallLosObserver

from compare_graph_representations import holdout_states, placement_stats
from train_grpo import ROOT, Scenes

CONDITIONS = ("disk6", "sector6", "front_first6", "front3", "cover")
HFOV = math.radians(70.6)                # D435 RGB 1280x720 (configs/robot sensors.camera_hfov_deg)
DRIVE_MPS, TURN_S, TURN_RADPS, OBSERVE_S = 0.35, 2.0, 0.35, 1.0


def modelled_time(distance_m, rotations, turn_rad, headings):
    return distance_m / DRIVE_MPS + rotations * TURN_S + turn_rad / TURN_RADPS + headings * OBSERVE_S


def run(p, start, graph, navigation, policy, condition, coverage, max_steps):
    sector = condition != "disk6"
    observer = (SectorWallLosObserver(p.observer.walls, p.observer.instances, p.observer.spec, HFOV)
                if sector else p.observer)
    env = SymbolicEnvironment(p.grid, start, p.world, observer, p.surface.height_at,
                              p.heading_count, verify_path=False)
    session = SearchSession(graph, navigation, [p.target], policy=policy, max_steps=max_steps,
                            coverage=coverage)
    while True:
        decision = session.next_decision(env.pose)
        if decision is None:
            break
        goal = decision.goals[0]
        if condition == "disk6":
            outcome = env.visit(decision)
        else:
            loc = graph.locations[decision.location_id]
            hs = (H.full(goal) if condition == "sector6" else H.front(goal) if condition == "front3"
                  else H.front_first(goal) if condition == "front_first6"
                  else H.cover(goal, loc.bbox_min, loc.bbox_max, HFOV))
            outcome = env.visit(decision, headings=hs, stop=session.all_found_in)
        session.report(decision, outcome)
    if condition == "disk6":                  # the full turn of the training model
        rotations, turn = 5 * env.visits, env.visits * math.radians(300)
    else:
        rotations, turn = env.rotations, env.turn_rad
    return {"found": session.targets[0].status.value == "found", "distance_m": round(env.distance_m, 3),
            "visits": env.visits, "headings": env.headings_observed, "rotations": rotations,
            "turn_deg": round(math.degrees(turn), 1),
            "time_s": round(modelled_time(env.distance_m, rotations, turn, env.headings_observed), 1)}


_STATE = {}


def _init(pattern, stats):
    _STATE["scenes"], _STATE["stats"] = Scenes(pattern), stats


def episode(args, state):
    p = _STATE["scenes"].problems(state["scene_id"])[(state["component"], state["target"])]
    start = Pose2D(*p.grid.to_xy(tuple(state["start_row_col"])), 0.0)
    row = {"state_id": state["state_id"], "scene_id": state["scene_id"], "target": state["target"],
           "oracle_distance_m": state["oracle_distance_m"]}
    graph, navigation = p.graph, p.navigation()
    for name in args.planners:
        for condition in args.conditions:
            inner = (MinCostPolicy() if name == "greedy"
                     else CooccurrencePolicy(_STATE["stats"], step_cost=args.step_cost))
            policy = ShuffledPolicy(inner, f"order:{state['state_id']}")     # same orders everywhere
            coverage = p.coverage() if name == "cooccurrence" else None
            row[f"{condition}/{name}"] = run(p, start, graph, navigation, policy, condition,
                                             coverage, args.max_steps)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--planners", nargs="+", choices=["greedy", "cooccurrence"], default=["greedy"])
    parser.add_argument("--start-states", default="outputs/eval_v5/*/start_states.jsonl")
    parser.add_argument("--scenes", choices=["holdout", "val", "test"], default="val")
    parser.add_argument("--step-cost", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grpo_cfg = yaml.safe_load((ROOT / "configs/training/grpo.yaml").read_text(encoding="utf-8"))
    sft_cfg = yaml.safe_load((ROOT / "configs/training/sft_lora.yaml").read_text(encoding="utf-8"))
    fraction = sft_cfg["validation"]["holdout_scene_fraction"]
    states, split_of = holdout_states(args.start_states, grpo_cfg, fraction, args.scenes)
    states.sort(key=lambda s: (s["scene_id"], zlib.crc32(s["state_id"].encode())))
    if args.limit:
        states = sorted(states, key=lambda s: zlib.crc32(s["state_id"].encode()))[:args.limit]
        states.sort(key=lambda s: s["scene_id"])
    stats = placement_stats(grpo_cfg, fraction, split_of) if "cooccurrence" in args.planners else None
    done = set()
    if args.output.exists():
        done = {json.loads(l)["state_id"] for l in args.output.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [s for s in states if s["state_id"] not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"에피소드 {len(states)}개 (남은 {len(todo)}), 조건 {args.conditions}, 플래너 {args.planners}", flush=True)
    # Episodes of one building go to one worker (scene loading dominates).
    by_scene = {}
    for s in todo:
        by_scene.setdefault(s["scene_id"], []).append(s)
    t0 = time.time()
    with args.output.open("a", encoding="utf-8") as out, \
            ProcessPoolExecutor(args.workers, initializer=_init, initargs=(grpo_cfg["layouts"], stats)) as pool:
        futures = [pool.submit(_scene_rows, args, rows) for rows in by_scene.values()]
        k = 0
        for f in futures:
            for row in f.result():
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                k += 1
            out.flush()
            print(f"[{k}/{len(todo)}] 경과 {(time.time() - t0) / 60:.1f}분", flush=True)


def _scene_rows(args, states):
    return [episode(args, s) for s in states]


if __name__ == "__main__":
    main()
