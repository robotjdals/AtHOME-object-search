"""How much better could the Student search with its own choices? (RL
headroom, as pass@k for reasoning RL, Yue et al. 2025: RL mostly moves the
greedy result toward the best of the policy's own samples.)

From each start state the served SFT planner (vLLM: room -> room, workspace
and standalone -> search_location) searches once greedily (the robot's
temperature 0) and ``--samples`` times with room and workspace decisions
sampled (temperature 1, one vLLM seed per sample; the stages GRPO trains),
standalone decisions greedy. Random-policy searches with the same number of
samples give the reference for how much best-of-k gains from variance alone.
Candidate orders and detections are the same for every search of a start
(``--detection``: athome.symbolic.ViewpointDetectionModel config; default
perfect detection), so the samples differ only in the planner's choices.

Per search: found (the real target; ending on a false detection is a
failure), visits, distance, cost D + 3N, and the efficiency score of
AUC-E for one episode, found x (max_steps - visits + 1) / max_steps (the area
under its success-within-budget curve).

    python scripts/measure_search_headroom.py run --planner-url http://127.0.0.1:16007 \\
        --detection configs/perception/detection_412.json --scene-ids <ids> --output <shard>.json
    python scripts/measure_search_headroom.py summarize <shard>.json ... --output <summary>.json
"""
import argparse
import collections
import glob
import json
import os
from pathlib import Path
import time
import zlib

import numpy as np
import yaml

from athome.inference.llm_policy import LLMPolicy
from athome.schemas import Pose2D
from athome.search import SearchSession
from athome.search.policy import RandomPolicy, ShuffledPolicy, Stage
from athome.symbolic.environment import FALSE_DETECTION, SymbolicEnvironment, ViewpointDetectionModel
from athome.training.findability import oracle_distance

from evaluate_student import _ByStage
from train_grpo import ROOT, Scenes, held_out

SFT_MODELS = {Stage.ROOM: "room", Stage.WORKSPACE: "search_location", Stage.STANDALONE: "search_location"}


def search(p, start, policy, state_id, detection_cfg, max_steps, step_cost, coverage):
    detection = (ViewpointDetectionModel(detection_cfg["mean"], detection_cfg["rho"],
                                         tuple(detection_cfg["false_positive_rates"]), p_scene(state_id),
                                         false_labels=(p.target,)) if detection_cfg else None)
    env = SymbolicEnvironment(p.grid, start, p.world, p.observer, p.surface.height_at, p.heading_count,
                              verify_path=False, detection=detection)
    session = SearchSession(p.graph, p.navigation(), [p.target], policy=ShuffledPolicy(policy, f"order:{state_id}"),
                            max_steps=max_steps, coverage=p.coverage() if coverage else None)
    while True:
        decision = session.next_decision(env.pose)
        if decision is None:
            break
        session.report(decision, env.visit(decision))
    target = session.targets[0]
    false_found = (target.status.value == "found"
                   and env.id_map[target.found_object.object_id].startswith(FALSE_DETECTION))
    found = target.status.value == "found" and not false_found
    return {"found": found, "false_found": false_found, "visits": env.visits, "distance_m": env.distance_m,
            "cost": env.distance_m + step_cost * env.visits,
            "auc": found * (max_steps - env.visits + 1) / max_steps,
            "fallbacks": sum(r.policy_fallback for r in session.history),
            "connection_failures": sum(r.policy_fallback and CONNECTION in r.fallback_reason for r in session.history)}


# LLMPolicy's message when the server is not reached (tunnel down): such a
# search measured nearest-first, not the planner, so the start is searched again.
CONNECTION = "연결 실패"


def p_scene(state_id):
    return state_id.split(":")[0]


def run(args):
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    grpo_cfg = yaml.safe_load((ROOT / "configs/training/grpo.yaml").read_text(encoding="utf-8"))
    llm = yaml.safe_load((ROOT / "configs/robot/demo.yaml").read_text(encoding="utf-8"))["planner"]["llm"]
    key = os.environ.get(llm["api_key_env"]) if llm.get("api_key_env") else None
    detection_cfg = json.loads(args.detection.read_text(encoding="utf-8")) if args.detection else None
    states = [json.loads(line) for path in sorted(glob.glob(str(ROOT / args.start_states)))
              for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    states = [s for s in states if s["oracle_distance_m"] is not None
              and held_out(s["scene_id"], grpo_cfg["holdout_scene_fraction"])
              and (not args.scene_ids or s["scene_id"] in set(args.scene_ids))]
    if args.state_ids:
        states = [s for s in states if s["state_id"] in set(args.state_ids)]
    states = states[:args.limit] if args.limit else states
    scenes = Scenes(grpo_cfg["layouts"])

    # The server must answer: a failed request falls back to nearest-first and
    # would be measured as the planner.
    from athome.inference.http import get_json
    served = {m["id"] for m in get_json(args.planner_url.rstrip("/") + "/v1/models", 10.0,
                                        {"Authorization": f"Bearer {key}"} if key else {})["data"]}
    if set(SFT_MODELS.values()) - served:
        raise SystemExit(f"서버에 없는 모델: {sorted(set(SFT_MODELS.values()) - served)}")

    def planner(temperature=0.0, seed=None):
        return LLMPolicy(args.planner_url, SFT_MODELS, timeout=llm["timeout"], retries=llm.get("retries", 1),
                         api_key=key, temperature=temperature, seed=seed)
    rows, t0 = [], time.time()
    for k, s in enumerate(states, 1):
        p = scenes.problems(s["scene_id"])[(s["component"], s["target"])]
        start = Pose2D(*p.grid.to_xy(tuple(s["start_row_col"])), 0.0)
        common = dict(state_id=s["state_id"], detection_cfg=detection_cfg, max_steps=args.max_steps,
                      step_cost=args.step_cost)
        for attempt in range(1, args.attempts + 1):
            row = {"state_id": s["state_id"],
                   "oracle_distance_m": oracle_distance(p.navigation(), p.graph.locations, start, p.observer,
                                                        p.target_ids),
                   "greedy": search(p, start, planner(), coverage=True, **common), "sampled": [], "random": []}
            for j in range(args.samples):
                seed = zlib.crc32(f"{s['state_id']}:{j}".encode())
                sampled = planner(args.temperature, seed)
                policy = _ByStage({Stage.ROOM: sampled, Stage.WORKSPACE: sampled, Stage.STANDALONE: planner()})
                row["sampled"].append(search(p, start, policy, coverage=True, **common))
                row["random"].append(search(p, start, RandomPolicy(seed), coverage=False, **common))
            lost = row["greedy"]["connection_failures"] + sum(x["connection_failures"] for x in row["sampled"])
            if not lost:
                break
            print(f"{s['state_id']}: 서버 연결 실패 {lost}회, 다시 탐색 ({attempt}/{args.attempts})", flush=True)
            time.sleep(60)
        else:
            raise SystemExit(f"{s['state_id']}: 서버에 연결되지 않아 멈춥니다.")
        rows.append(row)
        print(f"[{k}/{len(states)}] {(time.time() - t0) / 60:.1f}분", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"detection": str(args.detection) if args.detection else "perfect",
                                       "samples": args.samples, "temperature": args.temperature,
                                       "max_steps": args.max_steps, "rows": rows}, ensure_ascii=False),
                           encoding="utf-8")
    print("wrote", args.output)


def summarize(args):
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    parts = [json.loads(Path(f).read_text(encoding="utf-8")) for f in args.shards]
    rows = [r for part in parts for r in part["rows"]]
    n = parts[0]["samples"]
    rng = np.random.default_rng(0)
    by_building = collections.defaultdict(list)
    for r in rows:
        by_building[p_scene(r["state_id"])].append(r)
    buildings = sorted(by_building)

    def metrics(get):
        """Per-start value -> mean and building-bootstrap 95% CI."""
        mean = float(np.mean([get(r) for r in rows]))
        boot = []
        for _ in range(5000):
            pick = rng.choice(len(buildings), len(buildings))
            boot.append(np.mean([get(r) for i in pick for r in by_building[buildings[i]]]))
        return {"mean": mean, "ci95": [float(np.percentile(boot, q)) for q in (2.5, 97.5)]}

    def best(runs, k):
        return max(x["auc"] for x in runs[:k])
    out = {"starts": len(rows), "buildings": len(buildings), "samples": n, "detection": parts[0]["detection"],
           "fallbacks": sum(r["greedy"]["fallbacks"] + sum(x["fallbacks"] for x in r["sampled"]) for r in rows)}
    for name, get in (("auc", lambda x: x["auc"]), ("success", lambda x: float(x["found"])),
                      ("sr_at_5", lambda x: float(x["found"] and x["visits"] <= 5)),
                      ("false_found", lambda x: float(x["false_found"]))):
        out[name] = {"sft_greedy": metrics(lambda r: get(r["greedy"])),
                     "sft_sample_mean": metrics(lambda r: np.mean([get(x) for x in r["sampled"]])),
                     "random_sample_mean": metrics(lambda r: np.mean([get(x) for x in r["random"]]))}
    out["auc_best_of_k"] = {
        "sft": {k: float(np.mean([best(r["sampled"], k) for r in rows])) for k in range(1, n + 1)},
        "random": {k: float(np.mean([best(r["random"], k) for r in rows])) for k in range(1, n + 1)}}
    out["headroom"] = {
        "sft_best_minus_greedy_auc": metrics(lambda r: best(r["sampled"], n) - r["greedy"]["auc"]),
        "random_best_minus_mean_auc": metrics(lambda r: best(r["random"], n) - np.mean([x["auc"] for x in r["random"]])),
        "starts_where_a_sample_beats_greedy": float(np.mean([best(r["sampled"], n) > r["greedy"]["auc"] for r in rows]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--planner-url", required=True)
    r.add_argument("--detection", type=Path, help="ViewpointDetectionModel 설정 (없으면 완벽 검출)")
    r.add_argument("--start-states", default="outputs/eval_v5_everygoal/*/start_states.jsonl")
    r.add_argument("--scene-ids", nargs="*", default=[], help="이 장면들만 (나눠 돌리기용)")
    r.add_argument("--limit", type=int, help="앞에서부터 이 개수만(시험 실행)")
    r.add_argument("--state-ids", nargs="*", default=[], help="이 시작 상태들만 (다시 돌리기용)")
    r.add_argument("--attempts", type=int, default=5, help="서버 연결 실패가 있으면 그 시작 상태를 다시 탐색할 횟수")
    r.add_argument("--samples", type=int, default=8)
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--max-steps", type=int, default=60)
    r.add_argument("--step-cost", type=float, default=3.0)
    r.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("shards", nargs="+")
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args) if args.mode == "run" else summarize(args)


if __name__ == "__main__":
    main()
