"""Teacher pseudo-label episodes on the component masked graphs (proposal 6-3).

For every component x target with a target instance in the component and
several start cells, one episode runs with the robot's own search rules
(SearchSession, shared candidate construction and Visited update) and the
symbolic environment (observation model in configs/data/symbolic_observation.json,
wall_los_2d). The Teacher chooses at every planner query; each query is
recorded with the Student-format input, the choice, its reason and the GT
verification (src/athome/training/verify.py). The episode follows the
Teacher's choices (behavior cloning on Teacher trajectories); failed
selections stay in the episode but are marked for exclusion from SFT.

The label is argmax likelihood / (path cost + lambda_s) over the Teacher's
per-candidate likelihoods (src/athome/training/teacher.py); --step-cost sets
lambda_s for the episode, and other lambda values are graded on the same
states for comparison.

Start cells are drawn uniformly from the component's free cells (area
proportional) with a seed derived from --seed, component and target.

--teacher mincost runs the offline stand-in (no API) to test the pipeline and
to count prompt tokens before any OpenAI submission. --teacher openai calls
the Teacher (external transfer and cost: get approval first).
Layout: ATHOME_HM3D_LAYOUT (see src/athome/data/hm3d/layout.py).
"""
import argparse
from dataclasses import asdict
import json
import time
from pathlib import Path
import zlib

from athome.inference.chat import ChatJSON, QuotaExceeded
from athome.navigation import NavigationPlanner
from athome.scene_graph.semantic_labeling import MODEL
from athome.schemas import Pose2D
from athome.search import SearchSession
from athome.symbolic import SymbolicEnvironment
from athome.training.findability import findable_from, findable_starts
from athome.training.teacher import MinCostTeacher, TeacherPolicy
from athome.training.verify import GroundTruth

from scene_episodes import SceneProblems


def make_teacher(kind, step_cost):
    if kind == "mincost":
        return MinCostTeacher(step_cost=step_cost)
    client = ChatJSON("https://api.openai.com", MODEL, api_key_env="OPENAI_API_KEY",
                      timeout=60.0, schema_name="teacher_likelihoods", retries=5, backoff_s=2.0)
    return TeacherPolicy(client, step_cost=step_cost)


def api_usage(records):
    """Actual tokens reported by the API and their cost (configs/data/llm_pricing.json)."""
    total = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "calls": 0}
    for r in records:
        if r.usage:
            total["calls"] += 1
            for k in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                total[k] += r.usage.get(k, 0)
    pricing = json.loads((Path(__file__).resolve().parents[1] / "configs/data/llm_pricing.json")
                         .read_text(encoding="utf-8"))
    price = pricing["models"].get(MODEL)
    if price:
        total["cost_usd"] = round(
            ((total["prompt_tokens"] - total["cached_tokens"]) * price["input"]
             + total["cached_tokens"] * price["cached_input"]
             + total["completion_tokens"] * price["output"]) / 1e6, 4)
        total["pricing_checked"] = pricing["checked"]
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=["mincost", "openai"], required=True)
    parser.add_argument("--starts", type=int, default=3, help="component x target 당 시작 위치 수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--step-cost", type=float, default=3.0,
                        help="lambda_s [m]: label = argmax likelihood / (cost + lambda_s); 에피소드 진행에 사용")
    parser.add_argument("--targets", nargs="*", help="기본값: pilot_targets 전체")
    parser.add_argument("--room-verification", choices=["containment", "observation"],
                        default="containment",
                        help="방 단계 검증: 목표가 들어 있는 방(기본) / 목표가 보이는 위치가 있는 방(이전)")
    parser.add_argument("--commit-to-room", action="store_true",
                        help="이전 방식: 고른 방의 위치를 다 돌 때까지 방을 유지(비교용)")
    parser.add_argument("--output", type=Path, required=True, help="새 폴더(기존 폴더 거부)")
    parser.add_argument("--findable-rule", choices=["every_goal", "any_goal"], default="every_goal",
                        help="시작 위치 규칙 (v5 train 라벨은 any_goal로 생성됨)")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")

    scene = SceneProblems(args.targets)
    records, episodes, unfindable = [], [], []
    aborted = None
    try:
        for problem in scene.problems():
            name, target, grid, graph = problem.component, problem.target, problem.grid, problem.graph
            observer, gt = problem.observer, problem.gt
            seed = args.seed + zlib.crc32(f"{name}:{target}".encode())
            # Only starts from which the target is findable (proposal reward
            # assumes rollouts end by finding it; athome.training.findability).
            probe = problem.navigation()
            starts = findable_starts(grid, args.starts, seed, lambda cell: findable_from(
                probe, graph.locations, Pose2D(*grid.to_xy(cell), 0.0), observer,
                problem.target_ids, args.findable_rule) is not None)
            if len(starts) < args.starts:
                unfindable.append({"component": name, "target": target, "starts": len(starts)})
            for k, cell in enumerate(starts):
                episode_id = f"{name}:{target}:{k}"
                navigation = NavigationPlanner(grid, problem.nav_config)
                teacher = make_teacher(args.teacher, args.step_cost)
                session = SearchSession(graph, navigation, [target], policy=teacher,
                                        max_steps=args.max_steps,
                                        commit_to_room=args.commit_to_room,
                                        coverage=problem.coverage())
                start = Pose2D(*grid.to_xy(cell), 0.0)
                env = SymbolicEnvironment(grid, start, problem.world, observer,
                                          problem.surface.height_at, problem.heading_count)
                truth = GroundTruth(graph, navigation, observer, problem.target_ids,
                                    problem.gt_workspaces, gt_rooms=problem.gt_rooms,
                                    room_rule=args.room_verification)
                fallbacks = 0
                while True:
                    before, pose, visited = len(teacher.records), env.pose, set(session.visited)
                    decision = session.next_decision(pose)
                    for rec in teacher.records[before:]:
                        rec.meta = {"episode_id": episode_id, "component": name,
                                    "step": session.steps + 1, "pose": [pose.x, pose.y, pose.yaw],
                                    "visited": sorted(visited)}
                        if not rec.single_candidate and rec.selected_id is not None:
                            check = truth.verify(
                                rec.stage, rec.selected_id, [c["candidate_id"] for c in rec.candidates],
                                pose, visited)
                            rec.meta["verification"] = check
                            # Same state, other lambda_s: offline comparison only.
                            rec.meta["passed_by_step_cost"] = {
                                k: v in check["valid_candidates"]
                                for k, v in rec.selection_by_step_cost.items()}
                        records.append(rec)
                    if decision is None:
                        break
                    outcome = env.visit(decision)
                    step = session.report(decision, outcome)
                    fallbacks += step.policy_fallback
                result = session.targets[0]
                episodes.append({
                    "episode_id": episode_id, "component": name, "target": target,
                    "start_row_col": list(cell), "status": result.status.value,
                    "session_status": session.status.value, "steps": session.steps,
                    "distance_m": env.distance_m, "gt_count": len(gt), "policy_fallbacks": fallbacks,
                    "found_object_id": (env.id_map[result.found_object.object_id]
                                        if result.found_object else None)})
                print(f"{episode_id}: {result.status.value}, 방문 {session.steps}, "
                      f"질의 {len(teacher.records)}", flush=True)
    except QuotaExceeded as e:
        # Out of credit: stop and keep everything collected so far.
        aborted = f"quota_exceeded: {e}"
        print("API 크레딧 소진으로 중단:", e, flush=True)

    labelled = [r for r in records if "verification" in r.meta]
    passed = [r for r in labelled if r.meta["verification"]["passed"]]
    by_stage = {}
    for r in labelled:
        s = by_stage.setdefault(r.stage, {"queries": 0, "passed": 0, "no_valid_candidate": 0})
        s["queries"] += 1
        s["passed"] += r.meta["verification"]["passed"]
        s["no_valid_candidate"] += not r.meta["verification"]["valid_candidates"]
    by_step_cost = {}
    for r in labelled:
        for k, ok in r.meta["passed_by_step_cost"].items():
            s = by_step_cost.setdefault(k, {}).setdefault(r.stage, {"queries": 0, "passed": 0})
            s["queries"] += 1
            s["passed"] += ok
    summary = {
        "teacher": args.teacher, "model": MODEL if args.teacher == "openai" else None,
        "prompt_version": TeacherPolicy.prompt_version,
        "teacher_prompt_version": TeacherPolicy.teacher_prompt_version,
        "step_cost_m": args.step_cost, "passed_by_step_cost": by_step_cost,
        "layout": scene.layout.source,
        "observation_model": scene.observation_config, "seed": args.seed, "starts_per_target": args.starts,
        "max_steps": args.max_steps, "episodes": len(episodes),
        "start_policy": f"uniform_free_cells_target_findable (athome.training.findability, {args.findable_rule})",
        "room_selection": "commit_to_room" if args.commit_to_room else "replan_each_visit",
        "room_verification": args.room_verification,
        "combinations_short_of_findable_starts": unfindable,
        "found": sum(e["status"] == "found" for e in episodes),
        "queries": len(records), "single_candidate": sum(r.single_candidate for r in records),
        "labelled_queries": len(labelled), "passed": len(passed), "by_stage": by_stage,
        "teacher_errors": sum(bool(r.error) for r in records),
        "aborted": aborted,
        "api_usage": api_usage(records),
    }
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4.1")
        summary["teacher_prompt_tokens"] = sum(
            len(enc.encode(m["content"])) for r in records if not r.single_candidate
            for m in r.teacher_messages)
    except Exception as e:  # noqa: BLE001 - estimate only
        summary["teacher_prompt_tokens"] = f"unavailable: {e}"
    # An aborted run is incomplete: keep what was paid for next to the output
    # (not as the output), and fail so a re-run labels the scene from the start.
    out = args.output
    if aborted:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        out = args.output.with_name(f"{args.output.name}.aborted_{stamp}")
    out.mkdir(parents=True)
    with (out / "queries.jsonl").open("x", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    with (out / "episodes.json").open("x", encoding="utf-8") as f:
        json.dump({"summary": summary, "episodes": episodes}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if aborted:
        raise SystemExit(f"중단된 Teacher 실행(미완성) → {out}. 크레딧 충전 후 다시 실행하세요.")


if __name__ == "__main__":
    main()
