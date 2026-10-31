"""Scene graph ablations with the same planner (no training).

On the held-out start states of scripts/evaluate_student.py, each planner
searches with each representation of the same masked scene graph, in the
symbolic environment and with the robot's search rules. Names follow the
ablation convention of SG-Nav / MoMa-LLM ("w/o ..."):

  full                rooms -> workspaces (function, objects on them) ->
                      standalone groups, Search Location policy, observed
                      share of rooms
  wo_observation      w/o observation state: rooms without their observed share
  wo_location_policy  w/o Search Location policy: every object but structural
                      classes (walls, doors) is a location
  wo_workspace        w/o workspace layer: rooms -> every object of the room
                      (HSG-ON's ablation)
  object_list         unstructured object list: every object of the scene in
                      one list (MoMa-LLM's "Unstructured LLM" baseline)

Planners (MoMa-LLM's baseline names where they exist): llm (GPT-4.1
likelihoods, the Teacher's search index rule; --teacher dry counts prompt
tokens without calls), qwen (the robot's planner prompt on the untrained
Qwen3-4B base served by vLLM: zero-shot, no adapter trained on any
representation), cooccurrence (object co-occurrence statistics from the
training scenes, Kollar & Roy 2009, same rule), greedy (nearest first),
random.

    python scripts/compare_graph_representations.py --planners greedy random cooccurrence \
        --output outputs/graph_compare/free.jsonl
    python scripts/compare_graph_representations.py --planners llm --teacher openai \
        --variants full wo_observation wo_workspace object_list --per-building 14 \
        --output outputs/graph_compare/teacher.jsonl
"""
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
import glob
import json
from pathlib import Path
import threading
import time
import zlib

import yaml

from athome.data.hm3d.layout import load_layout, repo_path
from athome.navigation import NavigationPlanner
from athome.scene_graph.query import DEFAULT_EXCLUDED_CATEGORIES, SceneGraph
from athome.schemas import Pose2D
from athome.search import SearchSession
from athome.search.baselines import (CooccurrencePolicy, FlatSearchSession, PlacementStats,
                                     WithoutObservation, flat_graph)
from athome.inference.llm_policy import LLMPolicy
from athome.search.policy import MinCostPolicy, PolicyError, RandomPolicy, ShuffledPolicy, Stage
from athome.symbolic.environment import SymbolicEnvironment
from athome.training.teacher import MinCostTeacher, TeacherPolicy

from export_sft_dataset import scene_split
from train_grpo import ROOT, Scenes, held_out

VARIANTS = ("full", "wo_observation", "wo_location_policy", "wo_workspace", "object_list")
PLANNERS = ("llm", "qwen", "cooccurrence", "greedy", "random")
LLM_PLANNERS = ("llm", "qwen")


CONTEXT_ERROR = re.compile(r"maximum context length is (\d+) tokens.*?(\d+) in the messages", re.S)


class CountingLLMPolicy(LLMPolicy):
    """The robot's planner client, recording each query's candidate count,
    the server's token usage and, for queries without a valid answer, why."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queries = []                    # (candidates, prompt tokens or None)
        self.failures = []                   # (kind, candidates, prompt tokens or None)

    def select(self, stage, target, candidates, context):
        self._candidates = len(candidates)
        try:
            return super().select(stage, target, candidates, context)
        except PolicyError as e:
            m = CONTEXT_ERROR.search(str(e))
            kind = "context_length" if m else str(e)[:120]
            self.failures.append((kind, len(candidates), int(m.group(2)) if m else None))
            raise

    def _parse(self, response, aliases):
        self.queries.append((self._candidates, (response.get("usage") or {}).get("prompt_tokens")))
        return super()._parse(response, aliases)


def holdout_states(pattern, grpo_cfg, fraction, scenes="holdout"):
    states = [json.loads(l) for p in sorted(glob.glob(str(ROOT / pattern)))
              for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    split_config = json.loads((ROOT / "configs/data/scene_splits.json").read_text(encoding="utf-8"))
    splits = {}

    def split_of(scene_id):
        if scene_id not in splits:
            layout = load_layout(ROOT / grpo_cfg["layouts"].format(scene_id=scene_id))
            scene = json.loads(layout.annotations.read_text(encoding="utf-8"))["scene_path"]
            splits[scene_id] = scene_split(repo_path(scene), split_config)
        return splits[scene_id]

    def wanted(s):
        if scenes == "holdout":        # SFT validation buildings (train split)
            return split_of(s["scene_id"]) == "train" and held_out(s["scene_id"], fraction)
        return split_of(s["scene_id"]) == scenes         # official val / test buildings

    return [s for s in states if s["oracle_distance_m"] is not None and wanted(s)], split_of


def placement_stats(grpo_cfg, fraction, split_of):
    """Statistics of the training buildings (not the held-out ones)."""
    stats = PlacementStats()
    for path in sorted(glob.glob(str(ROOT / grpo_cfg["layouts"].format(scene_id="*")))):
        layout = load_layout(path)
        scene_id = Path(path).name.split(".")[0]
        if split_of(scene_id) != "train" or held_out(scene_id, fraction):
            continue
        graph = json.loads(layout.graph.read_text(encoding="utf-8"))
        catalog = json.loads(layout.catalog.read_text(encoding="utf-8"))
        stats.add_scene(graph, {t["target_category"]: t["instances"] for t in catalog["targets"]})
    return stats


class VariantGraphs:
    """Graph and navigation planner of each representation, per problem."""

    def __init__(self, grpo_cfg):
        self.pattern = grpo_cfg["layouts"]
        self.cache = {}
        self.lock = threading.Lock()

    def get(self, scene_id, problem, variant):
        key = (scene_id, problem.component, problem.target, variant)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        if variant in ("full", "wo_observation"):
            graph, navigation = problem.graph, problem.navigation()
        else:
            layout = load_layout(ROOT / self.pattern.format(scene_id=scene_id))
            raw = json.loads((layout.scene_dir / "component_masked_graphs.review"
                              / f"{problem.component}.{problem.target}.workspace_graph.json")
                             .read_text(encoding="utf-8"))
            if variant != "wo_location_policy":
                raw = flat_graph(raw)
            graph = SceneGraph(raw, room_floor_z=problem.graph.room_floor_z,
                               excluded_categories=DEFAULT_EXCLUDED_CATEGORIES)
            navigation = NavigationPlanner(problem.grid, problem.nav_config)
            for lid, loc in graph.locations.items():
                navigation.add_location(lid, loc.bbox_min, loc.bbox_max)
        with self.lock:
            self.cache[key] = (graph, navigation)
        return graph, navigation


def make_planner(name, args, state_id, stats, usage):
    if name == "greedy":
        return MinCostPolicy()
    if name == "random":
        return RandomPolicy(zlib.crc32(state_id.encode()))
    if name == "cooccurrence":
        return CooccurrencePolicy(stats, step_cost=args.step_cost)
    if name == "qwen":
        import os
        model = args.qwen_model
        return CountingLLMPolicy(args.planner_url, {s: model for s in Stage}, timeout=args.planner_timeout,
                                 retries=1, api_key=os.environ.get("ATHOME_PLANNER_API_KEY"))
    if args.teacher == "dry":
        return MinCostTeacher(step_cost=args.step_cost)
    from athome.inference.chat import ChatJSON
    from athome.scene_graph.semantic_labeling import MODEL

    client = ChatJSON("https://api.openai.com", MODEL, api_key_env="OPENAI_API_KEY",
                      timeout=120.0, retries=4, backoff_s=2.0)
    usage.append(client)
    return TeacherPolicy(client, step_cost=args.step_cost, compare_step_costs=())


def episode(args, p, graphs, stats, state):
    start = Pose2D(*p.grid.to_xy(tuple(state["start_row_col"])), 0.0)
    row = {"state_id": state["state_id"], "scene_id": state["scene_id"], "target": state["target"],
           "oracle_distance_m": state["oracle_distance_m"]}
    for variant in args.variants:
        graph, navigation = graphs.get(state["scene_id"], p, variant)
        for name in args.planners:
            usage = []
            inner = make_planner(name, args, state["state_id"], stats, usage)
            policy = ShuffledPolicy(inner, f"order:{state['state_id']}")     # same orders for every run
            if variant == "wo_observation":
                policy = WithoutObservation(policy)
            env = SymbolicEnvironment(p.grid, start, p.world, p.observer, p.surface.height_at,
                                      p.heading_count, verify_path=False)
            rooms = variant in ("full", "wo_location_policy", "wo_workspace")
            # The observed share only feeds selection rules that use it (the
            # Teacher's and the statistics' after-search update).
            coverage = p.coverage() if rooms and name in ("llm", "qwen", "cooccurrence") else None
            if variant == "object_list":
                session = FlatSearchSession(graph, navigation, [p.target], policy=policy,
                                            max_steps=args.max_steps)
            else:
                session = SearchSession(graph, navigation, [p.target], policy=policy,
                                        max_steps=args.max_steps, coverage=coverage,
                                        group_standalone=variant != "wo_workspace")
            stages = []
            while True:
                decision = session.next_decision(env.pose)
                if decision is None:
                    break
                session.report(decision, env.visit(decision))
                stages.append(decision.stage.value)
            found = session.targets[0].status.value == "found"
            result = {"found": found, "distance_m": env.distance_m, "visits": env.visits,
                      "cost": env.distance_m + args.step_cost * env.visits,
                      "fallbacks": sum(r.policy_fallback for r in session.history),
                      "locations": len(graph.locations)}
            if name == "llm":
                records = inner.records
                result["queries"] = sum(not r.single_candidate for r in records)
                result["candidates_max"] = max((len(r.candidates) for r in records), default=0)
                if args.teacher == "dry":
                    result["prompt_tokens"] = sum(count_tokens(r.teacher_messages) for r in records
                                                  if not r.single_candidate)
                    # Likelihood answer: ~16 tokens per candidate + reasoning.
                    result["completion_tokens_est"] = sum(16 * len(r.candidates) + 200 for r in records
                                                          if not r.single_candidate)
                else:
                    result["prompt_tokens"] = sum(c.usage["prompt_tokens"] for c in usage)
                    result["completion_tokens"] = sum(c.usage["completion_tokens"] for c in usage)
                    result["errors"] = sum(bool(r.error) for r in records)
            if name == "qwen":
                queries = inner.queries
                result["queries"] = len(queries)
                result["candidates_max"] = max((c for c, _ in queries), default=0)
                result["candidates_sum"] = sum(c for c, _ in queries)
                result["prompt_tokens"] = sum(t or 0 for _, t in queries)
                result["failures"] = [{"kind": k, "candidates": c, "prompt_tokens": t} for k, c, t in inner.failures]
            row[f"{variant}/{name}"] = result
    return row


_ENC = None


def count_tokens(messages):
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("o200k_base")
    return sum(len(_ENC.encode(m["content"])) + 4 for m in messages)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--planners", nargs="+", choices=PLANNERS, required=True)
    parser.add_argument("--teacher", choices=["dry", "openai"], default="dry")
    parser.add_argument("--planner-url", default="http://127.0.0.1:16007",
                        help="qwen: vLLM server (SSH tunnel end, as configs/robot/demo.yaml)")
    parser.add_argument("--qwen-model", default="qwen3-4b", help="qwen: served base model name")
    parser.add_argument("--planner-timeout", type=float, default=60.0)
    parser.add_argument("--start-states", default="outputs/eval_v5_everygoal/*/start_states.jsonl")
    parser.add_argument("--scenes", choices=["holdout", "val", "test"], default="holdout",
                        help="val/test: --start-states outputs/eval_v5/*/start_states.jsonl")
    parser.add_argument("--step-cost", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--limit", type=int, help="앞에서부터 이 개수만 (장면별 고르게 섞은 순서)")
    parser.add_argument("--per-building", type=int, help="건물마다 이 개수만 (같은 섞은 순서의 앞에서부터)")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env",
                        help="OPENAI_API_KEY / ATHOME_PLANNER_API_KEY (셸에 불러오지 않고 이 프로세스만)")
    parser.add_argument("--workers", type=int, default=1, help="에피소드 병렬 수 (API 호출용)")
    parser.add_argument("--shard", default="0/1", help="i/n: 에피소드를 n개로 나눈 i번째만 (여러 프로세스용)")
    parser.add_argument("--state-ids", type=Path, help="이 파일에 적힌 state_id만 (한 줄에 하나)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grpo_cfg = yaml.safe_load((ROOT / "configs/training/grpo.yaml").read_text(encoding="utf-8"))
    sft_cfg = yaml.safe_load((ROOT / "configs/training/sft_lora.yaml").read_text(encoding="utf-8"))
    fraction = sft_cfg["validation"]["holdout_scene_fraction"]
    states, split_of = holdout_states(args.start_states, grpo_cfg, fraction, args.scenes)
    # A fixed spread over scenes and targets for --limit (crc order, not file order).
    states.sort(key=lambda s: zlib.crc32(s["state_id"].encode()))
    if args.per_building:
        taken = {}
        states = [s for s in states
                  if taken.setdefault(s["scene_id"], []).append(s) or len(taken[s["scene_id"]]) <= args.per_building]
    if args.limit:
        states = states[:args.limit]
    if "qwen" in args.planners or ("llm" in args.planners and args.teacher == "openai"):
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    shard, shards = (int(v) for v in args.shard.split("/"))
    states = states[shard::shards]
    stats = placement_stats(grpo_cfg, fraction, split_of) if "cooccurrence" in args.planners else None
    scenes, graphs = Scenes(grpo_cfg["layouts"]), VariantGraphs(grpo_cfg)
    done = set()
    if args.output.exists():          # resume: episodes already written are kept
        done = {json.loads(l)["state_id"] for l in args.output.read_text(encoding="utf-8").splitlines() if l.strip()}
    if args.state_ids:                # e.g. the episodes to re-examine
        wanted = {l.strip() for l in args.state_ids.read_text(encoding="utf-8").splitlines() if l.strip()}
        states = [s for s in states if s["state_id"] in wanted]
    todo = [s for s in states if s["state_id"] not in done]
    # Problems resolved before the threads (the scene cache is not thread-safe).
    problems = {s["state_id"]: scenes.problems(s["scene_id"])[(s["component"], s["target"])] for s in todo}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"에피소드 {len(states)}개 (남은 {len(todo)}), 표현 {args.variants}, 플래너 {args.planners}", flush=True)
    t0, lock = time.time(), threading.Lock()
    with args.output.open("a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
        for k, row in enumerate(pool.map(lambda s: episode(args, problems[s["state_id"]], graphs, stats, s), todo), 1):
            with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            if k % 10 == 0 or k == len(todo):
                eta = (time.time() - t0) / k * (len(todo) - k) / 60
                print(f"[{k}/{len(todo)}] 경과 {(time.time() - t0) / 60:.1f}분, 남은 약 {eta:.0f}분", flush=True)


if __name__ == "__main__":
    main()
