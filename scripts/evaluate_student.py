"""Evaluate the trained Student planner (LoRA adapters) without API calls.

Mode ``decisions``: on the held-out buildings of an SFT export (the same
sha256 split as scripts/train_sft_lora.py; recorded candidate order only),
the Student's choice against the verified Teacher label, with both decoding
rules: ``score`` (argmax of the whole-answer probability, as GRPO) and
``greedy`` (token by token within the candidate aliases, as the robot's vLLM
server with the JSON-schema enum), and how often the two agree.

Mode ``episodes``: from fixed start states (scripts/export_start_states.py),
the Student searches in the symbolic environment (room / workspace /
standalone adapters) next to the nearest-first and random baselines on the
same starts: success, SPL, distance, visits and the reward cost D + lambda_s N
(proposal 6-4). Scenes: the SFT held-out buildings (default) or all.

    python scripts/evaluate_student.py decisions --data outputs/sft_v5g/train/room.jsonl \
        --adapter outputs/adapters/room/adapter --output outputs/eval_student/room_decisions.json
    python scripts/evaluate_student.py episodes --room outputs/adapters/room/adapter \
        --search-location outputs/adapters/search_location/adapter [--workspace <GRPO adapter>] \
        --output outputs/eval_student/episodes.jsonl
"""
import argparse
from collections import Counter
import glob
import json
from pathlib import Path
import re
import statistics
import zlib

import yaml

from athome.schemas import Pose2D
from athome.search import SearchSession
from athome.search.policy import MinCostPolicy, RandomPolicy, ShuffledPolicy, Stage
from athome.symbolic.environment import SymbolicEnvironment
from athome.training.findability import oracle_distance
from athome.training.hf_policy import (
    HFCandidatePolicy, candidate_logprobs, greedy_constrained, prompt_text)

from export_sft_dataset import scene_split
from athome.data.hm3d.layout import load_layout, repo_path
from train_grpo import ROOT, Scenes, held_out

CONFIG = ROOT / "configs/training/sft_lora.yaml"


def load_base(cfg, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base["name"], revision=base.get("revision"))
    model = AutoModelForCausalLM.from_pretrained(base["name"], revision=base.get("revision"),
                                                 torch_dtype=getattr(torch, base["dtype"])).to(device)
    return model, tokenizer


def decisions(args, cfg, device):
    import torch
    from peft import PeftModel
    model, tokenizer = load_base(cfg, device)
    if args.base_only:                      # the untrained base model (zero-shot reference)
        model.set_adapter = lambda name: None
        model.eval()
    else:
        model = PeftModel.from_pretrained(model, str(args.adapter), adapter_name="student").eval()
    thinking = cfg["base_model"].get("enable_thinking", False)
    fraction = cfg["validation"]["holdout_scene_fraction"]
    examples = [json.loads(l) for l in args.data.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta_path = args.data.with_name(args.data.name.replace(".jsonl", ".meta.jsonl"))
    metas = [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows, records = [], {}

    def candidate_ids(m):
        """alias -> candidate ID of the recorded query (recorded order: C1 = first candidate)."""
        run = m["run"]
        if run not in records:
            records[run] = (Path(run) / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        rec = json.loads(records[run][m["line"]])
        return {f"C{i}": c["candidate_id"] for i, c in enumerate(rec["candidates"], 1)}

    for e, m in zip(examples, metas):
        if m.get("candidate_order", "recorded") != "recorded" or not held_out(m["scene_id"], fraction):
            continue
        *messages, answer = e["messages"]
        aliases = re.findall(r"(?m)^- (C\d+)", messages[-1]["content"])
        label = json.loads(answer["content"])["selected_id"]
        prompt = prompt_text(tokenizer, messages, thinking)
        with torch.no_grad():
            scores = candidate_logprobs(model, tokenizer, prompt, aliases)
        by_score = aliases[int(torch.argmax(scores))]
        by_greedy = greedy_constrained(model, tokenizer, prompt, aliases)
        ids = candidate_ids(m)
        if ids[label] != m["selected_id"]:
            raise SystemExit(f"{e['id']}: 기록과 예제의 후보 순서가 다릅니다.")
        valid = set(m["valid_candidates"])
        rows.append({"id": e["id"], "stage": m["stage"], "candidates": len(aliases), "label": label,
                     "score": by_score, "greedy": by_greedy,
                     # GT: the chosen candidate finds the target (room contains it /
                     # its goal pose observes it), athome.training.verify.
                     "gt_score": ids[by_score] in valid, "gt_greedy": ids[by_greedy] in valid,
                     "gt_share": len(valid) / len(aliases)})
        if args.limit and len(rows) >= args.limit:
            break
    if not rows:
        raise SystemExit("검증 건물의 예제가 없습니다.")

    def rate(sel, key):
        return round(sum(r[key] == r["label"] for r in sel) / len(sel), 4) if sel else None
    summary = {"examples": len(rows), "by_stage": dict(Counter(r["stage"] for r in rows))}
    for name, sel in (("all", rows), ("10+ candidates", [r for r in rows if r["candidates"] >= 10])):
        summary[name] = {"n": len(sel), "accuracy_score": rate(sel, "score"), "accuracy_greedy": rate(sel, "greedy"),
                         "gt_accuracy_score": round(statistics.mean(r["gt_score"] for r in sel), 4) if sel else None,
                         "gt_accuracy_greedy": round(statistics.mean(r["gt_greedy"] for r in sel), 4) if sel else None,
                         "gt_random_expected": round(statistics.mean(r["gt_share"] for r in sel), 4) if sel else None,
                         "score_greedy_agreement": round(sum(r["score"] == r["greedy"] for r in sel) / len(sel), 4)
                         if sel else None,
                         "random_expected": round(statistics.mean(1 / r["candidates"] for r in sel), 4) if sel else None}
    return summary, rows


def spl(found, l_star, distance):
    if not found or l_star is None:
        return 0.0
    longest = max(l_star, distance)
    return l_star / longest if longest > 0 else 1.0


def episodes(args, cfg, device):
    from peft import PeftModel
    grpo_cfg = yaml.safe_load((ROOT / "configs/training/grpo.yaml").read_text(encoding="utf-8"))
    model, tokenizer = load_base(cfg, device)
    if not args.base_only:
        model = PeftModel.from_pretrained(model, str(args.room), adapter_name="room", is_trainable=False)
        model.load_adapter(str(args.search_location), adapter_name="search_location", is_trainable=False)
        model.load_adapter(str(args.workspace or args.search_location), adapter_name="workspace",
                           is_trainable=False)
    model.eval()
    thinking = cfg["base_model"].get("enable_thinking", False)
    fraction = cfg["validation"]["holdout_scene_fraction"]
    states = [json.loads(l) for p in sorted(glob.glob(str(ROOT / args.start_states)))
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
        if args.scenes == "all":
            return True
        if args.scenes == "holdout":        # SFT validation buildings (train split)
            return split_of(s["scene_id"]) == "train" and held_out(s["scene_id"], fraction)
        return split_of(s["scene_id"]) == args.scenes      # official val / test buildings
    states = [s for s in states if s["oracle_distance_m"] is not None and wanted(s)]
    if args.limit:
        states = states[:args.limit]
    scenes = Scenes(grpo_cfg["layouts"])
    step_cost, max_steps = args.step_cost, args.max_steps
    rows = []
    import time
    t0 = time.time()
    for k, s in enumerate(states, 1):
        if k == 1 or k % 10 == 0:
            done = k - 1
            eta = (time.time() - t0) / done * (len(states) - done) / 60 if done else float("nan")
            print(f"[{done}/{len(states)}] 경과 {(time.time() - t0) / 60:.1f}분, 남은 약 {eta:.0f}분", flush=True)
        p = scenes.problems(s["scene_id"])[(s["component"], s["target"])]
        start = Pose2D(*p.grid.to_xy(tuple(s["start_row_col"])), 0.0)
        l_star = oracle_distance(p.navigation(), p.graph.locations, start, p.observer, p.target_ids)
        student = ({st: None for st in (Stage.ROOM, Stage.WORKSPACE, Stage.STANDALONE)} if args.base_only else
                   {Stage.ROOM: "room", Stage.WORKSPACE: "workspace", Stage.STANDALONE: "search_location"})
        targets = json.loads(load_layout(ROOT / grpo_cfg["layouts"].format(scene_id=s["scene_id"]))
                             .targets.read_text(encoding="utf-8"))
        category = ("unseen" if s["target"] in targets.get("unseen_categories", []) else
                    "synonym" if s["target"] in targets.get("synonym_categories", []) else "seen")
        row = {"state_id": s["state_id"], "oracle_distance_m": l_star, "target_split": category}
        for name in ("student", "mincost", "random"):
            if name == "student":
                inner = {st: HFCandidatePolicy(model, tokenizer, ad, enable_thinking=thinking, decoding=args.decoding)
                         for st, ad in student.items()}
                policy = _ByStage(inner)
                if args.oracle != "none":
                    policy = _Oracle(policy, args.oracle, p)
            else:
                policy = MinCostPolicy() if name == "mincost" else RandomPolicy(zlib.crc32(s["state_id"].encode()))
            policy = ShuffledPolicy(policy, f"order:{s['state_id']}")      # same orders for every policy
            env = SymbolicEnvironment(p.grid, start, p.world, p.observer, p.surface.height_at, p.heading_count, verify_path=False)
            session = SearchSession(p.graph, p.navigation(), [p.target], policy=policy, max_steps=max_steps,
                                    coverage=p.coverage() if name == "student" else None)
            while True:
                if isinstance(policy.policy, _Oracle):
                    policy.policy.pose, policy.policy.visited = env.pose, set(session.visited)
                decision = session.next_decision(env.pose)
                if decision is None:
                    break
                session.report(decision, env.visit(decision))
            found = session.targets[0].status.value == "found"
            row[name] = {"found": found, "distance_m": env.distance_m, "visits": env.visits,
                         "spl": spl(found, l_star, env.distance_m),
                         "cost": env.distance_m + step_cost * env.visits}
        rows.append(row)

    summary = {"episodes": len(rows), "decoding": args.decoding, "step_cost_m": step_cost, "scenes": args.scenes,
               "oracle": args.oracle,
               "by_target_split": dict(Counter(r["target_split"] for r in rows))}

    def means(sel):
        return {name: {k: round(statistics.mean(r[name][m] for r in sel), 4)
                       for k, m in (("success_rate", "found"), ("spl", "spl"), ("distance_m", "distance_m"),
                                    ("visits", "visits"), ("cost", "cost"))}
                for name in ("student", "mincost", "random")}
    summary["all"] = means(rows)
    # HM3D-OVON style breakdown: seen / synonym / unseen target categories.
    for split in ("seen", "synonym", "unseen"):
        sel = [r for r in rows if r["target_split"] == split]
        if sel:
            summary[split] = means(sel)
    return summary, rows


class _Oracle:
    """Headroom analysis (oracle ablation): one level of the search decided
    with ground truth, the others by the Student. ``room``: a room that
    contains a target (GT room, the containment rule of the labels), the
    nearest if several; ``within``: a workspace / standalone group whose goal
    pose (the nearest instance for a group) observes a target or is a GT
    workspace (athome.training.verify.GroundTruth), the nearest if several.
    Without such a candidate the Student decides. ``pose`` and ``visited``
    are set by the episode loop before every decision."""

    def __init__(self, student, mode, problem):
        from athome.training.verify import GroundTruth
        self.student, self.mode = student, mode
        self.truth = GroundTruth(problem.graph, problem.navigation(), problem.observer, problem.target_ids,
                                 problem.gt_workspaces, gt_rooms=problem.gt_rooms)
        self.gt_rooms = set(problem.gt_rooms)
        self.pose, self.visited = None, set()

    def select(self, stage, target, candidates, context):
        good = []
        if self.mode == "room" and stage == Stage.ROOM:
            good = [c for c in candidates if c.candidate_id in self.gt_rooms]
        elif self.mode == "within" and stage in (Stage.WORKSPACE, Stage.STANDALONE):
            ids = {c.candidate_id: c.info.get("nearest_location_id", c.candidate_id) for c in candidates}
            open_ids = [l for l in self.truth.graph.locations if l not in self.visited]
            valid = self.truth.valid_locations(self.pose, open_ids)
            good = [c for c in candidates if ids[c.candidate_id] in valid]
        if good:
            return min(good, key=lambda c: (c.cost, c.candidate_id)).candidate_id
        return self.student.select(stage, target, candidates, context)


class _ByStage:
    """Routes each stage to its adapter policy."""

    def __init__(self, policies):
        self.policies = policies

    def select(self, stage, target, candidates, context):
        return self.policies[stage].select(stage, target, candidates, context)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["decisions", "episodes"])
    parser.add_argument("--config", type=Path, default=CONFIG, help="기본 모델 설정(sft_lora.yaml)")
    parser.add_argument("--data", type=Path, help="decisions: <adapter>.jsonl")
    parser.add_argument("--adapter", type=Path, help="decisions: 평가할 어댑터")
    parser.add_argument("--room", type=Path)
    parser.add_argument("--search-location", type=Path)
    parser.add_argument("--workspace", type=Path, help="GRPO 어댑터(없으면 search_location)")
    parser.add_argument("--base-only", action="store_true", help="어댑터 없이 기본 모델만(학습 전 비교)")
    parser.add_argument("--oracle", choices=["none", "room", "within"], default="none",
                        help="개선 여지 분석: 한 단계만 정답(GT)으로 선택")
    parser.add_argument("--start-states", default="outputs/eval_v5_everygoal/*/start_states.jsonl")
    parser.add_argument("--scenes", choices=["holdout", "val", "test", "all"], default="holdout",
                        help="holdout: SFT 검증 건물, val/test: 공식 val 폴더 건물(처음 보는 건물)")
    parser.add_argument("--decoding", choices=["score", "greedy"], default="greedy",
                        help="episodes: Student 선택 방식(로봇과 같은 greedy가 기본)")
    parser.add_argument("--step-cost", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--limit", type=int, help="앞에서부터 이 개수만(시험 실행)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update-meta", type=Path, nargs="*", default=[],
                        help="결과 요약을 이 어댑터 폴더들의 meta.json validation에 추가")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    import torch
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "decisions":
        if not (args.data and (args.adapter or args.base_only)):
            raise SystemExit("decisions에는 --data와 --adapter가 필요합니다.")
        summary, rows = decisions(args, cfg, device)
    else:
        if not (args.base_only or (args.room and args.search_location)):
            raise SystemExit("episodes에는 --room과 --search-location이 필요합니다.")
        summary, rows = episodes(args, cfg, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    from athome.training.adapter_meta import add_validation
    for adapter_dir in args.update_meta:
        add_validation(adapter_dir, {"mode": args.mode, "result_file": str(args.output),
                                     "data": str(args.data) if args.data else args.start_states,
                                     "scenes": args.scenes, "summary": summary})
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
