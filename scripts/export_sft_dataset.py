"""Teacher query records -> LoRA-SFT files (proposal 6-3).

Input: Teacher runs of scripts/generate_teacher_episodes.py (queries.jsonl,
episodes.json). Output (new folder):

    <split>/<adapter>.jsonl        {"id", "messages"}: Student prompt + answer
    <split>/<adapter>.meta.jsonl   provenance per example, same line order
    manifest.json                  inputs, versions, split rule, counts

- Kept: GT-verified selections (meta.verification.passed) with more than one
  candidate and no Teacher error. After a failed Teacher call the session
  continued with a stand-in policy, so later records of that episode are not
  on the Teacher's trajectory (behaviour cloning) and are left out. The answer is the Student output format of
  athome.inference.llm_policy: {"selected_id": "<alias>"}. The Teacher's
  reasoning and likelihoods stay in the meta file, not in the training text.
- Adapters (proposal 6-3/6-4, configs/robot/*.yaml planner.llm.models):
  room -> "room"; workspace and standalone -> "search_location" (the
  workspace adapter for GRPO starts from its weights).
- Target categories (``--categories``, scripts/build_target_categories.py):
  Train keeps only seen categories; Val/Test keep every listed category.
  Records of other targets (removed from the list after labelling, or held
  out for evaluation) are left out.
- Candidate order (RankVicuna, Pradeep et al. 2023: teacher outputs kept in
  the original order plus a copy with shuffled input order): candidates are
  listed by travel cost, so a Student could learn "pick C1". Train keeps
  every example as recorded and adds ``--shuffled-copies`` copies with the
  candidates in a seeded random order, aliases C1..Cn renumbered and the
  answer moved to the same candidate. Val/Test keep the recorded order (what
  the robot sees). Every prompt is rebuilt from the record with
  athome.inference.prompts and must equal the recorded Student prompt before
  it is shuffled (records that do not are left out and counted); "current_room" (not stored in older records) is recovered
  from the recorded "(current room)" marker.
- Split by building (configs/data/scene_splits.json). Exact duplicates
  (same prompt and answer) within a split/adapter are kept once.
- The Student prompt of every run must equal the current planner prompt
  (athome.inference.prompts), since the robot uses that prompt.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from athome.data.hm3d.layout import load_layout
from athome.inference.prompts import PROMPT_VERSION, SYSTEM, build_messages
from athome.search.policy import Candidate, PlanningContext, Stage

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = {"room": "room", "workspace": "search_location", "standalone": "search_location"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def planner_input(record):
    """(stage, candidates, context) of a record; current_room from the marker
    when the record predates its storage."""
    ctx = dict(record["context"])
    if "current_room" not in ctx:
        user = next(m["content"] for m in record["student_messages"] if m["role"] == "user")
        marked = re.findall(r"(?m)^- (C\d+) \(current room\)$", user)
        ctx["current_room"] = record["aliases"][marked[0]] if marked else None
    context = PlanningContext(**{k: tuple(v) if isinstance(v, list) else v for k, v in ctx.items()})
    candidates = [Candidate(c["candidate_id"], c["cost"], c["info"]) for c in record["candidates"]]
    return Stage(record["stage"]), candidates, context


def rendered(record, candidates, stage, context):
    """Student messages and answer for the given candidate order."""
    messages, aliases = build_messages(stage, record["target"], candidates, context)
    alias = next(a for a, cid in aliases.items() if cid == record["selected_id"])
    return messages, json.dumps({"selected_id": alias})


def scene_split(scene_path: Path, config: dict) -> str:
    """Official HM3D split directory -> train / val / test."""
    parts = scene_path.parts
    official = next((config["official_split_dirs"][p] for p in reversed(parts)
                     if p in config["official_split_dirs"]), None)
    if official is None:
        raise ValueError(f"HM3D 공식 split 폴더를 찾을 수 없습니다: {scene_path}")
    if official == "train":
        return "train"
    scene_id = scene_path.name.split(".")[0]
    return "val" if int(sha(scene_id.encode()), 16) % 2 == 0 else "test"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="Teacher 실행 폴더들")
    parser.add_argument("--split-config", type=Path, default=ROOT / "configs/data/scene_splits.json")
    parser.add_argument("--output", type=Path, required=True, help="새 폴더(기존 폴더 거부)")
    parser.add_argument("--categories", type=Path, help="목표 범주 목록(Train: seen만, Val/Test: 전체)")
    parser.add_argument("--shuffled-copies", type=int, default=1, help="Train 예제마다 후보 순서를 섞은 복사본 수")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    split_config = json.loads(args.split_config.read_text(encoding="utf-8"))
    allowed = None
    if args.categories:
        category_list = json.loads(args.categories.read_text(encoding="utf-8"))
        allowed = {"train": set(category_list["seen"])}
        allowed["val"] = allowed["test"] = set(category_list["categories"])
    system_sha = sha(SYSTEM.encode())

    examples = {}            # (split, adapter) -> {key: (example, meta)}
    dropped, duplicates, conflicts = Counter(), Counter(), Counter()
    answers_by_prompt = {}   # (split, adapter, prompt key) -> answers
    runs, scene_splits = [], {}
    for run in args.runs:
        summary = json.loads((run / "episodes.json").read_text(encoding="utf-8"))["summary"]
        if summary["prompt_version"] != PROMPT_VERSION:
            raise SystemExit(f"{run}: 프롬프트 버전 {summary['prompt_version']} != 현재 {PROMPT_VERSION}")
        layout = load_layout(summary["layout"])
        annotation = json.loads(layout.annotations.read_text(encoding="utf-8"))
        scene_path = Path(annotation["scene_path"])
        split = scene_split(scene_path, split_config)
        if scene_splits.setdefault(layout.scene_id, split) != split:
            raise SystemExit(f"{layout.scene_id}: 여러 split에 배정됨")
        raw = (run / "queries.jsonl").read_bytes()
        runs.append({"run": str(run), "queries_sha256": sha(raw), "scene_id": layout.scene_id,
                     "split": split, "layout": summary["layout"], "teacher": summary["teacher"],
                     "model": summary.get("model"), "teacher_prompt_version": summary["teacher_prompt_version"],
                     "step_cost_m": summary["step_cost_m"]})
        lines = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
        first_error = {}
        for record in lines:
            if record["error"]:
                key = record["meta"].get("episode_id")
                first_error[key] = min(first_error.get(key, 10**9), record["meta"].get("step", 0))
        for line_no, record in enumerate(lines):
            stage = record["stage"]
            episode = record["meta"].get("episode_id")
            if episode in first_error and record["meta"].get("step", 0) >= first_error[episode]:
                dropped["after_teacher_fallback"] += 1
                continue
            verification = (record.get("meta") or {}).get("verification")
            reason = ("not_planner_stage" if stage not in ADAPTER else
                      "single_candidate" if record["single_candidate"] else
                      "teacher_error" if record["error"] else
                      "not_verified" if verification is None else
                      "failed_verification" if not verification["passed"] else
                      "target_not_in_categories" if allowed is not None
                      and record["target"] not in allowed[split] else None)
            if reason:
                dropped[reason] += 1
                continue
            messages = record["student_messages"]
            if messages[0]["role"] != "system" or sha(messages[0]["content"].encode()) != system_sha:
                raise SystemExit(f"{run}:{line_no}: Student system 프롬프트가 현재 플래너와 다릅니다.")
            answer = json.dumps({"selected_id": record["selected_alias"]})
            if record["aliases"].get(record["selected_alias"]) != record["selected_id"]:
                raise SystemExit(f"{run}:{line_no}: alias와 선택 ID 불일치")
            stage_enum, candidates, context = planner_input(record)
            if rendered(record, candidates, stage_enum, context) != (messages, answer):
                dropped["prompt_rebuild_mismatch"] += 1
                continue
            recorded_ids = [c.candidate_id for c in candidates]
            variants = [("recorded", messages, answer, None)]
            if split == "train":
                seed = f"{args.seed}:{run.name}:{line_no}"
                rng = random.Random(seed)
                for copy in range(args.shuffled_copies):
                    order = candidates[:]
                    rng.shuffle(order)
                    permutation = [recorded_ids.index(c.candidate_id) for c in order]
                    variants.append(("shuffled", *rendered(record, order, stage_enum, context),
                                     {"seed": seed, "copy": copy, "recorded_indices": permutation}))
            adapter = ADAPTER[stage]
            bucket = examples.setdefault((split, adapter), {})
            for order_kind, messages, answer, shuffle in variants:
                prompt_key = sha(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode())
                key = sha((prompt_key + answer).encode())
                seen = answers_by_prompt.setdefault((split, adapter, prompt_key), set())
                if key in bucket:
                    duplicates[f"{split}/{adapter}"] += 1
                    continue
                if seen:
                    conflicts[f"{split}/{adapter}"] += 1   # same prompt, another verified answer
                seen.add(answer)
                example = {"id": key[:16], "messages": messages + [{"role": "assistant", "content": answer}]}
                meta = {"id": key[:16], "scene_id": layout.scene_id, "run": str(run), "line": line_no,
                        "stage": stage, "target": record["target"], "candidate_order": order_kind,
                        "shuffle": shuffle,
                        "episode_id": record["meta"].get("episode_id"), "step": record["meta"].get("step"),
                        "selected_id": record["selected_id"], "valid_candidates": verification["valid_candidates"],
                        "teacher_reason": record["reason"], "teacher_likelihoods": record["likelihoods"]}
                bucket[key] = (example, meta)

    args.output.mkdir(parents=True)
    counts = {}
    for (split, adapter), bucket in sorted(examples.items()):
        folder = args.output / split
        folder.mkdir(exist_ok=True)
        with (folder / f"{adapter}.jsonl").open("x", encoding="utf-8") as data, \
                (folder / f"{adapter}.meta.jsonl").open("x", encoding="utf-8") as meta_file:
            for example, meta in bucket.values():
                data.write(json.dumps(example, ensure_ascii=False) + "\n")
                meta_file.write(json.dumps(meta, ensure_ascii=False) + "\n")
        counts[f"{split}/{adapter}"] = {
            "examples": len(bucket),
            "by_stage": dict(Counter(m["stage"] for _, m in bucket.values())),
            "by_candidate_order": dict(Counter(m["candidate_order"] for _, m in bucket.values()))}
    manifest = {
        "schema_version": "0.1", "purpose": "LoRA-SFT data (proposal 6-3)",
        "prompt_version": PROMPT_VERSION, "student_system_sha256": system_sha,
        "answer_format": '{"selected_id": "<alias>"} (athome.inference.llm_policy)',
        "chat_template_note": "Qwen3: train and serve without the thinking block (enable_thinking=False)",
        "adapters": ADAPTER, "split_config": str(args.split_config),
        "split_config_sha256": sha(args.split_config.read_bytes()), "scene_splits": scene_splits,
        "categories": str(args.categories) if args.categories else None,
        "candidate_order": {"train": f"recorded + {args.shuffled_copies} shuffled copies (seed {args.seed})",
                            "val_test": "recorded", "reference": "RankVicuna (Pradeep et al. 2023) data augmentation"},
        "categories_sha256": sha(args.categories.read_bytes()) if args.categories else None,
        "runs": runs, "counts": counts, "dropped": dict(dropped),
        "exact_duplicates_removed": dict(duplicates), "same_prompt_other_answer": dict(conflicts),
    }
    with (args.output / "manifest.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("장면 split:", scene_splits)
    print("예제 수:", {k: v["examples"] for k, v in counts.items()})
    print("제외:", dict(dropped), "중복 제거:", dict(duplicates), "같은 입력·다른 정답:", dict(conflicts))
    print("저장:", args.output)


if __name__ == "__main__":
    main()
