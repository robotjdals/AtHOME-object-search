"""Paired zero-shot comparison of planner LLMs on Teacher records.

    # 1. Freeze a stratified sample once (labeling keeps appending records)
    python3 scripts/compare_llm_models.py sample --records 'outputs/teacher_v5/*/queries.jsonl' \
        --out outputs/model_compare/planner_sample.jsonl
    # 2. Per model served on the LLM server (check /v1/models root first)
    python3 scripts/compare_llm_models.py run --sample outputs/model_compare/planner_sample.jsonl \
        --robot-config configs/robot/demo.yaml --base-url http://<host>:<port> \
        --served-root /data/models/Qwen3-4B --out outputs/model_compare/answers_qwen3-4b.jsonl
    # 3. Paired analysis with a criterion fixed before looking at results
    python3 scripts/compare_llm_models.py analyze --sample ... --a answers_A.jsonl --b answers_B.jsonl

Every sampled planner query is asked in its recorded candidate order and in a
seeded shuffled order (LLMs are sensitive to option order in multiple-choice
selection). Prompts are rebuilt with athome.inference.prompts.build_messages
from the recorded candidates and context; the current room comes from the
recorded Student prompt. Requests go through the robot's own LLMPolicy.

Metrics per answer: agreement with the Teacher label (primary, what SFT
imitates) and GT validity (a candidate containing the target). Decision:
switch from A to B only if B's Teacher agreement over all answers is at least
--min-gain percentage points higher and the exact McNemar test on the
discordant pairs gives p < --alpha; otherwise keep A.
Credentials come from --env-file, read by this tool.
"""

import argparse
import glob
import json
import random
import re
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from athome.inference.factory import _env_key, make_policy  # noqa: E402
from athome.search.policy import Candidate, PlanningContext, PolicyError, Stage  # noqa: E402

ORDERS = ("recorded", "shuffled")
STAGES = ("room", "workspace", "standalone")


def usable(r):
    return not r.get("single_candidate") and not r.get("error") and r.get("selected_alias")


def cmd_sample(args):
    records = []
    for path in sorted(glob.glob(args.records)):
        with open(path, encoding="utf-8") as f:
            records += [r for r in map(json.loads, f) if usable(r)]
    rng = random.Random(args.seed)
    sizes = dict(zip(STAGES, args.sizes))
    sample = []
    for stage in STAGES:
        pool = [r for r in records if r["stage"] == stage]
        sample += rng.sample(pool, min(sizes[stage], len(pool)))
    with args.out.open("x", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(sample)}문항 고정 ({sizes}), 원본 {len(records)}개 → {args.out}")


def load_sample(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def planner_inputs(record, order, index):
    user = record["student_messages"][1]["content"]
    marked = re.search(r"^- (C\d+) \(current room\)$", user, re.M)
    context = PlanningContext(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in record["context"].items()},
        current_room=record["aliases"][marked.group(1)] if marked else None)
    candidates = [Candidate(c["candidate_id"], c["cost"], c["info"]) for c in record["candidates"]]
    if order == "shuffled":
        random.Random(1000 + index).shuffle(candidates)
    return Stage(record["stage"]), record["target"], candidates, context


def served_roots(llm):
    request = urllib.request.Request(
        llm["base_url"].rstrip("/") + "/v1/models",
        headers={"Authorization": f"Bearer {_env_key(llm.get('api_key_env'))}"}
        if llm.get("api_key_env") else {})
    with urllib.request.urlopen(request, timeout=10) as response:
        return [m.get("root") for m in json.loads(response.read())["data"]]


def cmd_run(args):
    load_dotenv(args.env_file, override=False)
    llm = dict(yaml.safe_load(args.robot_config.read_text(encoding="utf-8"))["planner"]["llm"])
    if args.base_url:
        llm["base_url"] = args.base_url
    llm["models"] = {stage: args.model for stage in STAGES}      # zero-shot: base model only
    roots = served_roots(llm)
    if args.served_root not in roots:
        raise SystemExit(f"서버 모델 {roots} ≠ 기대 {args.served_root}: 비교에 섞이지 않게 중단")
    policy = make_policy({"type": "llm", "llm": llm})
    sample = load_sample(args.sample)

    def ask(job):
        index, order = job
        start = time.monotonic()
        try:
            output = policy.select(*planner_inputs(sample[index], order, index))
        except PolicyError:
            output = None
        return {"i": index, "order": order, "output": output,
                "latency_s": round(time.monotonic() - start, 4)}

    jobs = [(i, o) for i in range(len(sample)) for o in ORDERS]
    with ThreadPoolExecutor(args.workers) as pool:
        answers = list(pool.map(ask, jobs))
    with args.out.open("x", encoding="utf-8") as f:
        for a in answers:
            f.write(json.dumps({**a, "served_root": args.served_root}) + "\n")
    failed = sum(a["output"] is None for a in answers)
    print(f"{args.served_root}: {len(answers)}개 답, 실패 {failed}, "
          f"지연 중앙값 {statistics.median(a['latency_s'] for a in answers):.2f}s → {args.out}")


def mcnemar_p(only_a, only_b):
    """Exact two-sided McNemar (binomial) test on discordant pairs."""
    n, k = only_a + only_b, min(only_a, only_b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def load_answers(path):
    with open(path, encoding="utf-8") as f:
        # Earlier runs used "orig"/"shuf" for the two orders.
        rename = {"orig": "recorded", "shuf": "shuffled"}
        return {(a["i"], rename.get(a["order"], a["order"])): a["output"] for a in map(json.loads, f)}


def cmd_analyze(args):
    sample = load_sample(args.sample)
    a, b = load_answers(args.a), load_answers(args.b)
    if set(a) != set(b):
        raise SystemExit("두 답 파일의 문항이 다릅니다")
    checks = {
        "teacher": lambda r, x: x == r["selected_id"],
        "gt": lambda r, x: x in r["meta"]["verification"]["valid_candidates"],
    }
    report = {}
    for metric, ok in checks.items():
        rows = {}
        for key in sorted(a):
            r = sample[key[0]]
            ka, kb = ok(r, a[key]), ok(r, b[key])
            row = rows.setdefault(r["stage"], {"n": 0, "a": 0, "b": 0, "only_a": 0, "only_b": 0})
            row["n"] += 1
            row["a"] += ka
            row["b"] += kb
            row["only_a"] += ka and not kb
            row["only_b"] += kb and not ka
        rows["total"] = {k: sum(rows[s][k] for s in STAGES if s in rows) for k in ("n", "a", "b", "only_a", "only_b")}
        for row in rows.values():
            row["gain_pp"] = round(100 * (row["b"] - row["a"]) / row["n"], 1)
            row["p"] = round(mcnemar_p(row["only_a"], row["only_b"]), 4)
        report[metric] = rows
        print(f"[{'Teacher 일치' if metric == 'teacher' else 'GT 정답'}]  A={args.a.name}  B={args.b.name}")
        for stage, row in rows.items():
            print(f"  {stage:10s} A {row['a'] / row['n']:.1%}  B {row['b'] / row['n']:.1%}  "
                  f"차이 {row['gain_pp']:+.1f}%p  A만 {row['only_a']}  B만 {row['only_b']}  "
                  f"McNemar p={row['p']:.3f}  (n={row['n']})")
    for tag, answers in (("A", a), ("B", b)):
        same = sum(answers[(i, "recorded")] == answers[(i, "shuffled")] for i in range(len(sample)))
        print(f"순서를 바꿔도 같은 답 {tag}: {same / len(sample):.1%}")
    total = report["teacher"]["total"]
    switch = total["gain_pp"] >= args.min_gain and total["p"] < args.alpha
    print(f"판단 (Teacher 일치 합계 +{args.min_gain}%p 이상 & p<{args.alpha}): "
          f"{'B로 변경' if switch else 'A 유지'} (차이 {total['gain_pp']:+.1f}%p, p={total['p']:.3f})")
    if args.out:
        with args.out.open("x", encoding="utf-8") as f:
            json.dump({"a": str(args.a), "b": str(args.b), "min_gain_pp": args.min_gain,
                       "alpha": args.alpha, "switch_to_b": switch, "report": report}, f, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--records", required=True, help="Teacher queries.jsonl glob")
    s.add_argument("--sizes", type=int, nargs=3, default=[150, 100, 100], help="room workspace standalone")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--sample", type=Path, required=True)
    r.add_argument("--robot-config", type=Path, required=True, help="planner.llm만 읽음")
    r.add_argument("--base-url")
    r.add_argument("--model", default="qwen3-4b", help="서버의 기본 모델 이름")
    r.add_argument("--served-root", required=True, help="/v1/models의 root (다르면 중단)")
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--env-file", type=Path, default=ROOT / ".env")
    r.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--sample", type=Path, required=True)
    a.add_argument("--a", type=Path, required=True, help="현재 모델 답")
    a.add_argument("--b", type=Path, required=True, help="후보 모델 답")
    a.add_argument("--min-gain", type=float, default=5.0)
    a.add_argument("--alpha", type=float, default=0.05)
    a.add_argument("--out", type=Path)
    args = parser.parse_args()
    {"sample": cmd_sample, "run": cmd_run, "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__":
    main()
