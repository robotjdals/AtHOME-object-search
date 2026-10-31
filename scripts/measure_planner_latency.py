"""Decision latency of the GPT-4.1 Teacher and the served Student on the same
planner decisions (OpenAI API calls: announce the cost before running).

Decisions are drawn (seeded) from the Teacher label records
(outputs/teacher_v5/*/queries.jsonl), with more than one candidate. Each is
sent as recorded: the Teacher messages and likelihood schema to the OpenAI API
exactly as in label generation (ChatJSON, temperature 0), and the Student
messages to the robot's vLLM server with the request of
athome.inference.llm_policy.LLMPolicy (adapter of the stage, alias enum,
temperature 0, max_tokens 32). Calls alternate Teacher / Student, one at a
time; wall-clock time per request, plus one untimed Student warm-up. The Teacher answers only
the first --teacher-decisions of them (API cost); keys come from .env.

    python scripts/measure_planner_latency.py --decisions 100 --teacher-decisions 50 \
        --planner-url http://127.0.0.1:16007 --output outputs/paper/planner_latency_n100.json
"""
import argparse
import glob
import json
import os
import random
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
import yaml

from athome.inference.chat import ChatJSON
from athome.inference.http import post_json
from athome.inference.prompts import teacher_schema
from athome.scene_graph.semantic_labeling import MODEL

ROOT = Path(__file__).resolve().parents[1]


def draw(n, seed):
    records = []
    for path in sorted(glob.glob(str(ROOT / "outputs/teacher_v5/*/queries.jsonl"))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if not r["single_candidate"] and not r["error"] and len(r["aliases"]) > 1:
                records.append(r)
    return random.Random(seed).sample(records, n), len(records)


def student_body(record, models):
    return {"model": models[record["stage"]], "messages": record["student_messages"], "temperature": 0.0,
            "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "selection", "schema": {"type": "object", "properties": {
                    "selected_id": {"type": "string", "enum": list(record["aliases"])}},
                    "required": ["selected_id"]}}}}


def summary(values):
    values = sorted(values)
    return {"n": len(values), "median_s": statistics.median(values), "mean_s": statistics.fmean(values),
            "p90_s": values[int(0.9 * (len(values) - 1))], "min_s": values[0], "max_s": values[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decisions", type=int, default=10)
    parser.add_argument("--teacher-decisions", type=int, help="Teacher에 보낼 결정 수 (기본: 전부)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planner-url", required=True, help="로봇이 쓰는 vLLM 서버 (터널 주소)")
    parser.add_argument("--robot-config", type=Path, default=ROOT / "configs/robot/demo.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    n_teacher = args.decisions if args.teacher_decisions is None else args.teacher_decisions
    llm = yaml.safe_load(args.robot_config.read_text(encoding="utf-8"))["planner"]["llm"]
    key = os.environ.get(llm["api_key_env"]) if llm.get("api_key_env") else None
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    url = args.planner_url.rstrip("/") + "/v1/chat/completions"
    teacher = ChatJSON("https://api.openai.com", MODEL, api_key_env="OPENAI_API_KEY", timeout=60.0,
                       schema_name="teacher_likelihoods")
    records, pool = draw(args.decisions, args.seed)
    post_json(url, student_body(records[0], llm["models"]), 30.0, headers)      # warm-up, untimed
    rows = []
    for k, r in enumerate(records, 1):
        t_teacher = usage = None
        if k <= n_teacher:
            t0 = time.perf_counter()
            teacher(r["teacher_messages"], teacher_schema(r["aliases"]))
            t_teacher, usage = time.perf_counter() - t0, teacher.last_usage
        t0 = time.perf_counter()
        reply = post_json(url, student_body(r, llm["models"]), 30.0, headers)
        t_student = time.perf_counter() - t0
        rows.append({"stage": r["stage"], "candidates": len(r["aliases"]),
                     "teacher_s": t_teacher, "teacher_usage": usage,
                     "student_s": t_student, "student_usage": reply.get("usage")})
        told = "-" if usage is None else f"{t_teacher:5.2f}s ({usage['completion_tokens']} 토큰)"
        print(f"[{k}/{len(records)}] {r['stage']:10s} 후보 {len(r['aliases']):2d}  Teacher {told}  "
              f"Student {t_student:5.2f}s", flush=True)
    asked = [x for x in rows if x["teacher_s"] is not None]
    result = {"model_teacher": MODEL, "planner_url_models": llm["models"], "decisions": len(rows),
              "drawn_from": pool, "seed": args.seed,
              "teacher": {**summary([x["teacher_s"] for x in asked]),
                          "mean_completion_tokens": statistics.fmean(x["teacher_usage"]["completion_tokens"] for x in asked)}
                         if asked else None,
              "student": {**summary([x["student_s"] for x in rows]),
                          "mean_completion_tokens": statistics.fmean((x["student_usage"] or {}).get("completion_tokens", 0)
                                                                     for x in rows)},
              "teacher_api_usage": teacher.usage, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps({k: result[k] for k in ("teacher", "student", "teacher_api_usage")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
