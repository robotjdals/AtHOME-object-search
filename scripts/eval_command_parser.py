"""Evaluate natural-language command parsing against held-out cases.

    python3 scripts/eval_command_parser.py --robot-config configs/robot/demo.yaml \
        --base-url http://<host>:<port> --model qwen3-4b --output out.json

The parser is the robot's own (athome.inference.factory.make_command_parser,
same prompt, schema and target vocabulary). Credentials come from --env-file
(read by this tool; never sourced into the shell). Only the command_parser
and scene_graph.target_categories sections of the robot config are read.
Cases: configs/eval/command_parser_cases.json.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from athome.inference.command_parser import COMMAND_PROMPT_VERSION, CommandParseError  # noqa: E402
from athome.inference.factory import make_command_parser  # noqa: E402
from athome.scene_graph.vocabulary import Vocabulary  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--base-url", help="command_parser.base_url 대신 사용")
    parser.add_argument("--model", help="command_parser.model 대신 사용")
    parser.add_argument("--api-key-env", help="command_parser.api_key_env 대신 사용")
    parser.add_argument("--extra-body", type=json.loads,
                        help='command_parser.extra_body 대신 사용 (JSON, 예: vLLM Qwen3 thinking 끄기)')
    parser.add_argument("--target-categories", type=Path,
                        help="scene_graph.target_categories 대신 사용 (어휘 목록 비교용)")
    parser.add_argument("--cases", type=Path, default=ROOT / "configs/eval/command_parser_cases.json")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--output", type=Path, help="문항별 결과 JSON (새 파일)")
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)     # the process environment wins
    raw = yaml.safe_load(args.robot_config.read_text(encoding="utf-8"))
    section = dict(raw.get("command_parser") or {})
    for key, value in (("base_url", args.base_url), ("model", args.model),
                       ("api_key_env", args.api_key_env), ("extra_body", args.extra_body)):
        if value is not None:
            section[key] = value
    categories_path = (raw.get("scene_graph") or {}).get("target_categories")
    vocabulary = Vocabulary.load(
        args.target_categories if args.target_categories
        else (args.robot_config.parent / categories_path).resolve() if categories_path else None)
    parse = make_command_parser(section, vocabulary.categories)

    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    results, latencies, by_kind = [], [], {}
    for case in cases:
        start = time.monotonic()
        try:
            output = parse(case["command"])
        except CommandParseError as e:
            output = [] if "찾을 물체가 없음" in str(e) else f"ERROR {e}"
        latencies.append(time.monotonic() - start)
        expected = case["expected"]
        correct = (isinstance(output, list) and len(output) == len(expected)
                   and all(o in accepted for o, accepted in zip(output, expected)))
        kind = by_kind.setdefault(case["kind"], [0, 0])
        kind[0] += correct
        kind[1] += 1
        results.append({**case, "output": output, "correct": correct})

    wrong = [r for r in results if not r["correct"]]
    print(f"{section.get('model')} / 프롬프트 {COMMAND_PROMPT_VERSION}: "
          f"{len(cases)}개 중 정답 {len(cases) - len(wrong)}, "
          f"지연 중앙값 {statistics.median(latencies):.2f}s 최대 {max(latencies):.2f}s")
    print("유형별:", {k: f"{a}/{b}" for k, (a, b) in by_kind.items()})
    for r in wrong:
        print(f"- [{r['kind']}] {r['command']!r}\n    기대 {[e[0] for e in r['expected']]}\n    출력 {r['output']}")
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            json.dump({"model": section.get("model"), "prompt_version": COMMAND_PROMPT_VERSION,
                       "results": results}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
