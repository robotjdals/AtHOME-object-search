"""Batch for measuring LLM semantic-label stability (no graph is changed).

Resends existing compact requests with the same prompt, schema and model,
but samples ``n`` completions at ``temperature`` so per-object selection
frequencies can be estimated. Each input set gets a tag in the custom_id.
"""
import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True,
                        help="TAG=compact.batch.jsonl (여러 번 지정 가능)")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    lines = []
    for item in args.input:
        tag, path = item.split("=", 1)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            body = copy.deepcopy(record["body"])
            body["n"] = args.n
            body["temperature"] = args.temperature
            lines.append(json.dumps({
                "custom_id": f"stability:{tag}:{record['custom_id']}",
                "method": "POST", "url": record["url"], "body": body,
            }, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"요청 {len(lines)}개 (각 n={args.n}, temperature={args.temperature}) → {args.output}")


if __name__ == "__main__":
    main()
