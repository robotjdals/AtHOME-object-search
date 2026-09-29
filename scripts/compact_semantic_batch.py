import argparse
import json
from pathlib import Path

from athome.scene_graph.semantic_labeling import PROMPTS, PROTOCOLS, apply_protocol, compact_room

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/teacher"
SOURCE = BASE / "wcojb4TFT35.semantic.batch.jsonl"
OUTPUT = BASE / "wcojb4TFT35.semantic.compact.batch.jsonl"

# Prompt and compact payload are shared with the real-robot labeling code.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--protocol", required=True, choices=sorted(PROTOCOLS),
                        help="v1: 최초 HM3D 라벨 재현, v2: 현재 기준(수납가구 윗면만 + 9회 다수결)")
    args = parser.parse_args()
    run(args.input, args.output, args.protocol)


def run(SOURCE, OUTPUT, protocol):
    PROMPT = PROMPTS[PROTOCOLS[protocol]["prompt"]]
    records = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    total_objects = 0
    seen = set()
    lines = []

    for record in records:
        cid = record["custom_id"]
        if cid in seen:
            raise ValueError(f"Duplicate custom_id: {cid}")
        seen.add(cid)

        body = record["body"]
        users = [m for m in body["messages"] if m["role"] == "user"]
        if len(users) != 1:
            raise ValueError(f"{cid}: Expected one user message")

        room = json.loads(users[0]["content"])
        objects = room["objects"]
        by_id = {obj["object_id"]: obj for obj in objects}
        if len(by_id) != len(objects):
            raise ValueError(f"{cid}: Duplicate object IDs")

        compact = compact_room(room)

        # Preserve the original output schema and its allowed object IDs.
        schema = body["response_format"]["json_schema"]["schema"]
        allowed = schema["properties"]["workspace_sources"]["items"][
            "properties"
        ]["source_object_id"]["enum"]
        if set(allowed) != set(by_id):
            raise ValueError(f"{cid}: Schema object IDs do not match input")

        body["messages"] = [
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    compact, ensure_ascii=False,
                    separators=(",", ":"), allow_nan=False,
                ),
            },
        ]

        apply_protocol(body, protocol)
        total_objects += len(objects)
        lines.append(json.dumps(
            record, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        ))
        print(f"{room['room_id']}: 객체 {len(objects)}개 유지")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    before = SOURCE.stat().st_size
    after = OUTPUT.stat().st_size
    print("\n=== 압축 완료 ===")
    print("요청 수:", len(records))
    print("유지한 Object 수:", total_objects)
    print(f"파일 크기: {before:,} → {after:,} bytes")
    print(f"파일 크기 감소율: {(1 - after / before) * 100:.1f}%")
    print("저장 위치:", OUTPUT)
    print("API 제출은 하지 않았습니다.")


if __name__ == "__main__":
    main()
