import hashlib
import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/teacher"
PREFIX = "wcojb4TFT35.semantic.compact.batch"


def save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    state = json.loads(
        (BASE / f"{PREFIX}.state.json").read_text(encoding="utf-8")
    )
    source = Path(state["input_path"])
    raw_input = source.read_bytes()
    if hashlib.sha256(raw_input).hexdigest() != state["input_sha256"]:
        raise SystemExit("제출 이후 입력 파일이 변경됐습니다.")

    requests = [
        json.loads(line)
        for line in raw_input.splitlines() if line.strip()
    ]
    expected = {r["custom_id"]: r for r in requests}
    if len(expected) != len(requests):
        raise SystemExit("입력 custom_id 중복")

    client = OpenAI()
    batch = client.batches.retrieve(state["batch_id"])
    save_json(BASE / f"{PREFIX}.result_metadata.json", batch.model_dump())

    print("Status:", batch.status)
    if batch.status != "completed":
        raise SystemExit("아직 완료되지 않았습니다.")

    if batch.error_file_id:
        text = client.files.content(batch.error_file_id).text
        (BASE / f"{PREFIX}.errors.jsonl").write_text(text, encoding="utf-8")

    if not batch.output_file_id:
        raise SystemExit("출력 파일이 없습니다.")

    text = client.files.content(batch.output_file_id).text
    (BASE / f"{PREFIX}.output.jsonl").write_text(text, encoding="utf-8")

    errors = []
    warnings = []
    seen = set()
    labels = {}
    input_tokens = 0
    output_tokens = 0

    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row.get("custom_id")

        if cid not in expected or cid in seen:
            errors.append(f"{cid}: 알 수 없거나 중복된 custom_id")
            continue
        seen.add(cid)

        try:
            if row.get("error"):
                raise ValueError(str(row["error"]))

            response = row["response"]
            if response["status_code"] != 200:
                raise ValueError(f"HTTP {response['status_code']}")

            body = response["body"]
            usage = body.get("usage") or {}
            input_tokens += usage.get("prompt_tokens", 0)
            output_tokens += usage.get("completion_tokens", 0)

            choices = body["choices"]
            if len(choices) != 1:
                raise ValueError("예상하지 않은 응답 개수")
            choice = choices[0]
            message = choice["message"]

            if choice["finish_reason"] != "stop":
                raise ValueError(f"finish_reason={choice['finish_reason']}")
            if message.get("refusal"):
                raise ValueError(f"refusal={message['refusal']}")

            label = json.loads(message["content"])
            request_body = expected[cid]["body"]
            schema = request_body["response_format"]["json_schema"]["schema"]
            Draft202012Validator(schema).validate(label)

            payload = json.loads(next(
                m["content"] for m in request_body["messages"]
                if m["role"] == "user"
            ))
            objects = {o["id"]: o for o in payload["objects"]}

            sources = label["workspace_sources"]
            ids = [s["source_object_id"] for s in sources]
            if len(ids) != len(set(ids)):
                raise ValueError("Source ID 중복")

            for item in sources:
                oid = item["source_object_id"]
                function = item["function_label"]
                if not re.fullmatch(r"[a-z]+(?:_[a-z]+)*", function):
                    raise ValueError(f"{oid}: function_label 형식 오류")

                category = objects[oid]["category"]
                if category.strip().lower() in {
                    "floor", "wall", "ceiling", "unknown"
                }:
                    warnings.append(f"{cid}/{oid}: Source 선정 검토 필요")

            labels[cid] = label

        except (ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append(f"{cid}: {exc}")
        except Exception as exc:
            # Includes JSON Schema validation failures.
            errors.append(f"{cid}: {type(exc).__name__}: {exc}")

    for cid in sorted(set(expected) - seen):
        errors.append(f"{cid}: 응답 누락")

    report = {
        "batch_id": batch.id,
        "expected": len(expected),
        "valid": len(labels),
        "errors": errors,
        "warnings": warnings,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
    save_json(BASE / f"{PREFIX}.validation.json", report)

    print("\n=== 결과 검증 ===")
    print("예상 Room 수:", len(expected))
    print("형식 검증 통과:", len(labels))
    print("오류 수:", len(errors))
    print("검토 경고 수:", len(warnings))
    print("실제 입력 토큰:", input_tokens)
    print("실제 출력 토큰:", output_tokens)

    for entry in errors + warnings:
        print("-", entry)

    if errors:
        raise SystemExit("오류를 확인한 뒤 진행하세요. 자동 재제출하지 않습니다.")

    output = BASE / "wcojb4TFT35.semantic_labels.review.json"
    save_json(output, {
        "status": "pending_semantic_review",
        "batch_id": batch.id,
        "input_sha256": state["input_sha256"],
        "rooms": [labels[cid] for cid in expected],
    })

    print("\n=== Room별 선정 결과 ===")
    for cid, request in expected.items():
        label = labels[cid]
        payload = json.loads(next(
            m["content"] for m in request["body"]["messages"]
            if m["role"] == "user"
        ))
        objects = {o["id"]: o for o in payload["objects"]}
        sources = label["workspace_sources"]

        print(f"\n{label['room_id']}: {label['room_label']}, "
              f"Source {len(sources)}개")
        for item in sources:
            oid = item["source_object_id"]
            print(f"  {oid} ({objects[oid]['category']})"
                  f" -> {item['function_label']}")

    print("\n검토용 라벨 저장:", output)


if __name__ == "__main__":
    main()
