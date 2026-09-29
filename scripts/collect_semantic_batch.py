import argparse
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from openai import OpenAI

from athome.scene_graph.label_voting import aggregate, sample_problem
from athome.scene_graph.semantic_labeling import PROTOCOLS, ROOM_LABELS, protocol_of

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/teacher"
PREFIX = "wcojb4TFT35.semantic.compact.batch"


def save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    global BASE, PREFIX
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=BASE / f"{PREFIX}.state.json",
                        help="submit_batch.py state 파일; 결과는 같은 폴더에 저장")
    parser.add_argument("--labels-output", type=Path,
                        default=BASE / "wcojb4TFT35.semantic_labels.review.json")
    args = parser.parse_args()
    if not args.state.name.endswith(".state.json"):
        raise SystemExit("--state는 *.state.json 이어야 합니다.")
    BASE, PREFIX = args.state.parent, args.state.name[:-len(".state.json")]
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
    dropped_samples = []   # unreadable samples left out of the vote
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

            request_body = expected[cid]["body"]
            protocol = protocol_of(request_body)
            choices = body["choices"]
            if len(choices) != PROTOCOLS[protocol]["n"]:
                raise ValueError("예상하지 않은 응답 개수")
            schema = request_body["response_format"]["json_schema"]["schema"]
            payload = json.loads(next(
                m["content"] for m in request_body["messages"]
                if m["role"] == "user"
            ))
            objects = {o["id"]: o for o in payload["objects"]}

            # Self-consistency (Wang et al. 2023): samples that cannot be
            # read are left out of the vote; min_votes stays the same count.
            samples = []
            for choice in choices:
                message = choice["message"]
                problem = (f"finish_reason={choice['finish_reason']}" if choice["finish_reason"] != "stop" else
                           f"refusal={message['refusal']}" if message.get("refusal") else "")
                if not problem:
                    try:
                        label = json.loads(message["content"])
                        Draft202012Validator(schema).validate(label)
                        problem = sample_problem(label)
                    except Exception as exc:  # noqa: BLE001 - recorded, sample left out
                        problem = f"{type(exc).__name__}: {exc}"
                if problem:
                    dropped_samples.append(f"{cid}: {problem}")
                    continue
                for item in label["workspace_sources"]:
                    oid = item["source_object_id"]
                    category = objects[oid]["category"]
                    if category.strip().lower() in {
                        "floor", "wall", "ceiling", "unknown"
                    }:
                        warnings.append(f"{cid}/{oid}: Source 선정 검토 필요")
                samples.append(label)
            if len(samples) < PROTOCOLS[protocol]["min_votes"]:
                raise ValueError(f"유효 응답 {len(samples)}개 < min_votes")
            labels[cid] = aggregate(samples, PROTOCOLS[protocol]["min_votes"], ROOM_LABELS)
            labels[cid]["protocol"] = protocol

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
        "dropped_samples": dropped_samples,
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

    output = args.labels_output
    save_json(output, {
        "status": "pending_semantic_review",
        "batch_id": batch.id,
        "input_sha256": state["input_sha256"],
        "protocol": sorted({labels[cid]["protocol"] for cid in expected}),
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
            votes = label["votes"]
            print(f"  {oid} ({objects[oid]['category']})"
                  f" -> {item['function_label']}"
                  f" [{votes['sources'][oid]}/{votes['samples']}]")

    print("\n검토용 라벨 저장:", output)


if __name__ == "__main__":
    main()
