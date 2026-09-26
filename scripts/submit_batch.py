import argparse
import hashlib
import json
from pathlib import Path

from openai import OpenAI


def save(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Batch 제출 및 상태 조회")
    parser.add_argument(
        "--input", required=True, type=Path,
        help="제출할 Batch JSONL 파일 경로",
    )
    args = parser.parse_args()

    source = args.input.resolve()
    state_path = source.with_suffix(".state.json")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    client = OpenAI(max_retries=0)

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

        if state["input_sha256"] != digest:
            raise SystemExit(
                "제출 이후 입력 파일이 변경됐습니다. 기존 작업을 확인하세요."
            )

        if not state.get("batch_id"):
            raise SystemExit(
                "이전 제출 기록에 Batch ID가 없습니다. "
                "중복 제출 방지를 위해 중단합니다. "
                "기록 파일을 삭제하지 말고 작업 상태를 확인하세요."
            )

        batch = client.batches.retrieve(state["batch_id"])
        print("기존 작업을 조회했습니다.")

    else:
        records = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip()
        ]
        ids = [record["custom_id"] for record in records]

        if not records or len(ids) != len(set(ids)):
            raise SystemExit("요청이 없거나 custom_id가 중복됐습니다.")

        if any(
            record["method"] != "POST"
            or record["url"] != "/v1/chat/completions"
            for record in records
        ):
            raise SystemExit("요청 방식 또는 API 경로가 올바르지 않습니다.")

        state = {
            "input_path": str(source),
            "input_sha256": digest,
            "request_count": len(records),
            "input_file_id": None,
            "batch_id": None,
        }

        # 원격 요청 전에 기록을 생성하여 중복 제출을 방지합니다.
        with state_path.open("x", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)

        with source.open("rb") as file:
            uploaded = client.files.create(file=file, purpose="batch")

        state["input_file_id"] = uploaded.id
        save(state_path, state)

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"input_sha256": digest},
        )

        print("생성된 Batch ID:", batch.id, flush=True)
        state["batch_id"] = batch.id
        save(state_path, state)
        print("제출이 완료됐습니다.")

    status_names = {
        "validating": "입력 검사 중",
        "in_progress": "처리 중",
        "finalizing": "결과 정리 중",
        "completed": "완료",
        "failed": "실패",
        "expired": "처리 기한 만료",
        "cancelling": "취소 중",
        "cancelled": "취소됨",
    }

    print("Batch ID:", batch.id)
    print("상태:", status_names.get(batch.status, batch.status))

    if batch.request_counts:
        counts = batch.request_counts
        print(
            f"요청 수: 전체 {counts.total}, "
            f"완료 {counts.completed}, 실패 {counts.failed}"
        )

    if batch.errors:
        print("API 오류 상세:", batch.errors.model_dump())

    print("작업 기록:", state_path)


if __name__ == "__main__":
    main()
