import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/teacher"
SCENE = "wcojb4TFT35"
MASKED = ROOT / f"outputs/hm3d/{SCENE}/masked_inputs_v2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    text = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compact_room(room):
    objects = room["objects"]
    by_id = {o["object_id"]: o for o in objects}
    if len(by_id) != len(objects) or not objects:
        raise ValueError(f"{room['room_id']}: 빈 Room 또는 객체 ID 중복")

    compact_objects = []
    for obj in objects:
        child_tags = [
            by_id[cid]["semantic_tag"]
            for cid in obj["child_candidate_ids"]
        ]
        compact_objects.append({
            "id": obj["object_id"],
            "category": obj["semantic_tag"],
            "size_m": [round(v, 3) for v in obj["bbox"]["size"]],
            "child_candidate_counts": dict(
                sorted(Counter(child_tags).items())
            ),
        })

    return {
        "room_id": room["room_id"],
        "object_counts": dict(sorted(Counter(
            o["semantic_tag"] for o in objects
        ).items())),
        "objects": compact_objects,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", default=SCENE)
    parser.add_argument("--masked-dir", type=Path, default=MASKED)
    parser.add_argument("--labels", type=Path,
                        default=BASE / f"{SCENE}.semantic_labels.reviewed.json")
    parser.add_argument("--baseline-batch", type=Path,
                        default=BASE / f"{SCENE}.semantic.compact.batch.jsonl")
    parser.add_argument("--batch-output", type=Path,
                        default=BASE / f"{SCENE}.masked.semantic.batch.jsonl")
    parser.add_argument("--manifest-output", type=Path,
                        default=BASE / f"{SCENE}.masked.semantic.manifest.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/pilot_targets.json",
                        help="목표 범주 파일 (실행기는 장면별 <scene>.targets.json)")
    args = parser.parse_args()
    config = read(args.config)
    labels_path = args.labels
    labels = read(labels_path)
    if labels["status"] != "reviewed_for_graph_assembly":
        raise ValueError("기존 라벨 검토 상태를 확인하세요.")

    baseline = {}
    original_path = args.baseline_batch
    for line in original_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        body = record["body"]
        payload = json.loads(next(
            m["content"] for m in body["messages"] if m["role"] == "user"
        ))
        rid = payload["room_id"]
        if rid in baseline:
            raise ValueError(f"기존 Room 요청 중복: {rid}")
        baseline[rid] = body

    if set(baseline) != {r["room_id"] for r in labels["rooms"]}:
        raise ValueError("기존 요청과 검토 라벨의 Room 목록 불일치")

    requests = {}
    mappings = []
    reuse_count = 0
    changed_count = 0

    for target in config["target_categories"]:
        masked_path = args.masked_dir / f"{target}.semantic_inputs.json"
        masked = read(masked_path)

        room_ids = [r["room_id"] for r in masked["rooms"]]
        if len(room_ids) != len(set(room_ids)) or set(room_ids) != set(baseline):
            raise ValueError(f"{target}: Room 목록 불일치")

        for room in masked["rooms"]:
            rid = room["room_id"]
            payload = compact_room(room)
            body = deepcopy(baseline[rid])

            for message in body["messages"]:
                if message["role"] == "user":
                    message["content"] = json.dumps(
                        payload, ensure_ascii=False,
                        separators=(",", ":"), allow_nan=False,
                    )

            schema = body["response_format"]["json_schema"]["schema"]
            schema["properties"]["room_id"]["enum"] = [rid]
            schema["properties"]["workspace_sources"]["items"][
                "properties"
            ]["source_object_id"]["enum"] = sorted(
                obj["id"] for obj in payload["objects"]
            )

            fingerprint = digest(body)
            entry = {
                "target_category": target,
                "room_id": rid,
                "request_sha256": fingerprint,
                "masked_input_sha256": digest(masked),
            }

            if fingerprint == digest(baseline[rid]):
                entry["result_source"] = "reviewed_baseline"
                reuse_count += 1
            else:
                changed_count += 1
                custom_id = f"masked-{fingerprint}"
                entry["result_source"] = "new_batch"
                entry["custom_id"] = custom_id
                requests.setdefault(custom_id, {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                })

            mappings.append(entry)

    batch_path = args.batch_output
    manifest_path = args.manifest_output

    batch_path.write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
            for r in requests.values()
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps({
            "scene_id": args.scene_id,
            "baseline_labels_path": str(labels_path),
            "baseline_labels_sha256": digest(labels),
            "entries": mappings,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    estimated = 0
    for request in requests.values():
        body = request["body"]
        enc = tiktoken.encoding_for_model(body["model"])
        texts = [m["content"] for m in body["messages"]]
        texts.append(json.dumps(
            body["response_format"], ensure_ascii=False,
            separators=(",", ":"),
        ))
        estimated += sum(
            len(enc.encode(t, disallowed_special=())) for t in texts
        )

    print("=== マスキング後のリクエスト ===".replace(
        "マスキング後のリクエスト", "마스킹 후 요청 구성"
    ))
    print("목표–Room 조합:", len(mappings))
    print("기존 검토 결과 재사용:", reuse_count)
    print("새 라벨이 필요한 조합:", changed_count)
    print("중복 제거 후 Batch 요청:", len(requests))
    print(f"입력 텍스트 토큰 추정: {estimated:,}")
    print("Batch 파일:", batch_path)
    print("결과 연결 목록:", manifest_path)
    print("API 제출은 하지 않았습니다.")


if __name__ == "__main__":
    main()
