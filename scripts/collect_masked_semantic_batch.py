import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator
from openai import OpenAI

from athome.scene_graph.label_voting import aggregate
from athome.scene_graph.semantic_labeling import PROTOCOLS, ROOM_LABELS, protocol_of

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/teacher"
SCENE = "wcojb4TFT35"
PREFIX = f"{SCENE}.masked.semantic"
MASKED = ROOT / f"outputs/hm3d/{SCENE}/masked_inputs_v2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def digest(value):
    text = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    global BASE, PREFIX
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=BASE / f"{PREFIX}.batch.state.json",
                        help="<prefix>.batch.state.json; 결과는 같은 폴더에 저장")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--masked-dir", type=Path, default=MASKED)
    parser.add_argument("--labels-dir", type=Path, default=BASE / f"{SCENE}.masked_labels")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/pilot_targets.json",
                        help="목표 범주 파일 (실행기는 장면별 <scene>.targets.json)")
    args = parser.parse_args()
    suffix = ".batch.state.json"
    if not args.state.name.endswith(suffix):
        raise SystemExit(f"--state는 *{suffix} 이어야 합니다.")
    BASE, PREFIX = args.state.parent, args.state.name[:-len(suffix)]
    manifest = read(args.manifest or BASE / f"{PREFIX}.manifest.json")
    baseline = read(Path(manifest["baseline_labels_path"]))
    if digest(baseline) != manifest["baseline_labels_sha256"]:
        raise ValueError("기존 검토 라벨이 변경됐습니다.")
    baseline_rooms = {r["room_id"]: r for r in baseline["rooms"]}

    # Nothing to relabel (no target object changes any room input): every
    # combination reuses the reviewed labels and no batch is submitted.
    needs_batch = any(e["result_source"] == "new_batch" for e in manifest["entries"])
    requests, results, batch_id = {}, {}, None
    if needs_batch:
        state = read(args.state)
        raw = Path(state["input_path"]).read_bytes()

        if hashlib.sha256(raw).hexdigest() != state["input_sha256"]:
            raise ValueError("제출한 입력 파일이 변경됐습니다.")

        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        requests = {r["custom_id"]: r for r in records}
        if len(requests) != len(records):
            raise ValueError("입력 요청 ID 중복")

        client = OpenAI()
        batch = client.batches.retrieve(state["batch_id"])
        save(BASE / f"{PREFIX}.result_metadata.json", batch.model_dump())

        if batch.status != "completed":
            raise SystemExit(f"아직 완료되지 않았습니다: {batch.status}")

        if batch.error_file_id:
            error_text = client.files.content(batch.error_file_id).text
            (BASE / f"{PREFIX}.errors.jsonl").write_text(
                error_text, encoding="utf-8"
            )
        if not batch.output_file_id:
            raise ValueError("결과 파일이 없습니다.")

        text = client.files.content(batch.output_file_id).text
        (BASE / f"{PREFIX}.output.jsonl").write_text(text, encoding="utf-8")

        results = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row["custom_id"]
            if cid not in requests or cid in results:
                raise ValueError(f"{cid}: 알 수 없거나 중복된 요청 ID")
            if row.get("error"):
                raise ValueError(f"{cid}: {row['error']}")

            response = row["response"]
            if response["status_code"] != 200:
                raise ValueError(f"{cid}: HTTP 오류")

            protocol = protocol_of(requests[cid]["body"])
            choices = response["body"]["choices"]
            if len(choices) != PROTOCOLS[protocol]["n"]:
                raise ValueError(f"{cid}: 응답 개수 오류")
            schema = requests[cid]["body"]["response_format"]["json_schema"]["schema"]
            samples = []
            for choice in choices:
                if choice["finish_reason"] != "stop":
                    raise ValueError(f"{cid}: 응답이 정상 종료되지 않음")
                if choice["message"].get("refusal"):
                    raise ValueError(f"{cid}: 응답 거절")

                label = json.loads(choice["message"]["content"])
                Draft202012Validator(schema).validate(label)

                ids = [s["source_object_id"] for s in label["workspace_sources"]]
                if len(ids) != len(set(ids)):
                    raise ValueError(f"{cid}: Source 중복")
                for source in label["workspace_sources"]:
                    if not re.fullmatch(
                        r"[a-z]+(?:_[a-z]+)*", source["function_label"]
                    ):
                        raise ValueError(f"{cid}: 기능 라벨 형식 오류")
                samples.append(label)
            results[cid] = aggregate(samples, PROTOCOLS[protocol]["min_votes"], ROOM_LABELS)
            results[cid]["protocol"] = protocol

        if set(results) != set(requests):
            raise ValueError("누락된 응답이 있습니다.")
        batch_id = batch.id

    targets = read(args.config)["target_categories"]
    masked_data = {
        t: read(args.masked_dir / f"{t}.semantic_inputs.json") for t in targets
    }
    expected_pairs = {
        (t, r["room_id"])
        for t, data in masked_data.items() for r in data["rooms"]
    }
    seen = set()
    merged = {t: [] for t in targets}
    changes = []
    warnings = []

    for entry in manifest["entries"]:
        target, rid = entry["target_category"], entry["room_id"]
        pair = (target, rid)
        if pair not in expected_pairs or pair in seen:
            raise ValueError("결과 연결 목록에 중복 또는 잘못된 조합이 있습니다.")
        seen.add(pair)

        if digest(masked_data[target]) != entry["masked_input_sha256"]:
            raise ValueError(f"{target}: 마스킹 입력이 변경됐습니다.")

        if entry["result_source"] == "reviewed_baseline":
            label = deepcopy(baseline_rooms[rid])
        elif entry["result_source"] == "new_batch":
            cid = entry["custom_id"]
            if digest(requests[cid]["body"]) != entry["request_sha256"]:
                raise ValueError(f"{target}/{rid}: 요청 해시 불일치")
            label = deepcopy(results[cid])
        else:
            raise ValueError("알 수 없는 결과 출처")

        if label["room_id"] != rid:
            raise ValueError(f"{target}/{rid}: 라벨 Room 불일치")

        room = next(r for r in masked_data[target]["rooms"] if r["room_id"] == rid)
        objects = {o["object_id"]: o for o in room["objects"]}
        for source in label["workspace_sources"]:
            oid = source["source_object_id"]
            if oid not in objects:
                raise ValueError(f"{target}/{rid}: 제거된 객체를 Source로 선정")
            category = " ".join(objects[oid]["semantic_tag"].casefold().split())
            if category in {"wall", "floor", "ceiling", "staircase wall", "tray"}:
                warnings.append(f"{target}/{rid}: Source 검토 필요 — {oid}")

        old = baseline_rooms[rid]
        old_sources = {
            s["source_object_id"]: s["function_label"]
            for s in old["workspace_sources"]
        }
        new_sources = {
            s["source_object_id"]: s["function_label"]
            for s in label["workspace_sources"]
        }

        if entry["result_source"] == "new_batch":
            changes.append({
                "target": target,
                "room_id": rid,
                "room_label_before": old["room_label"],
                "room_label_after": label["room_label"],
                "added_sources": sorted(new_sources.keys() - old_sources.keys()),
                "removed_sources": sorted(old_sources.keys() - new_sources.keys()),
                "function_changes": {
                    oid: [old_sources[oid], new_sources[oid]]
                    for oid in old_sources.keys() & new_sources.keys()
                    if old_sources[oid] != new_sources[oid]
                },
            })

        label["label_origin"] = entry["result_source"]
        merged[target].append(label)

    if seen != expected_pairs:
        raise ValueError("목표–Room 조합이 누락됐습니다.")

    output_dir = args.labels_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for target, rooms in merged.items():
        save(output_dir / f"{target}.review.json", {
            "status": "pending_semantic_review",
            "target_category": target,
            "batch_id": batch_id,
            "masked_input_sha256": digest(masked_data[target]),
            "rooms": rooms,
        })

    save(BASE / f"{PREFIX}.review_report.json", {
        "validated_requests": len(results),
        "merged_room_count": len(seen),
        "warnings": warnings,
        "changes": changes,
    })

    print("=== 마스킹 라벨 검증·병합 ===")
    print("새 응답 검증 통과:", len(results))
    print("병합한 목표–Room 조합:", len(seen))
    print("생성한 목표별 라벨 파일:", len(merged))
    print("검토 경고:", len(warnings))

    print("\n=== 기존 라벨 대비 변경 ===")
    for change in changes:
        changed = (
            change["room_label_before"] != change["room_label_after"]
            or change["added_sources"]
            or change["removed_sources"]
            or change["function_changes"]
        )
        if changed:
            print(json.dumps(change, ensure_ascii=False))

    for warning in warnings:
        print("검토:", warning)

    print("\n검토용 라벨 저장:", output_dir)
    print("아직 그래프에 반영하지 않았습니다.")


if __name__ == "__main__":
    main()
