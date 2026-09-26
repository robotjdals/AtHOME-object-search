import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENE = "wcojb4TFT35"

INPUT = ROOT / f"outputs/hm3d/{SCENE}.semantic_inputs.json"
BATCH = ROOT / f"outputs/teacher/{SCENE}.semantic.batch.jsonl"

errors = []


def check(condition, message):
    if not condition:
        errors.append(message)


def vector(value):
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and math.isfinite(x)
            for x in value
        )
    )


def validate_bbox(obj, prefix):
    bbox = obj.get("bbox")
    if not isinstance(bbox, dict):
        errors.append(f"{prefix}: bbox 없음")
        return

    keys = ("center", "size", "min", "max")
    if not all(vector(bbox.get(k)) for k in keys):
        errors.append(f"{prefix}: bbox 벡터 형식 또는 유한성 오류")
        return

    for axis in range(3):
        lo = bbox["min"][axis]
        hi = bbox["max"][axis]
        size = bbox["size"][axis]
        center = bbox["center"][axis]

        check(size > 0 and hi > lo,
              f"{prefix}: 축 {axis} 크기가 양수가 아님")
        check(math.isclose(size, hi - lo, abs_tol=1e-5, rel_tol=1e-5),
              f"{prefix}: 축 {axis} size와 min/max 불일치")
        check(math.isclose(center, (lo + hi) / 2,
                           abs_tol=1e-5, rel_tol=1e-5),
              f"{prefix}: 축 {axis} center와 min/max 불일치")


def main():
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    rooms = data["rooms"]
    check(bool(rooms), "Room 입력이 비어 있음")

    room_map = {}
    all_ids = set()
    total_objects = 0

    for room in rooms:
        rid = room["room_id"]
        check(rid not in room_map, f"중복 Room: {rid}")
        room_map[rid] = room

        check(room.get("coordinate_frame") == "athome_z_up",
              f"{rid}: 좌표계 불일치")
        check(room.get("up_axis") == "z", f"{rid}: 높이 축 불일치")
        check(room.get("length_unit") == "meter", f"{rid}: 단위 불일치")

        objects = room["objects"]
        ids = [obj["object_id"] for obj in objects]
        id_set = set(ids)
        total_objects += len(objects)

        check(bool(ids), f"{rid}: 빈 Room")
        check(len(ids) == len(id_set), f"{rid}: 객체 ID 중복")
        check(not (all_ids & id_set), f"{rid}: 다른 Room과 객체 ID 중복")
        all_ids.update(id_set)

        for obj in objects:
            oid = obj["object_id"]
            prefix = f"{rid}/{oid}"
            validate_bbox(obj, prefix)
            check(bool(str(obj.get("semantic_tag", "")).strip()),
                  f"{prefix}: semantic_tag 없음")

            if "room_id" in obj:
                check(obj["room_id"] == rid, f"{prefix}: 소속 Room 불일치")

            neighbors = [n["object_id"] for n in obj["neighbors"]]
            children = obj["child_candidate_ids"]

            for name, refs in (("neighbors", neighbors), ("children", children)):
                check(len(refs) == len(set(refs)),
                      f"{prefix}: {name} ID 중복")
                check(oid not in refs, f"{prefix}: 자기 자신을 {name}로 참조")
                check(set(refs) <= id_set,
                      f"{prefix}: {name}에 Room 밖 객체 포함")

            check(set(children) <= set(neighbors),
                  f"{prefix}: 이웃 목록에 없는 Child 후보")

    records = [
        json.loads(line)
        for line in BATCH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_ids = {f"{SCENE}:{rid}" for rid in room_map}
    request_ids = [r["custom_id"] for r in records]

    check(len(request_ids) == len(set(request_ids)), "Batch custom_id 중복")
    check(set(request_ids) == expected_ids, "Batch 요청 누락 또는 추가 발견")

    for record in records:
        cid = record["custom_id"]
        check(record["method"] == "POST", f"{cid}: HTTP method 오류")
        check(record["url"] == "/v1/chat/completions", f"{cid}: endpoint 오류")

        body = record["body"]
        check(body["model"] == "gpt-4.1-2025-04-14", f"{cid}: 모델 불일치")

        messages = body["messages"]
        system = [m for m in messages if m["role"] == "system"]
        users = [m for m in messages if m["role"] == "user"]
        check(len(system) == 1 and bool(system[0]["content"].strip()),
              f"{cid}: system prompt 오류")
        check(len(users) == 1, f"{cid}: user message 수 오류")
        if len(users) != 1:
            continue

        payload = json.loads(users[0]["content"])
        rid = payload["room_id"]
        check(cid == f"{SCENE}:{rid}", f"{cid}: 요청 ID와 Room 불일치")
        check(payload == room_map.get(rid), f"{cid}: 원본 Room 입력과 불일치")

        fmt = body["response_format"]
        check(fmt["type"] == "json_schema", f"{cid}: 출력 형식 오류")
        check(fmt["json_schema"]["strict"] is True, f"{cid}: strict 미설정")

        props = fmt["json_schema"]["schema"]["properties"]
        check(props["room_id"]["enum"] == [rid], f"{cid}: Room enum 오류")

        allowed = props["workspace_sources"]["items"]["properties"][
            "source_object_id"
        ]["enum"]
        actual = {obj["object_id"] for obj in payload["objects"]}
        check(set(allowed) == actual, f"{cid}: Source enum과 객체 목록 불일치")

    print("=== Scene Graph 구성용 입력 검증 ===")
    print("Room 수:", len(rooms))
    print("Object 수:", total_objects)
    print("Batch 요청 수:", len(records))
    print("오류 수:", len(errors))

    for error in errors[:50]:
        print("-", error)
    if len(errors) > 50:
        print(f"... 나머지 오류 {len(errors) - 50}개")

    if errors:
        raise SystemExit(1)

    print("입력 구조·참조·Batch 일관성 검사 통과")
    print("기하 관계의 정확성과 LLM 라벨 품질은 별도 검증 대상입니다.")


if __name__ == "__main__":
    main()
