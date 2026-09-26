"""Prepare room-level inputs for semantic labeling and source selection."""

import argparse
import json
from pathlib import Path


def prepare_inputs(scene, geometry):
    if scene["coordinate_frame"] != geometry["coordinate_frame"]:
        raise ValueError("Scene과 기하 정보의 좌표계가 다릅니다.")
    if scene["length_unit"] != geometry["length_unit"]:
        raise ValueError("거리 단위가 다릅니다.")

    objects = {obj["object_id"]: obj for obj in scene["objects"]}
    relations = {obj["object_id"]: obj for obj in geometry["objects"]}

    if len(objects) != len(scene["objects"]):
        raise ValueError("중복 Object ID가 있습니다.")
    if len(relations) != len(geometry["objects"]):
        raise ValueError("중복 기하 정보가 있습니다.")
    if set(objects) != set(relations):
        raise ValueError("Scene과 기하 정보의 Object ID가 일치하지 않습니다.")

    requests = []
    assigned = set()
    room_ids = set()

    for room in scene["rooms"]:
        room_id = room["room_id"]
        if room_id in room_ids:
            raise ValueError(f"중복 Room ID: {room_id}")
        room_ids.add(room_id)

        ids = room["object_ids"]
        room_set = set(ids)
        if len(ids) != len(room_set):
            raise ValueError(f"Room 내 중복 Object ID: {room_id}")

        entries = []
        for object_id in sorted(ids):
            if object_id not in objects or object_id in assigned:
                raise ValueError(f"잘못된 객체 연결: {object_id}")
            assigned.add(object_id)

            obj = objects[object_id]
            rel = relations[object_id]
            if obj["room_id"] != room_id or rel["room_id"] != room_id:
                raise ValueError(f"Room 소속 불일치: {object_id}")

            neighbor_ids = {n["object_id"] for n in rel["neighbors"]}
            child_ids = set(rel["child_candidate_ids"])

            if (
                not neighbor_ids <= room_set
                or not child_ids <= neighbor_ids
                or object_id in neighbor_ids
            ):
                raise ValueError(f"잘못된 기하 관계: {object_id}")

            entries.append({
                "object_id": object_id,
                "semantic_tag": obj["semantic_tag"],
                "bbox": obj["bbox"],
                "top_surface_center": rel["top_surface_center"],
                "neighbors": rel["neighbors"],
                "child_candidate_ids": rel["child_candidate_ids"],
            })

        requests.append({
            "room_id": room_id,
            "coordinate_frame": scene["coordinate_frame"],
            "up_axis": scene["up_axis"],
            "length_unit": scene["length_unit"],
            "geometry_thresholds": geometry["thresholds"],
            "objects": entries,
        })

    if assigned != set(objects):
        raise ValueError("Room에 연결되지 않은 객체가 있습니다.")

    return {
        "schema_version": "0.1",
        "purpose": "unmasked_scene_construction_check",
        "rooms": requests,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.resolve() in {
        args.scene.resolve(), args.geometry.resolve()
    }:
        raise ValueError("출력 경로가 입력 파일과 같습니다.")

    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))

    if Path(geometry["source_map"]).resolve() != args.scene.resolve():
        raise ValueError("기하 계산에 사용한 Scene 파일과 다릅니다.")

    result = prepare_inputs(scene, geometry)
    result["source_scene"] = str(args.scene.resolve())
    result["source_geometry"] = str(args.geometry.resolve())

    payload = json.dumps(
        result, ensure_ascii=False, indent=2, allow_nan=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(args.output)

    print("=== Room별 LLM 입력 생성 ===")
    print("Room 수:", len(result["rooms"]))
    print("전체 Object 수:",
          sum(len(room["objects"]) for room in result["rooms"]))
    for room in result["rooms"]:
        count = sum(
            bool(obj["child_candidate_ids"]) for obj in room["objects"]
        )
        print(
            f'{room["room_id"]}: 객체 {len(room["objects"])}개, '
            f'Child 후보가 있는 객체 {count}개'
        )
    print("저장 위치:", args.output.resolve())


if __name__ == "__main__":
    main()
