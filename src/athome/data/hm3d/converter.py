"""Convert exported HM3DSem annotations into a Room–Object map."""

import argparse
import json
from collections import Counter
from pathlib import Path


def build_room_object_map(source):
    if source.get("coordinate_frame") != "habitat_native":
        raise ValueError("예상한 좌표계가 아닙니다.")
    if source.get("up_axis") != "y":
        raise ValueError("높이 축 정보가 없거나 다릅니다.")

    # Keep source Region IDs; the synthetic unknown region is not a Room.
    rooms = {}
    excluded_regions = []
    for region in source["regions"]:
        region_id = region["region_id"]
        if region_id is None or region_id == "_-1":
            excluded_regions.append(region_id)
            continue
        if region_id in rooms:
            raise ValueError(f"중복 Region ID: {region_id}")
        rooms[region_id] = {
            "room_id": region_id,
            "source_region_id": region_id,
            "room_label": None,
            "object_ids": [],
        }

    objects = []
    excluded_objects = []
    seen_ids = set()

    for obj in source["objects"]:
        object_id = obj["object_id"]
        if object_id in seen_ids:
            raise ValueError(f"중복 Object ID: {object_id}")
        seen_ids.add(object_id)

        reasons = []
        category = obj.get("category")
        room_id = obj.get("region_id")

        if category is None or str(category).strip().casefold() in {
            "", "unknown"
        }:
            reasons.append("unknown_category")
        if obj.get("bbox") is None:
            reasons.append("invalid_bbox")
        if room_id not in rooms:
            reasons.append("invalid_region")

        if reasons:
            excluded_objects.append({
                "object_id": object_id,
                "category": category,
                "source_region_id": room_id,
                "reasons": reasons,
            })
            continue

        objects.append({
            "object_id": object_id,
            "semantic_tag": category,
            "room_id": room_id,
            "bbox": obj["bbox"],
        })
        rooms[room_id]["object_ids"].append(object_id)

    if len(objects) + len(excluded_objects) != len(source["objects"]):
        raise RuntimeError("변환 전후 객체 수가 일치하지 않습니다.")

    return {
        "schema_version": "0.1",
        "stage": "room_object_map",
        "coordinate_frame": source["coordinate_frame"],
        "up_axis": source["up_axis"],
        "length_unit": source["length_unit"],
        "bbox_method": source.get("bbox_method"),
        "rooms": list(rooms.values()),
        "objects": objects,
        "excluded_regions": excluded_regions,
        "excluded_objects": excluded_objects,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("입력과 출력 파일은 달라야 합니다.")

    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = build_room_object_map(source)
    result["source_annotations"] = str(args.input.resolve())

    payload = json.dumps(
        result, ensure_ascii=False, indent=2, allow_nan=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(args.output)

    counts = Counter(
        reason
        for obj in result["excluded_objects"]
        for reason in obj["reasons"]
    )
    print("=== Room–Object 변환 결과 ===")
    print("Room 수:", len(result["rooms"]))
    print("유지한 Object 수:", len(result["objects"]))
    print("제외한 Object 수:", len(result["excluded_objects"]))
    print("제외 사유별 수:", dict(counts))
    print("※ 객체 하나에 제외 사유가 여러 개일 수 있습니다.")
    print("저장 위치:", args.output.resolve())


if __name__ == "__main__":
    main()
