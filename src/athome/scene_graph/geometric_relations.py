"""Compute within-room geometry for Workspace source selection."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def pair_geometry(child_bbox, source_bbox):
    lo, hi = child_bbox["min"], child_bbox["max"]
    slo, shi = source_bbox["min"], source_bbox["max"]

    gaps = [
        max(slo[k] - hi[k], lo[k] - shi[k], 0.0)
        for k in (0, 1)
    ]
    intersection = math.prod(
        max(0.0, min(hi[k], shi[k]) - max(lo[k], slo[k]))
        for k in (0, 1)
    )
    child_area = (hi[0] - lo[0]) * (hi[1] - lo[1])

    return {
        "distance_xy": math.hypot(*gaps),
        "delta_z": abs(lo[2] - shi[2]),
        "overlap_xy": (
            intersection / child_area if child_area > 0 else None
        ),
    }


def build_relations(data, distance_threshold, height_threshold,
                    overlap_threshold):
    if (
        data.get("coordinate_frame") != "athome_z_up"
        or data.get("up_axis") != "z"
        or data.get("length_unit") != "meter"
    ):
        raise ValueError("AtHOME Z-up, meter 단위 데이터가 필요합니다.")

    values = [distance_threshold, height_threshold, overlap_threshold]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("임계값은 유한한 값이어야 합니다.")
    if distance_threshold < 0 or height_threshold <= 0:
        raise ValueError("거리 기준은 0 이상, 높이 기준은 양수여야 합니다.")
    if not 0 <= overlap_threshold < 1:
        raise ValueError("겹침 비율 기준은 0 이상 1 미만이어야 합니다.")

    groups = defaultdict(list)
    seen = set()
    valid_rooms = {room["room_id"] for room in data["rooms"]}

    for obj in data["objects"]:
        if obj["object_id"] in seen:
            raise ValueError(f"중복 객체: {obj['object_id']}")
        seen.add(obj["object_id"])
        if obj["room_id"] not in valid_rooms:
            raise ValueError(f"소속 Room 없음: {obj['object_id']}")

        bbox = obj["bbox"]
        if bbox is None:
            raise ValueError(f"Bounding Box 없음: {obj['object_id']}")
        lo, hi = bbox["min"], bbox["max"]
        if (
            len(lo) != 3 or len(hi) != 3
            or not all(math.isfinite(v) for v in lo + hi)
            or any(a > b for a, b in zip(lo, hi))
        ):
            raise ValueError(f"잘못된 Bounding Box: {obj['object_id']}")
        groups[obj["room_id"]].append(obj)

    records = []
    for room_id in sorted(groups):
        objects = sorted(groups[room_id], key=lambda obj: obj["object_id"])

        for source in objects:
            bbox = source["bbox"]
            lo, hi = bbox["min"], bbox["max"]
            neighbors = []
            children = []

            for child in objects:
                if child["object_id"] == source["object_id"]:
                    continue

                geometry = pair_geometry(child["bbox"], bbox)
                if geometry["distance_xy"] > distance_threshold:
                    continue

                neighbors.append({
                    "object_id": child["object_id"],
                    **geometry,
                })
                overlap = geometry["overlap_xy"]
                if (
                    overlap is not None
                    and geometry["delta_z"] < height_threshold
                    and overlap > overlap_threshold
                ):
                    children.append(child["object_id"])

            records.append({
                "room_id": room_id,
                "object_id": source["object_id"],
                "top_surface_center": [
                    (lo[0] + hi[0]) / 2,
                    (lo[1] + hi[1]) / 2,
                    hi[2],
                ],
                "neighbors": neighbors,
                "child_candidate_ids": children,
            })

    return {
        "schema_version": "0.1",
        "stage": "geometric_relations",
        "coordinate_frame": "athome_z_up",
        "length_unit": "meter",
        "thresholds": {
            "distance_xy_m": distance_threshold,
            "delta_z_m": height_threshold,
            "overlap_xy": overlap_threshold,
        },
        "objects": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--overlap", type=float, required=True)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("입력과 출력 경로는 달라야 합니다.")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    result = build_relations(data, args.distance, args.height, args.overlap)
    result["source_map"] = str(args.input.resolve())

    payload = json.dumps(
        result, ensure_ascii=False, indent=2, allow_nan=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(args.output)

    records = result["objects"]
    nonempty = [r for r in records if r["child_candidate_ids"]]

    print("=== 기하 관계 계산 결과 ===")
    print("처리한 Object 수:", len(records))
    print("Child 후보가 있는 Object 수:", len(nonempty))
    print("Child 후보 관계 수:",
          sum(len(r["child_candidate_ids"]) for r in records))
    print("저장 위치:", args.output.resolve())
    print("\n=== Child 후보 예시 ===")
    for record in nonempty[:10]:
        print(record["object_id"], "->", record["child_candidate_ids"][:10])


if __name__ == "__main__":
    main()
