"""Convert Habitat Y-up coordinates to AtHOME Z-up coordinates."""

import argparse
import copy
import json
import math
from pathlib import Path


def habitat_to_athome(point):
    x, y, z = point
    return [x, -z, y]


def athome_to_habitat(point):
    x, y, z = point
    return [x, z, -y]


def convert_bbox(bbox):
    lo = bbox["min"]
    hi = bbox["max"]

    if len(lo) != 3 or len(hi) != 3:
        raise ValueError("Bounding Box 좌표는 3차원이어야 합니다.")
    if not all(math.isfinite(v) for v in lo + hi):
        raise ValueError("Bounding Box에 비유한 값이 있습니다.")
    if any(a > b for a, b in zip(lo, hi)):
        raise ValueError("Bounding Box의 min/max가 잘못되었습니다.")

    # The sign reversal swaps the lower and upper Z bounds.
    new_lo = [lo[0], -hi[2], lo[1]]
    new_hi = [hi[0], -lo[2], hi[1]]

    return {
        "center": [(a + b) / 2 for a, b in zip(new_lo, new_hi)],
        "size": [b - a for a, b in zip(new_lo, new_hi)],
        "min": new_lo,
        "max": new_hi,
    }


def convert_map(source):
    if (
        source.get("coordinate_frame") != "habitat_native"
        or source.get("up_axis") != "y"
    ):
        raise ValueError("Habitat Y-up 데이터만 변환할 수 있습니다.")
    if source.get("length_unit") != "meter":
        raise ValueError("거리 단위가 meter가 아닙니다.")

    result = copy.deepcopy(source)
    for obj in result["objects"]:
        if obj["bbox"] is not None:
            obj["bbox"] = convert_bbox(obj["bbox"])

    result["schema_version"] = "0.2"
    result["coordinate_frame"] = "athome_z_up"
    result["up_axis"] = "z"
    result["coordinate_transform"] = {
        "source_frame": "habitat_native",
        "target_frame": "athome_z_up",
        "rotation_matrix": [
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ],
        "translation": [0, 0, 0],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("원본 보존을 위해 출력 경로를 다르게 지정하세요.")

    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = convert_map(source)
    result["source_room_object_map"] = str(args.input.resolve())

    payload = json.dumps(
        result, ensure_ascii=False, indent=2, allow_nan=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(args.output)

    print("=== 좌표 변환 결과 ===")
    print("좌표계:", result["coordinate_frame"])
    print("높이 축:", result["up_axis"])
    print("Room 수:", len(result["rooms"]))
    print("Object 수:", len(result["objects"]))
    print("저장 위치:", args.output.resolve())


if __name__ == "__main__":
    main()
