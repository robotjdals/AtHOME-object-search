import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from athome.data.hm3d.floors import BELOW_FLOOR_TOLERANCE_M, split_room


def main():
    parser = argparse.ArgumentParser(
        description="바닥 객체 높이로 단층 환경 분리안 생성"
    )
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--building-id", required=True)
    parser.add_argument("--floor-gap-m", type=float, default=1.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.floor_gap_m <= 0:
        raise ValueError("층 후보 분리 간격은 양수여야 합니다.")

    raw = args.graph.read_bytes()
    graph = json.loads(raw)

    if graph["coordinate_frame"] != "athome_z_up":
        raise ValueError("athome_z_up 좌표계가 필요합니다.")
    if graph["length_unit"] != "meter":
        raise ValueError("미터 단위가 필요합니다.")

    room_ids = {r["room_id"] for r in graph["rooms"]}
    floor_objects = sorted(
        (
            obj for obj in graph["objects"]
            if obj["semantic_tag"].strip().casefold() == "floor"
        ),
        key=lambda obj: obj["bbox"]["center"][2],
    )
    if not floor_objects:
        raise ValueError("층 분리에 사용할 바닥 객체가 없습니다.")

    groups = []
    for obj in floor_objects:
        height = obj["bbox"]["center"][2]
        if (
            not groups
            or height - groups[-1][-1]["bbox"]["center"][2]
            > args.floor_gap_m
        ):
            groups.append([])
        groups[-1].append(obj)

    room_groups = defaultdict(set)
    floor_level = {}
    for index, group in enumerate(groups):
        for obj in group:
            if obj["room_id"] not in room_ids:
                raise ValueError(f"그래프에 없는 Room: {obj['room_id']}")
            room_groups[obj["room_id"]].add(f"floor_{index}")
            floor_level[obj["object_id"]] = f"floor_{index}"

    # Rooms on several levels (or without a floor object) are split per level
    # (src/athome/data/hm3d/floors.py); single-level rooms are unchanged.
    all_levels = [(f"floor_{i}", min(o["bbox"]["center"][2] for o in g))
                  for i, g in enumerate(groups)]
    objects_by_room = defaultdict(list)
    for obj in graph["objects"]:
        objects_by_room[obj["room_id"]].append(obj)
    partial = {}
    for rid in sorted(room_ids):
        levels = [
            (fid, min(o["bbox"]["center"][2] for o in floor_objects
                      if o["room_id"] == rid and floor_level[o["object_id"]] == fid))
            for fid in sorted(room_groups[rid])
        ] if room_groups.get(rid) else all_levels
        if len(room_groups.get(rid, ())) == 1 or not objects_by_room[rid]:
            continue
        split = split_room(objects_by_room[rid], levels, floor_level)
        if len(split) == 1:
            room_groups[rid] = set(split)
            continue
        partial[rid] = split
        room_groups[rid] = set(split)

    environments = []
    for index, group in enumerate(groups):
        floor_id = f"floor_{index}"
        heights = [o["bbox"]["center"][2] for o in group]
        rooms = sorted(rid for rid, fids in room_groups.items() if floor_id in fids)
        env = {
            "environment_id": f"{args.building_id}__{floor_id}",
            "building_id": args.building_id,
            "floor_id": floor_id,
            "room_ids": rooms,
            "floor_object_ids": [o["object_id"] for o in group],
            "floor_center_height_range_m": [min(heights), max(heights)],
            "navigation_map_status": "not_generated",
        }
        here = {rid: split[floor_id] for rid, split in partial.items() if floor_id in split}
        if here:
            env["partial_room_object_ids"] = here
        environments.append(env)

    missing = sorted(rid for rid in room_ids - set(room_groups) if objects_by_room[rid])
    empty = sorted(rid for rid in room_ids - set(room_groups) if not objects_by_room[rid])
    ambiguous = {rid: sorted(split) for rid, split in partial.items()}

    result = {
        "schema_version": "0.1",
        "status": "pending_spatial_validation",
        "building_id": args.building_id,
        "split_group_id": args.building_id,
        "coordinate_frame": "athome_z_up",
        "source_graph": str(args.graph.resolve()),
        "source_graph_sha256": hashlib.sha256(raw).hexdigest(),
        "floor_assignment_is_ground_truth": False,
        "method": "floor_object_center_height_gap",
        "parameters": {"floor_gap_m": args.floor_gap_m},
        "environments": environments,
        "unassigned_room_ids": missing,
        "ambiguous_rooms": ambiguous,
    }
    if partial or empty:  # absent for single-level buildings (earlier files unchanged)
        result["multi_level_room_rule"] = {
            "method": "object_on_highest_floor_at_or_below_bottom",
            "below_floor_tolerance_m": BELOW_FLOOR_TOLERANCE_M,
        }
        result["empty_room_ids"] = empty

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== 단층 환경 분리안 ===")
    for env in environments:
        lo, hi = env["floor_center_height_range_m"]
        print(
            f"{env['floor_id']}: "
            f"Room {len(env['room_ids'])}개, "
            f"바닥 중심 높이 {lo:.3f} ~ {hi:.3f}m"
        )
        print("  Room:", ", ".join(env["room_ids"]))

    print("배정되지 않은 Room:", missing)
    print("여러 층에 걸친 Room(층별로 분할):", ambiguous)
    print("객체가 없는 Room:", empty)
    print("저장 위치:", args.output.resolve())

    if missing:
        raise SystemExit("배정이 불명확한 Room이 있어 후속 분리를 중단합니다.")

    print("Room 배정 검사 통과")
    print("지도 확인 전 분리안이며, 아직 이동 지도는 생성하지 않았습니다.")


if __name__ == "__main__":
    main()
