"""Navigable components and their room candidates (all floors, all islands).

Each connected group of retained NavMesh triangles (stair selection) is
assigned to the floor environment whose floor-centre height range is nearest
to the group's area-weighted surface height, if within half the floor gap of
prepare_floor_environments.py (floors are separated by more than the gap).
Its room candidates are the floors of that environment which the group
covers by at least one robot footprint. Groups without a room candidate
(furniture tops) are discarded. Each room's primary component (per level,
for rooms split across levels) is the one covering most of its floor; a component that is primary for no room is a
pocket of a room whose main navigable area is another component (the
largest-island convention of Habitat rearrangement, applied per room) and is
discarded too. A room may still belong to several components (e.g. a
corridor split by a staircase). Kept groups are named A, B, ... by
decreasing area; discarded ones are recorded with the reason.

A single-island selection (schema 0.1) gives the former report format.
Layout: ATHOME_HM3D_LAYOUT.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import habitat_sim
import numpy as np
from athome.data.hm3d.floors import on_floor
from athome.data.hm3d.layout import current_layout
from athome.data.hm3d.navmesh_levels import island_geometry, selection_islands


ROOT = Path(__file__).resolve().parents[1]
LAYOUT = current_layout()
BASE = LAYOUT.scene_dir
CONFIG_PATH = LAYOUT.stair_selection
GRAPH_PATH = LAYOUT.graph
FLOORS_PATH = LAYOUT.floors
OUTPUT_PATH = LAYOUT.room_report
# A floor counts for a component only if the component's NavMesh covers at
# least one robot footprint of it (Tidybot++ 0.50 x 0.54 m, proposal 2.(1)).
# Smaller touches (e.g. a stair landing grazed by 0.01 m^2) are recorded
# separately and are not room candidates.
MIN_FLOOR_OVERLAP_M2 = 0.50 * 0.54


def digest(data):
    return hashlib.sha256(data).hexdigest()


def component_name(index):
    """A..Z, AA, AB, ... (spreadsheet column order)."""
    name = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        name = chr(ord("A") + rest) + name
    return name


def intersection_area(triangle_xy, lower, upper):
    """삼각형과 XY 직사각형의 교집합 면적을 계산합니다."""
    polygon = [p.copy() for p in triangle_xy]

    for axis, boundary, greater in (
        (0, lower[0], True),
        (0, upper[0], False),
        (1, lower[1], True),
        (1, upper[1], False),
    ):
        if not polygon:
            return 0.0
        clipped = []
        previous = polygon[-1]
        previous_in = (
            previous[axis] >= boundary if greater
            else previous[axis] <= boundary
        )

        for current in polygon:
            current_in = (
                current[axis] >= boundary if greater
                else current[axis] <= boundary
            )
            if current_in != previous_in:
                t = (
                    (boundary - previous[axis])
                    / (current[axis] - previous[axis])
                )
                clipped.append(previous + t * (current - previous))
            if current_in:
                clipped.append(current.copy())
            previous, previous_in = current, current_in

        polygon = clipped

    if len(polygon) < 3:
        return 0.0
    p = np.asarray(polygon)
    return float(abs(
        np.dot(p[:, 0], np.roll(p[:, 1], -1))
        - np.dot(p[:, 1], np.roll(p[:, 0], -1))
    ) / 2)


def island_groups(pf, record, manual):
    """Validated geometry and edge-connected retained groups of one island."""
    island = record["island_id"]
    vertices, faces, signature = island_geometry(pf, island)
    if signature != record["geometry_sha256"]:
        raise ValueError(f"섬 {island}: 삼각형 순서 또는 좌표가 변경됐습니다.")

    listed = record["retained_triangle_ids"] + record["excluded_triangle_ids"]
    if (
        any(type(i) is not int for i in listed)
        or len(listed) != len(set(listed))
        or set(listed) != set(range(len(faces)))
    ):
        raise ValueError(f"섬 {island}: 선택 목록에 중복 또는 누락이 있습니다.")

    triangles = vertices[faces]
    areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ), axis=1,
    ) / 2

    # 앞서 위치와 높이를 확인한 계단 아래쪽 잔여 조각입니다(수동 선택 파일 전용).
    # 기존 선택 파일은 변경하지 않고 검사 중에만 제외합니다.
    extra_excluded = []
    active = set(record["retained_triangle_ids"])
    if manual and 82 in active:
        if not np.allclose(
            triangles[82, :, 2], 0.395, atol=0.001, rtol=0
        ):
            raise ValueError("삼각형 82의 높이가 이전 검사와 다릅니다.")
        active.remove(82)
        extra_excluded.append(82)

    _, remap = np.unique(vertices, axis=0, return_inverse=True)
    owners = defaultdict(list)
    for i, face in enumerate(remap.reshape(-1)[faces]):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            owners[tuple(sorted((int(face[a]), int(face[b]))))].append(i)

    adjacency = [set() for _ in faces]
    for ids in owners.values():
        if len(ids) > 2:
            raise ValueError("세 면 이상이 공유하는 변이 있습니다.")
        if len(ids) == 2:
            a, b = ids
            adjacency[a].add(b)
            adjacency[b].add(a)

    groups = []
    remaining = set(active)
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack, group = [seed], []
        while stack:
            node = stack.pop()
            group.append(node)
            neighbors = adjacency[node] & remaining
            remaining.difference_update(neighbors)
            stack.extend(sorted(neighbors))
        groups.append(sorted(group))
    return signature, triangles, areas, groups, extra_excluded


def nearest_floor(height, environments, max_distance):
    """Floor environment whose floor-centre height range is nearest."""
    best, best_distance = None, None
    for env in environments:
        lo, hi = env["floor_center_height_range_m"]
        distance = max(lo - height, 0.0, height - hi)
        if best_distance is None or distance < best_distance:
            best, best_distance = env, distance
    return (best if best_distance is not None and best_distance <= max_distance else None,
            best_distance)


def room_candidates(tris, floors, labels, z_range):
    candidates, grazed = [], []
    for obj in sorted(floors, key=lambda o: o["object_id"]):
        bbox = obj["bbox"]
        lower = np.asarray(bbox["min"], dtype=float)
        upper_bound = np.asarray(bbox["max"], dtype=float)

        overlap = sum(
            intersection_area(tri[:, :2], lower, upper_bound)
            for tri in tris
        )
        # 부동소수점 계산의 극소 면적만 제외합니다.
        if overlap <= 1e-8:
            continue
        if overlap < MIN_FLOOR_OVERLAP_M2:
            grazed.append({"floor_object_id": obj["object_id"], "room_id": obj["room_id"],
                           "bbox_xy_overlap_area_m2": overlap})
            print(f"  (무시) {obj['object_id']}: 겹침 {overlap:.3f} m² < 로봇 바닥 면적")
            continue

        floor_z = float(bbox["center"][2])
        rid = obj["room_id"]
        candidates.append({
            "room_id": rid,
            "floor_object_id": obj["object_id"],
            "bbox_xy_overlap_area_m2": overlap,
            "floor_center_z_m": floor_z,
            "navmesh_z_range_m": z_range,
        })
        print(
            f"  {rid} ({labels.get(rid, 'unknown')}) / "
            f"{obj['object_id']}: XY 겹침 {overlap:.3f} m², "
            f"바닥 중심 높이 {floor_z:.3f} m"
        )
    return candidates, grazed


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    floor_plan = json.loads(FLOORS_PATH.read_text(encoding="utf-8"))
    multi = "islands" in config
    environments = floor_plan["environments"]
    max_distance = floor_plan["parameters"]["floor_gap_m"] / 2

    nav_path = ROOT / config["navmesh_path"]
    if digest(nav_path.read_bytes()) != config["navmesh_sha256"]:
        raise ValueError("원본 NavMesh가 변경됐습니다.")

    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(nav_path)):
        raise RuntimeError("NavMesh 로딩 실패")

    labels = {
        room["room_id"]: room.get("room_label", "unknown")
        for room in graph["rooms"]
    }
    floors_by_env = {
        env["floor_id"]: [
            obj for obj in graph["objects"]
            if on_floor(env, obj)
            and obj.get("semantic_tag") == "floor"
            and obj.get("bbox")
        ]
        for env in environments
    }

    islands, kept, discarded, extra_excluded = [], [], [], []
    for record in selection_islands(config):
        island = record["island_id"]
        signature, triangles, areas, groups, extra = island_groups(pf, record, not multi)
        islands.append({"island_id": island, "geometry_sha256": signature})
        extra_excluded += extra
        for ids in groups:
            tris = triangles[ids]
            area = float(areas[ids].sum())
            z_range = [float(tris[:, :, 2].min()), float(tris[:, :, 2].max())]
            height = float((areas[ids] * tris[:, :, 2].mean(axis=1)).sum() / area)
            env, distance = nearest_floor(height, environments, max_distance)
            print(
                f"\n섬 {island} / 영역 후보: {len(ids)}개 삼각형, {area:.3f} m², "
                f"높이 {z_range[0]:.3f} ~ {z_range[1]:.3f} m, "
                f"층 {env['floor_id'] if env else '없음'} (거리 {distance:.3f} m)"
            )
            base = {"island_id": island, "triangle_ids": ids, "area_m2": area, "z_range_m": z_range}
            if env is None:
                discarded.append({**base, "reason": "no_floor_environment_within_half_floor_gap"})
                continue
            candidates, grazed = room_candidates(tris, floors_by_env[env["floor_id"]], labels, z_range)
            if not candidates:
                discarded.append({**base, "floor_id": env["floor_id"],
                                  "reason": "no_room_floor_overlap_of_robot_footprint"})
                continue
            kept.append((base, env["floor_id"], candidates, grazed))

    kept.sort(key=lambda item: (-item[0]["area_m2"], item[0]["island_id"], min(item[0]["triangle_ids"])))
    primary = {}  # (room, level) -> (floor overlap, index of kept component)
    for k, (_, floor_id, candidates, _) in enumerate(kept):
        cover = defaultdict(float)
        for c in candidates:
            cover[(c["room_id"], floor_id)] += c["bbox_xy_overlap_area_m2"]
        for key, overlap in cover.items():
            if key not in primary or overlap > primary[key][0]:
                primary[key] = (overlap, k)
    owners = {k for _, k in primary.values()}
    for k, (base, floor_id, candidates, _) in enumerate(kept):
        if k not in owners:
            discarded.append({**base, "floor_id": floor_id,
                              "room_ids": sorted({c["room_id"] for c in candidates}),
                              "reason": "pocket_primary_for_no_room"})
            print(f"(제외) 섬 {base['island_id']} {base['area_m2']:.3f} m²: 주 영역인 Room 없음")
    kept = [item for k, item in enumerate(kept) if k in owners]
    records = []
    touched_rooms = set()
    for index, (base, floor_id, candidates, grazed) in enumerate(kept):
        name = component_name(index)
        touched_rooms |= {c["room_id"] for c in candidates}
        record = {"component": name}
        if multi:  # the single-island report keeps its former fields
            record.update(island_id=base["island_id"], floor_id=floor_id)
        record.update(triangle_ids=base["triangle_ids"], area_m2=base["area_m2"],
                      z_range_m=base["z_range_m"], room_candidates=candidates)
        if grazed:  # only when present, so earlier reports stay byte-identical
            record["grazed_floors_below_robot_footprint"] = grazed
        records.append(record)
        print(f"영역 {name}: 섬 {base['island_id']}, {floor_id}, {base['area_m2']:.3f} m², "
              f"Room {sorted({c['room_id'] for c in candidates})}")

    used_floors = {floor_id for _, floor_id, _, _ in kept}
    room_ids = {rid for env in environments if env["floor_id"] in used_floors for rid in env["room_ids"]}
    missing = sorted(room_ids - touched_rooms)
    print("\n검사 중 추가 제외:", extra_excluded)
    print("유지 영역:", len(records), "제외 영역:", len(discarded))
    print("영역이 있는 층에서 XY 겹침 후보가 없는 Room:", missing)
    print("※ 겹침 면적은 바닥 객체별 값이며 서로 합산하지 않습니다.")

    report = {
        "status": "candidate_review_only",
        "navmesh_sha256": config["navmesh_sha256"],
    }
    if multi:
        report["islands"] = islands
    else:
        report["geometry_sha256"] = islands[0]["geometry_sha256"]
    report.update({
        "selection_sha256": digest(CONFIG_PATH.read_bytes()),
        "floor_plan_sha256": digest(FLOORS_PATH.read_bytes()),
        "graph_sha256": digest(GRAPH_PATH.read_bytes()),
        "temporary_extra_excluded_triangle_ids": extra_excluded,
        "method": "floor_bbox_xy_intersection_with_navmesh_triangles",
        "room_assignment_verified": False,
        "components": records,
        "rooms_without_xy_overlap": missing,
    })
    if multi:
        report["floor_assignment"] = {
            "method": "nearest_floor_center_height_range_to_area_weighted_navmesh_height",
            "max_distance_m": max_distance,
        }
        report["floors_without_component"] = sorted(
            env["floor_id"] for env in environments if env["floor_id"] not in used_floors)
        report["discarded_components"] = discarded
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n보고서 저장:", OUTPUT_PATH)
    print("기존 그래프·선택 파일·층 분리안은 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
