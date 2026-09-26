import hashlib
import json
from collections import defaultdict
from pathlib import Path

import habitat_sim
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/hm3d/wcojb4TFT35"
CONFIG_PATH = BASE / "stair_triangle_selection.review.json"
GRAPH_PATH = ROOT / "outputs/hm3d/wcojb4TFT35.workspace_graph.v2.json"
FLOORS_PATH = BASE / "floor_environments.json"
OUTPUT_PATH = BASE / "component_room_candidates.review.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


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


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    floor_plan = json.loads(FLOORS_PATH.read_text(encoding="utf-8"))

    upper = [
        env for env in floor_plan["environments"]
        if env["floor_id"] == "floor_2"
    ]
    if len(upper) != 1:
        raise ValueError("기존 분리안에서 floor_2를 확인할 수 없습니다.")
    room_ids = set(upper[0]["room_ids"])

    nav_path = ROOT / config["navmesh_path"]
    if digest(nav_path.read_bytes()) != config["navmesh_sha256"]:
        raise ValueError("원본 NavMesh가 변경됐습니다.")

    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(nav_path)):
        raise RuntimeError("NavMesh 로딩 실패")

    native = np.asarray(
        pf.build_navmesh_vertices(config["island_id"]), dtype=float
    )
    vertices = np.column_stack((
        native[:, 0], -native[:, 2], native[:, 1]
    ))
    faces = np.asarray(
        pf.build_navmesh_vertex_indices(config["island_id"]),
        dtype=np.int64,
    ).reshape(-1, 3)

    signature = digest(
        vertices.astype("<f8").tobytes()
        + faces.astype("<i8").tobytes()
    )
    if signature != config["geometry_sha256"]:
        raise ValueError("삼각형 순서 또는 좌표가 변경됐습니다.")

    listed = (
        config["retained_triangle_ids"]
        + config["excluded_triangle_ids"]
    )
    if (
        any(type(i) is not int for i in listed)
        or len(listed) != len(set(listed))
        or set(listed) != set(range(len(faces)))
    ):
        raise ValueError("선택 목록에 중복 또는 누락이 있습니다.")

    triangles = vertices[faces]
    areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ), axis=1,
    ) / 2

    # 앞서 위치와 높이를 확인한 계단 아래쪽 잔여 조각입니다.
    # 기존 선택 파일은 변경하지 않고 검사 중에만 제외합니다.
    extra_excluded = []
    active = set(config["retained_triangle_ids"])
    if 82 in active:
        if not np.allclose(
            triangles[82, :, 2], 0.395, atol=0.001, rtol=0
        ):
            raise ValueError("삼각형 82의 높이가 이전 검사와 다릅니다.")
        active.remove(82)
        extra_excluded.append(82)

    _, remap = np.unique(vertices, axis=0, return_inverse=True)
    owners = defaultdict(list)
    for i, face in enumerate(remap[faces]):
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

    groups.sort(key=lambda ids: (-float(areas[ids].sum()), min(ids)))

    floors = [
        obj for obj in graph["objects"]
        if obj["room_id"] in room_ids
        and obj.get("semantic_tag") == "floor"
        and obj.get("bbox")
    ]
    labels = {
        room["room_id"]: room.get("room_label", "unknown")
        for room in graph["rooms"]
    }

    records = []
    touched_rooms = set()

    print("=== 위층 이동 영역과 Room 후보 ===")
    print("검사 중 추가 제외:", extra_excluded)
    print("유지 연결 영역 수:", len(groups))
    print("기존 floor_2 Room 배정은 아직 분리안입니다.")

    for index, ids in enumerate(groups):
        name = chr(ord("A") + index)
        tris = triangles[ids]
        z_min = float(tris[:, :, 2].min())
        z_max = float(tris[:, :, 2].max())

        print(
            f"\n영역 {name}: {len(ids)}개 삼각형, "
            f"{areas[ids].sum():.3f} m², "
            f"높이 {z_min:.3f} ~ {z_max:.3f} m"
        )

        candidates = []
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

            floor_z = float(bbox["center"][2])
            rid = obj["room_id"]
            touched_rooms.add(rid)
            candidates.append({
                "room_id": rid,
                "floor_object_id": obj["object_id"],
                "bbox_xy_overlap_area_m2": overlap,
                "floor_center_z_m": floor_z,
                "navmesh_z_range_m": [z_min, z_max],
            })
            print(
                f"  {rid} ({labels.get(rid, 'unknown')}) / "
                f"{obj['object_id']}: XY 겹침 {overlap:.3f} m², "
                f"바닥 중심 높이 {floor_z:.3f} m"
            )

        records.append({
            "component": name,
            "triangle_ids": ids,
            "area_m2": float(areas[ids].sum()),
            "z_range_m": [z_min, z_max],
            "room_candidates": candidates,
        })

    missing = sorted(room_ids - touched_rooms)
    print("\nXY 겹침 후보가 없는 Room:", missing)
    print("※ 겹침 면적은 바닥 객체별 값이며 서로 합산하지 않습니다.")
    print("※ 높이가 다른 바닥도 XY에서 겹칠 수 있으므로 자동 배정하지 않습니다.")

    report = {
        "status": "candidate_review_only",
        "navmesh_sha256": config["navmesh_sha256"],
        "geometry_sha256": signature,
        "selection_sha256": digest(CONFIG_PATH.read_bytes()),
        "floor_plan_sha256": digest(FLOORS_PATH.read_bytes()),
        "graph_sha256": digest(GRAPH_PATH.read_bytes()),
        "temporary_extra_excluded_triangle_ids": extra_excluded,
        "method": "floor_bbox_xy_intersection_with_navmesh_triangles",
        "room_assignment_verified": False,
        "components": records,
        "rooms_without_xy_overlap": missing,
    }
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n보고서 저장:", OUTPUT_PATH)
    print("기존 그래프·선택 파일·층 분리안은 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
