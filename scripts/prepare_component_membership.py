import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/hm3d/wcojb4TFT35"
GRAPH_PATH = ROOT / "outputs/hm3d/wcojb4TFT35.workspace_graph.v2.json"
REPORT_PATH = BASE / "component_room_candidates.review.json"
SELECTION_PATH = BASE / "stair_triangle_selection.review.json"
FLOORS_PATH = BASE / "floor_environments.json"
OUTPUT_PATH = BASE / "component_membership.review.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    for field, path in (
        ("graph_sha256", GRAPH_PATH),
        ("selection_sha256", SELECTION_PATH),
        ("floor_plan_sha256", FLOORS_PATH),
    ):
        if report[field] != digest(path):
            raise RuntimeError(
                f"이전 검사 이후 파일이 변경됐습니다: {path.name}\n"
                "inspect_component_rooms.py를 다시 실행해 주세요."
            )

    objects = {obj["object_id"]: obj for obj in graph["objects"]}
    if len(objects) != len(graph["objects"]):
        raise ValueError("중복된 객체 ID가 있습니다.")

    room_components = defaultdict(set)
    component_floors = defaultdict(lambda: defaultdict(set))

    for component in report["components"]:
        name = component["component"]
        for candidate in component["room_candidates"]:
            rid = candidate["room_id"]
            floor_id = candidate["floor_object_id"]
            if floor_id not in objects:
                raise ValueError(f"바닥 객체 참조 오류: {floor_id}")
            room_components[rid].add(name)
            component_floors[name][rid].add(floor_id)

    provisional = {
        component["component"]: []
        for component in report["components"]
    }
    assigned_component = {}
    unresolved = []

    for obj in graph["objects"]:
        rid = obj["room_id"]
        names = sorted(room_components.get(rid, set()))
        if not names:
            continue

        oid = obj["object_id"]
        if len(names) == 1:
            name = names[0]
            provisional[name].append(oid)
            assigned_component[oid] = name
            continue

        # 여러 이동 영역에 걸친 Room은 자동 배정하지 않습니다.
        bbox = obj.get("bbox")
        evidence = []

        if bbox:
            lo = bbox["min"]
            hi = bbox["max"]

            for name in names:
                for floor_id in sorted(component_floors[name][rid]):
                    floor_box = objects[floor_id]["bbox"]
                    flo = floor_box["min"]
                    fhi = floor_box["max"]

                    width = max(0.0, min(hi[0], fhi[0]) - max(lo[0], flo[0]))
                    depth = max(0.0, min(hi[1], fhi[1]) - max(lo[1], flo[1]))
                    overlap = width * depth

                    evidence.append({
                        "component": name,
                        "floor_object_id": floor_id,
                        "bbox_xy_overlap_m2": overlap,
                        "object_bottom_minus_floor_center_m":
                            lo[2] - floor_box["center"][2],
                    })

        unresolved.append({
            "object_id": oid,
            "room_id": rid,
            "semantic_tag": obj.get("semantic_tag"),
            "role": obj.get("role"),
            "bbox": bbox,
            "room_component_candidates": names,
            "floor_bbox_evidence": evidence,
            "assignment": None,
        })

    # WorkspaceはSourceとChildが同じ候補領域に揃うか確認します。
    workspace_candidates = {name: [] for name in provisional}
    workspace_review = []

    for workspace in graph["workspaces"]:
        if workspace["room_id"] not in room_components:
            continue

        linked_ids = [
            workspace["source_object_id"],
            *workspace["child_object_ids"],
        ]
        if any(oid not in objects for oid in linked_ids):
            raise ValueError(
                f"Workspaceの参照先がありません: {workspace['workspace_id']}"
            )

        linked_components = [
            assigned_component.get(oid) for oid in linked_ids
        ]
        if (
            all(name is not None for name in linked_components)
            and len(set(linked_components)) == 1
        ):
            name = linked_components[0]
            workspace_candidates[name].append(workspace["workspace_id"])
        else:
            workspace_review.append({
                "workspace_id": workspace["workspace_id"],
                "linked_object_ids": linked_ids,
                "component_candidates": linked_components,
            })

    relevant_ids = {
        obj["object_id"] for obj in graph["objects"]
        if obj["room_id"] in room_components
    }
    provisional_ids = [
        oid for ids in provisional.values() for oid in ids
    ]
    unresolved_ids = [item["object_id"] for item in unresolved]

    if (
        len(provisional_ids) != len(set(provisional_ids))
        or set(provisional_ids) & set(unresolved_ids)
        or set(provisional_ids) | set(unresolved_ids) != relevant_ids
    ):
        raise ValueError("候補 목록의 객체 중복 또는 누락")
    
    result = {
        "status": "provisional_membership_pending_review",
        "source_graph_sha256": digest(GRAPH_PATH),
        "source_component_report_sha256": digest(REPORT_PATH),
        "room_assignment_verified": False,
        "navigation_accessibility_verified": False,
        "temporary_extra_excluded_triangle_ids":
            report["temporary_extra_excluded_triangle_ids"],
        "components": [
            {
                "component": name,
                "provisional_object_ids": sorted(provisional[name]),
                "provisional_workspace_ids":
                    sorted(workspace_candidates[name]),
            }
            for name in sorted(provisional)
        ],
        "unresolved_objects": unresolved,
        "workspace_review": workspace_review,
    }

    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== 영역별 임시 소속 목록 ===")
    for name in sorted(provisional):
        print(
            f"영역 {name}: 객체 {len(provisional[name])}개, "
            f"Workspace {len(workspace_candidates[name])}개"
        )

    print("\n=== 여러 영역에 걸친 Room의 객체 ===")
    for item in unresolved:
        bbox = item["bbox"]
        height = (
            f"{bbox['min'][2]:.3f} ~ {bbox['max'][2]:.3f}m"
            if bbox else "없음"
        )
        print(
            f"\n{item['object_id']} / {item['semantic_tag']} / "
            f"{item['role']} / 높이 {height}"
        )
        for evidence in item["floor_bbox_evidence"]:
            print(
                f"  {evidence['component']} / "
                f"{evidence['floor_object_id']}: "
                f"XY 겹침 {evidence['bbox_xy_overlap_m2']:.3f}m², "
                f"객체 하단−바닥 중심 "
                f"{evidence['object_bottom_minus_floor_center_m']:.3f}m"
            )

    print("\n미배정 객체:", len(unresolved))
    print("검토할 Workspace:", len(workspace_review))
    print("객체 중복·누락 검사 통과")
    print("저장 위치:", OUTPUT_PATH)
    print("원본 그래프와 선택 파일은 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
