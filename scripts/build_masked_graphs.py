import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from athome.scene_graph.workspace_builder import build_workspace_graph

ROOT = Path(__file__).resolve().parents[1]
SCENE = "wcojb4TFT35"
INPUT_DIR = ROOT / f"outputs/hm3d/{SCENE}/masked_inputs_v2"
LABEL_DIR = ROOT / f"outputs/teacher/{SCENE}.masked_labels"
OUTPUT_DIR = ROOT / f"outputs/hm3d/{SCENE}/masked_graphs"

STRUCTURAL = {"wall", "floor", "ceiling", "staircase wall"}
UNVERIFIED_SURFACES = {"bench", "piano", "toilet"}


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


def category(obj):
    return " ".join(obj["semantic_tag"].casefold().split())


def main():
    config = read(ROOT / "configs/data/pilot_targets.json")
    catalog = read(ROOT / f"outputs/hm3d/{SCENE}.target_catalog.json")
    gt_ids = {
        t["target_category"]: {o["object_id"] for o in t["instances"]}
        for t in catalog["targets"]
    }

    # 전체 목표의 검증을 마친 뒤 결과를 저장합니다.
    pending = []

    for target in config["target_categories"]:
        inputs = read(INPUT_DIR / f"{target}.semantic_inputs.json")
        original_labels = read(LABEL_DIR / f"{target}.review.json")

        if original_labels["target_category"] != target:
            raise ValueError(f"{target}: 라벨 목표 불일치")
        if digest(inputs) != original_labels["masked_input_sha256"]:
            raise ValueError(f"{target}: 마스킹 입력이 변경됐습니다.")

        objects = {
            o["object_id"]: o
            for room in inputs["rooms"] for o in room["objects"]
        }
        if set(objects) & gt_ids[target]:
            raise ValueError(f"{target}: 목표 객체가 입력에 남아 있습니다.")

        labels = deepcopy(original_labels)
        edits = []
        unverified = []

        for room in labels["rooms"]:
            retained = []
            for source in room["workspace_sources"]:
                oid = source["source_object_id"]
                obj = objects[oid]

                if obj["room_id"] != room["room_id"] if "room_id" in obj else False:
                    raise ValueError(f"{target}/{oid}: Source Room 불일치")

                tag = category(obj)
                if tag in STRUCTURAL:
                    raise ValueError(f"{target}/{oid}: 구조물이 Source로 선정됨")

                if tag == "tray":
                    edits.append({
                        "room_id": room["room_id"],
                        "source_object_id": oid,
                        "original_label": deepcopy(source),
                        "decision": "exclude_as_workspace_source",
                        "reason": "portable_tray_excluded_by_source_policy",
                    })
                    continue

                retained.append(source)
                if tag in UNVERIFIED_SURFACES:
                    unverified.append({
                        "room_id": room["room_id"],
                        "source_object_id": oid,
                        "category": tag,
                        "decision": "retain_provisionally",
                        "surface_geometry_verified": False,
                    })

            room["workspace_sources"] = retained

        labels["status"] = "reviewed_for_graph_assembly"
        labels["review"] = {
            "method": "masked_input_category_size_candidate_review",
            "is_ground_truth": False,
            "surface_geometry_verified": False,
            "changes": edits,
            "provisional_sources": unverified,
        }

        graph = build_workspace_graph(inputs, labels)
        graph_objects = {o["object_id"]: o for o in graph["objects"]}
        workspaces = {w["workspace_id"]: w for w in graph["workspaces"]}

        if len(graph_objects) != len(graph["objects"]):
            raise ValueError(f"{target}: 객체 중복")
        if set(graph_objects) != set(objects):
            raise ValueError(f"{target}: 객체 누락 또는 추가")

        linked_children = []
        source_ids = set()

        for wid, workspace in workspaces.items():
            sid = workspace["source_object_id"]
            source_ids.add(sid)
            source = graph_objects[sid]
            if source["role"] != "source" or source["parent_type"] != "room":
                raise ValueError(f"{target}/{sid}: Source 연결 오류")
            if source["workspace_id"] != wid:
                raise ValueError(f"{target}/{sid}: Workspace 참조 오류")

            for cid in workspace["child_object_ids"]:
                child = graph_objects[cid]
                if (
                    child["role"] != "child"
                    or child["parent_id"] != wid
                    or child["room_id"] != workspace["room_id"]
                    or category(child) in STRUCTURAL
                ):
                    raise ValueError(f"{target}/{cid}: Child 연결 오류")
                linked_children.append(cid)

        role_children = {
            oid for oid, obj in graph_objects.items() if obj["role"] == "child"
        }
        if (
            len(linked_children) != len(set(linked_children))
            or set(linked_children) != role_children
            or source_ids & role_children
        ):
            raise ValueError(f"{target}: Child 배정 중복 또는 누락")

        graph["provenance"] = {
            "target_category": target,
            "masked_input_sha256": digest(inputs),
            "reviewed_labels_sha256": digest(labels),
            "batch_id": labels["batch_id"],
            "review": labels["review"],
        }
        graph["association_policy"] = {
            "version": "0.2",
            "child_excluded_categories": sorted(STRUCTURAL),
        }
        pending.append((target, labels, graph))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 목표별 마스킹 그래프 ===")
    for target, labels, graph in pending:
        save(LABEL_DIR / f"{target}.reviewed.json", labels)
        save(OUTPUT_DIR / f"{target}.workspace_graph.json", graph)

        roles = Counter(o["role"] for o in graph["objects"])
        print(
            f"{target}: Room {len(graph['rooms'])}, "
            f"Workspace {len(graph['workspaces'])}, "
            f"Object {len(graph['objects'])}, "
            f"Child {roles['child']}, "
            f"Standalone {roles['standalone']}"
        )

    print("\n목표 객체 제거·객체 보존·Child 연결 검사 통과")
    print("생성한 그래프:", len(pending))
    print("저장 위치:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
