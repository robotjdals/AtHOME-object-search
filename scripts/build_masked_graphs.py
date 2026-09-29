import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from athome.scene_graph.source_review import (
    ASSOCIATION_POLICY, apply_source_policy, validate_workspace_graph)
from athome.scene_graph.workspace_builder import build_workspace_graph

ROOT = Path(__file__).resolve().parents[1]
SCENE = "wcojb4TFT35"
INPUT_DIR = ROOT / f"outputs/hm3d/{SCENE}/masked_inputs_v2"
LABEL_DIR = ROOT / f"outputs/teacher/{SCENE}.masked_labels"
OUTPUT_DIR = ROOT / f"outputs/hm3d/{SCENE}/masked_graphs"
CATALOG = ROOT / f"outputs/hm3d/{SCENE}.target_catalog.json"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--label-dir", type=Path, default=LABEL_DIR,
                        help="<target>.review.json 위치; <target>.reviewed.json도 여기에 저장")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/pilot_targets.json",
                        help="목표 범주 파일 (실행기는 장면별 <scene>.targets.json)")
    args = parser.parse_args()
    config = read(args.config)
    catalog = read(args.catalog)
    gt_ids = {
        t["target_category"]: {o["object_id"] for o in t["instances"]}
        for t in catalog["targets"]
    }

    # 전체 목표의 검증을 마친 뒤 결과를 저장합니다.
    pending = []

    for target in config["target_categories"]:
        inputs = read(args.input_dir / f"{target}.semantic_inputs.json")
        original_labels = read(args.label_dir / f"{target}.review.json")

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

        labels = apply_source_policy(
            original_labels, objects, "masked_input_category_size_candidate_review")
        graph = build_workspace_graph(inputs, labels)
        validate_workspace_graph(graph, objects, target)

        graph["provenance"] = {
            "target_category": target,
            "masked_input_sha256": digest(inputs),
            "reviewed_labels_sha256": digest(labels),
            "batch_id": labels["batch_id"],
            "review": labels["review"],
        }
        graph["association_policy"] = dict(ASSOCIATION_POLICY)
        pending.append((target, labels, graph))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 목표별 마스킹 그래프 ===")
    for target, labels, graph in pending:
        save(args.label_dir / f"{target}.reviewed.json", labels)
        save(args.output_dir / f"{target}.workspace_graph.json", graph)

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
    print("저장 위치:", args.output_dir)


if __name__ == "__main__":
    main()
