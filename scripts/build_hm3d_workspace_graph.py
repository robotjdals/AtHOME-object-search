"""Unmasked workspace graph from semantic inputs and collected LLM labels.

Applies the shared source policy (src/athome/scene_graph/source_review.py)
to the collected labels, then builds and validates the graph.
"""
import argparse
import hashlib
import json
from pathlib import Path

from athome.scene_graph.source_review import (
    ASSOCIATION_POLICY, apply_source_policy, validate_workspace_graph)
from athome.scene_graph.workspace_builder import build_workspace_graph


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-inputs", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True,
                        help="collect_semantic_batch.py의 *.semantic_labels.review.json")
    parser.add_argument("--reviewed-output", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.reviewed_output, args.graph_output):
        if path.exists():
            raise RuntimeError(f"이미 존재하는 출력: {path}")
    inputs = json.loads(args.semantic_inputs.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if labels.get("status") != "pending_semantic_review":
        raise RuntimeError("수집 직후(pending_semantic_review) 라벨만 입력합니다.")
    objects = {o["object_id"]: o for room in inputs["rooms"] for o in room["objects"]}
    reviewed = apply_source_policy(labels, objects, "unmasked_input_category_size_candidate_review")
    graph = build_workspace_graph(inputs, reviewed)
    validate_workspace_graph(graph, objects, "unmasked")
    graph["association_policy"] = dict(ASSOCIATION_POLICY)
    graph["provenance"] = {
        "semantic_inputs": str(args.semantic_inputs.resolve()),
        "semantic_inputs_sha256": sha(args.semantic_inputs),
        "semantic_labels": str(args.labels.resolve()),
        "semantic_labels_sha256": sha(args.labels),
        "batch_id": labels["batch_id"], "review": reviewed["review"],
    }
    with args.reviewed_output.open("x", encoding="utf-8") as f:
        json.dump(reviewed, f, ensure_ascii=False, indent=2)
    with args.graph_output.open("x", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"Room {len(graph['rooms'])}, Workspace {len(graph['workspaces'])}, "
          f"Object {len(graph['objects'])}")
    print("검토 변경:", reviewed["review"]["changes"])
    print("잠정 Source:", [s["source_object_id"] for s in reviewed["review"]["provisional_sources"]])


if __name__ == "__main__":
    main()
