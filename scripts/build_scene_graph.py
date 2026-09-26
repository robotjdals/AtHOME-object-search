"""Build the demo-environment scene graph from perception Static Features.

    python3 scripts/build_scene_graph.py --static-features <file.json|dir> \
        --robot-config configs/robot/<env>.yaml --out outputs/<env>/graph.json

LLM labeling uses GPT-4.1 (OPENAI_API_KEY). ``--labels`` uses reviewed labels
from a file instead; ``--save-labels`` stores the LLM labels for review.
Writes the graph, the labels and a room label map image next to --out.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from athome.config import load_robot_config  # noqa: E402
from athome.navigation import load_map_server  # noqa: E402
from athome.scene_graph.pipeline import build_scene_graph  # noqa: E402
from athome.scene_graph.semantic_labeling import OpenAIChat  # noqa: E402
from athome.scene_graph.static_features import load_static_features  # noqa: E402


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                   encoding="utf-8")
    tmp.replace(path)


def save_room_map(path: Path, labels: np.ndarray) -> None:
    """PGM with one gray level per room (0 = no room), top row = max y."""
    n = max(int(labels.max()), 1)
    image = np.where(labels > 0, 60 + labels * (180 // n), 0).astype(np.uint8)[::-1]
    h, w = image.shape
    path.write_bytes(b"P5\n%d %d\n255\n" % (w, h) + image.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-features", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="reviewed semantic labels JSON")
    parser.add_argument("--llm-model", default="gpt-4.1")
    args = parser.parse_args()

    config = load_robot_config(args.robot_config)
    features = load_static_features(args.static_features)
    if features.map_version is None:
        print(f"경고: Static Feature에 지도 버전 없음. {config.map_yaml.name}과 "
              "같은 지도 좌표계에서 만든 것인지 확인 필요")
    elif features.map_version != config.map_version:
        raise SystemExit(
            f"Static Feature 지도({features.map_version})와 현재 지도"
            f"({config.map_version})가 다름")

    # Raw free space: segmentation must not depend on the robot radius.
    raw = load_map_server(config.map_yaml, inflation_radius=0.0,
                          unknown_as_occupied=config.unknown_as_occupied)

    if args.labels:
        given = {r["room_id"]: r for r in json.loads(args.labels.read_text())["rooms"]}

        def complete(messages, schema):
            rid = schema["properties"]["room_id"]["enum"][0]
            if rid not in given:
                raise SystemExit(f"--labels에 {rid} 없음 (Room 분할이 바뀌었을 수 있음)")
            return given[rid]
    else:
        complete = OpenAIChat(args.llm_model)

    result = build_scene_graph(features, raw.free, raw.origin, raw.resolution, complete)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result.graph)
    labels_path = args.out.with_name(args.out.stem + ".labels.json")
    write_json(labels_path, result.labels)
    save_room_map(args.out.with_name(args.out.stem + ".rooms.pgm"), result.segmentation.labels)

    print(f"Room {len(result.segmentation.room_ids)}개 "
          f"(객체 없는 Room: {result.empty_rooms or '없음'})")
    for room in result.graph["rooms"]:
        area = result.segmentation.area(int(room["room_id"].split("_")[1]))
        print(f"  {room['room_id']}: {room['room_label']}, {area:.1f} m², "
              f"Workspace {len(room['workspace_ids'])}, "
              f"Standalone {len(room['standalone_object_ids'])}")
    methods = list(result.assignment.methods.values())
    print(f"객체 {len(features.objects)}개: centroid {methods.count('centroid')}, "
          f"nearest {methods.count('nearest')}, points {methods.count('points')}, "
          f"미배정 {len(result.assignment.unassigned)} {result.assignment.unassigned}")
    if features.features is not None:
        print(f"CLIP feature: {features.features.shape[1]}차원")
    print("저장:", args.out, "/", labels_path.name)


if __name__ == "__main__":
    main()
