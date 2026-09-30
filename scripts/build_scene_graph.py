"""Build the demo-environment scene graph from perception Static Features.

    python3 scripts/build_scene_graph.py --static-features <file.json|dir> \
        --robot-config configs/robot/<env>.yaml --out outputs/<env>/graph.json

LLM labeling uses GPT-4.1 (OPENAI_API_KEY). ``--labels`` uses reviewed labels
from a file instead; ``--save-labels`` stores the LLM labels for review.
Writes the graph, the labels, the room map and an overview image (graph and
goal poses on the map, athome.scene_graph.overview) next to --out.
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
from athome.scene_graph.overview import describe_problems, write_overview  # noqa: E402
from athome.scene_graph.pipeline import BuildConfig, build_scene_graph  # noqa: E402
from athome.scene_graph.semantic_labeling import MODEL, OpenAIChat  # noqa: E402
from athome.scene_graph.static_features import load_static_features  # noqa: E402
from athome.scene_graph.vocabulary import Vocabulary  # noqa: E402


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
    parser.add_argument("--llm-model", default=MODEL)
    args = parser.parse_args()

    config = load_robot_config(args.robot_config)
    if config.floor_z_m is None:
        # Without it the 1.5 m search-height scope of the training data is off.
        raise SystemExit("robot config에 map.floor_z_m(지도 좌표계 바닥 높이) 필요")
    features = load_static_features(args.static_features)
    vocabulary = Vocabulary.load(config.target_categories, config.search_locations)
    unmapped = vocabulary.unmapped(o.label for o in features.objects)
    if unmapped:
        # Search Location policy and storage scope classify HM3DSem names only.
        print(f"경고: HM3DSem 범주 표에 없는 라벨 {len(unmapped)}개 "
              f"(탐색 위치 정책·수납 범위 규칙 미적용): {unmapped}")
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

        def sample(messages, schema, n, temperature):
            # Stored (already voted) labels: every sample is the same answer.
            rid = schema["properties"]["room_id"]["enum"][0]
            if rid not in given:
                raise SystemExit(f"--labels에 {rid} 없음 (Room 분할이 바뀌었을 수 있음)")
            label = {k: given[rid][k] for k in ("room_id", "room_label", "workspace_sources")}
            return [label] * n
    else:
        sample = OpenAIChat(args.llm_model).sample

    scope = (json.loads(config.scene_scope.read_text(encoding="utf-8"))
             if config.scene_scope else None)
    if scope is None:
        print("경고: scene_graph.scene_scope 없음 - 수납가구 안 물체가 그래프에 남음 (학습 그래프와 다름)")
    result = build_scene_graph(
        features, raw.free, raw.origin, raw.resolution, sample,
        BuildConfig(scene_scope=scope, floor_z_m=config.floor_z_m))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result.graph)
    labels_path = args.out.with_name(args.out.stem + ".labels.json")
    write_json(labels_path, result.labels)
    save_room_map(args.out.with_name(args.out.stem + ".rooms.pgm"), result.segmentation.labels)
    # Machine-readable room map: the search server tracks the observed part of
    # each room with it (athome.search.coverage).
    np.savez_compressed(args.out.with_name(args.out.stem + ".rooms.npz"),
                        labels=result.segmentation.labels,
                        origin_xy_m=np.asarray(result.segmentation.origin, dtype=float),
                        resolution_m=np.asarray(result.segmentation.resolution, dtype=float))

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
    excluded = result.graph["provenance"]["scope_excluded_objects"]
    if excluded:
        print(f"수납가구 안 물체 {len(excluded)}개 제외: {excluded}")
    if features.features is not None:
        print(f"CLIP feature: {features.features.shape[1]}차원")
    overview = args.out.with_name(args.out.stem + ".overview.png")
    for line in describe_problems(write_overview(config, args.out, overview)):
        print(line)
    print("저장:", args.out, "/", labels_path.name, "/", overview.name)


if __name__ == "__main__":
    main()
