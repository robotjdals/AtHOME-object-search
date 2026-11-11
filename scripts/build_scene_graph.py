"""Build the demo-environment scene graph from perception Static Features.

    python3 scripts/build_scene_graph.py --static-features <file.json|dir> \
        --robot-config configs/robot/<env>.yaml --out outputs/<env>/graph.json

LLM labeling uses GPT-4.1 (OPENAI_API_KEY, from --env-file). ``--labels`` uses reviewed labels
from a file instead; ``--save-labels`` stores the LLM labels for review.
Writes the graph, the labels, the room map and an overview image (graph and
goal poses on the map, athome.scene_graph.overview) next to --out.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from athome.config import load_robot_config, map_version_of  # noqa: E402
from athome.navigation import load_map_server, load_occupancy  # noqa: E402
from athome.scene_graph.overview import describe_problems, write_overview  # noqa: E402
from athome.scene_graph.pipeline import BuildConfig, build_scene_graph  # noqa: E402
from athome.scene_graph.semantic_labeling import MODEL, OpenAIChat  # noqa: E402
from athome.scene_graph.static_features import correct_labels, load_static_features  # noqa: E402
from athome.scene_graph.vocabulary import Vocabulary  # noqa: E402
from athome.scene_graph.wall_map import (  # noqa: E402
    WallBand,
    read_pcd_xyz,
    room_map_free_space,
    same_grid,
    segmentation_free_space,
)


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
    parser.add_argument("--room-types", help="과제로 정한 방 종류: room_1=bathroom,room_2=kitchen (라벨링 프로토콜 v2_given)")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env",
                        help="OPENAI_API_KEY (read by this tool, never sourced into the shell)")
    args = parser.parse_args()
    if not args.labels:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)     # the process environment wins

    config = load_robot_config(args.robot_config)
    if config.floor_z_m is None:
        # Without it the 1.5 m search-height scope of the training data is off.
        raise SystemExit("robot config에 map.floor_z_m(지도 좌표계 바닥 높이) 필요")
    features = load_static_features(args.static_features)
    corrected = []
    if config.label_corrections is not None:
        corrected = correct_labels(features, json.loads(config.label_corrections.read_text(encoding="utf-8")))
        print("사전 지도 수정: " + ", ".join(
            f"{c['source_id']} {c['from']} " + ("제거" if c.get("removed") else f"→ {c['label']}") for c in corrected))
    vocabulary = Vocabulary.load(config.target_categories, config.search_locations, config.label_aliases)
    unmapped = vocabulary.unmapped(o.label for o in features.objects)
    if unmapped:
        # Search Location policy and storage scope classify HM3DSem names only.
        print(f"경고: HM3DSem 범주 표에 없는 라벨 {len(unmapped)}개 "
              f"(탐색 위치 정책·수납 범위 규칙 미적용): {unmapped}")
    built_on = features.built_on_map(config.map_yaml)
    edited_from = None
    if built_on is False and config.map_edited_from is not None:
        # The navigation map is an edit (e.g. demo partitions) of the map the
        # objects were built on: valid if that is the recorded map and the
        # edit keeps its grid (same cells, same frame).
        if not features.built_on_map(config.map_edited_from):
            raise SystemExit(f"Static Feature는 map.edited_from({config.map_edited_from.name})로도 "
                             "만들어지지 않음")
        base, edit = load_occupancy(config.map_edited_from), load_occupancy(config.map_yaml)
        if not same_grid(base, edit):
            raise SystemExit(f"{config.map_yaml.name}이 {config.map_edited_from.name}과 격자가 다름 "
                             "(크기·원점·해상도): 물체 좌표를 그대로 쓸 수 없음")
        changed = int((base.occupancy != edit.occupancy).sum())
        edited_from = {"map": config.map_edited_from.name,
                       "map_version": map_version_of(config.map_edited_from),
                       "changed_cells": changed}
        print(f"지도: {config.map_yaml.name}은 {config.map_edited_from.name}의 편집본 "
              f"(같은 격자, 바뀐 칸 {changed}개) - 물체는 원본 지도에서 만들어짐")
        built_on = True
    if built_on is False:
        raise SystemExit(f"Static Feature는 다른 지도 파일로 만들어짐 (metadata.json의 sha256 "
                         f"≠ {config.map_yaml.name})")
    if built_on:
        features.map_version = config.map_version    # same files: record our version
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
    segmentation_free, segmentation_input = raw.free, {"source": "navigation map"}
    outside = None
    if config.room_map is not None:
        # Rooms drawn for the task (temporary partitions): walls and
        # partitions only, on the navigation map's grid. Its unknown cells
        # are outside the rooms (e.g. beyond the demo area).
        room_map = load_occupancy(config.room_map)
        # Door lines (room boundaries across open passages) are in the room
        # map only: the visibility map has them free.
        segmentation_free, missing = room_map_free_space(
            room_map, load_occupancy(config.map_yaml),
            load_occupancy(config.visibility_map) if config.visibility_map else None)
        outside = room_map.occupancy < 0
        if missing.any():
            raise SystemExit(
                f"방 경계 지도의 벽·파티션 {int(missing.sum())}칸이 주행 지도({config.map_yaml.name})에서 "
                "빈 칸: 두 지도에 같은 파티션을 그려야 함")
        segmentation_input = {
            "source": "room map", "room_map": config.room_map.name,
            "room_map_version": map_version_of(config.room_map)}
        print(f"방 분할: 방 경계 지도 {config.room_map.name}"
              + (" (map.pcd 대신)" if config.map_pcd is not None else ""))
    elif config.map_pcd is not None:
        # Rooms from walls only: furniture in the navigation map splits rooms.
        band = WallBand(*config.wall_band_m)
        segmentation_free, walls = segmentation_free_space(
            load_occupancy(config.map_yaml), read_pcd_xyz(config.map_pcd), config.floor_z_m, band)
        segmentation_input = {
            "source": "walls from map.pcd", "pcd": config.map_pcd.name,
            "pcd_sha256": hashlib.sha256(config.map_pcd.read_bytes()).hexdigest(),
            "wall_band_above_floor_m": [band.low, band.high], "wall_cells": int(walls.sum())}
        print(f"방 분할: {config.map_pcd.name}의 벽 (바닥 위 {band.low}~{band.high} m), "
              f"벽 칸 {int(walls.sum())}")
    else:
        print("경고: map.pcd 없음 - 가구까지 장애물인 주행 지도로 방 분할 (방이 쪼개질 수 있음)")

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
        features, segmentation_free, raw.origin, raw.resolution, sample,
        BuildConfig(scene_scope=scope, floor_z_m=config.floor_z_m, outside=outside,
                    given_room_labels=dict(kv.split("=", 1) for kv in args.room_types.split(","))
                    if args.room_types else None))
    result.graph["provenance"]["room_segmentation_input"] = segmentation_input
    if edited_from is not None:
        result.graph["provenance"]["map_edited_from"] = edited_from
    if corrected:
        result.graph["provenance"]["label_corrections"] = corrected

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
