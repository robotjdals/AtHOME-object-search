"""Report of recorded search runs (rosbag of athome_bringup record.launch.py).

    ros2 run athome_ros run_report <bag dir> --robot-config $PWD/configs/robot/demo.yaml

Writes <bag>/report/ (or --out):
  report.md    per command: result, time, distance, visits, cost, planner
               queries, pauses and a table of the visits
  run<N>.png   robot path, numbered visits and found targets on the scene graph
  report.json  the metrics (combine runs: scripts/summarize_runs.py)
The robot path is map->base_link from the recorded TF. The scene graph and
map come from --robot-config, so it must be the setup the run used (checked
against the map version recorded at each command start).
"""

import argparse
import json
import math
from pathlib import Path

import rosbag2_py
import yaml
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from tf2_ros import Buffer, TransformException

from athome.config import load_robot_config
from athome.navigation import load_map_server, load_occupancy
from athome.run_report import build_runs, markdown, render_run, run_metrics
from athome.scene_graph.overview import load_robot_graph, load_room_map


def read_bag(path: Path, topics):
    """(topic, message) of ``topics`` in recorded order."""
    meta = yaml.safe_load((path / "metadata.yaml").read_text())
    storage = meta["rosbag2_bagfile_information"]["storage_identifier"]
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=storage),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    while reader.has_next():
        topic, data, _ = reader.read_next()
        yield topic, deserialize_message(data, get_message(types[topic]))


def sample_path(buffer: Buffer, map_frame: str, base_frame: str,
                start_s: float, end_s: float, rate_hz: float):
    """(t, x, y, yaw) of base_frame in map_frame at ``rate_hz``."""
    out, n = [], int((end_s - start_s) * rate_hz) + 1
    for i in range(n):
        t = start_s + i / rate_hz
        try:
            tf = buffer.lookup_transform(map_frame, base_frame, Time(nanoseconds=int(t * 1e9)))
        except TransformException:
            continue
        p, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        out.append((t, p.x, p.y, yaw))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="기본: <bag>/report")
    parser.add_argument("--step-cost", type=float, default=3.0,
                        help="비용 = 이동 거리 + step_cost x 방문 수 [m] (evaluate_student.py와 같은 값)")
    parser.add_argument("--rate", type=float, default=10.0, help="경로 표본 [Hz]")
    parser.add_argument("--events-topic", default="/athome/search/events")
    parser.add_argument("--decision-topic", default="/athome/planner/decision")
    args = parser.parse_args()

    config = load_robot_config(args.robot_config)
    events, decisions = [], []
    buffer = Buffer(cache_time=Duration(seconds=24 * 3600))    # the whole recording
    for topic, msg in read_bag(args.bag, [args.events_topic, args.decision_topic,
                                          "/tf", "/tf_static"]):
        if topic == args.events_topic:
            events.append(json.loads(msg.data))
        elif topic == args.decision_topic:
            decisions.append(json.loads(msg.data))
        else:
            for t in msg.transforms:
                if topic == "/tf_static":
                    buffer.set_transform_static(t, "bag")
                else:
                    buffer.set_transform(t, "bag")
    if not events:
        raise SystemExit(f"{args.bag}: {args.events_topic} 기록 없음 (record.launch.py로 녹화했는지 확인)")

    stamps = [e["stamp"] for e in events]
    samples = sample_path(buffer, config.map_frame, config.base_frame,
                          min(stamps), max(stamps), args.rate)
    runs = build_runs(events, decisions, samples)
    metrics = [run_metrics(r, args.step_cost) for r in runs]
    notes = []
    if not samples:
        notes.append(f"TF {config.map_frame}→{config.base_frame} 기록 없음: 경로와 이동 거리 없음")
    recorded = {r.setup.get("map_version") for r in runs} - {None}
    if recorded - {config.map_version}:
        notes.append(f"기록된 지도 {sorted(recorded)} ≠ 보고서 설정 지도 {config.map_version}: "
                     "그림의 지도·그래프가 실제와 다를 수 있음")

    out = args.out or args.bag / "report"
    out.mkdir(parents=True, exist_ok=True)
    _, graph = load_robot_graph(config, config.graph_path)
    occupancy = load_occupancy(config.map_yaml)
    grid = load_map_server(config.map_yaml, config.inflation_radius, config.unknown_as_occupied)
    rooms = load_room_map(config.graph_path)
    images = {}
    for run, m in zip(runs, metrics):
        images[run.run] = f"run{run.run}.png"
        render_run(out / images[run.run], run, m, graph, occupancy, grid, rooms,
                   title=args.bag.name)
    (out / "report.md").write_text(
        markdown(runs, metrics, graph, images, args.bag.name, notes), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(
        {"bag": str(args.bag), "robot_config": str(args.robot_config),
         "map_version": config.map_version, "notes": notes, "runs": metrics},
        ensure_ascii=False, indent=1), encoding="utf-8")

    for note in notes:
        print("주의:", note)
    for m in metrics:
        print(f"명령 {m['run']}: {m['status']}, 찾음 {m['found']}/{len(m['targets'])}, "
              f"{m['active_s']} s, {m['distance_m']} m, 방문 {m['visits']}, 비용 {m['cost_m']} m")
    print("저장:", out)


if __name__ == "__main__":
    main()
