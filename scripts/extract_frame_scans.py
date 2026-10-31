"""LiDAR points around each dataset frame's capture time, in the map frame and
cropped to the camera view: the input a real-time node lifts masks with
(scripts/evaluate_detectors_412.py evaluate --scans).

The robot runs without camera depth (D435 over USB 2.0, run5 camera.log
"depth off"); 3D comes from the Mid-360 (FAST-LIO /cloud_registered_body).
Scans whose stamp lies within --window seconds of the frame stamp are moved
to the map frame with the bag's TF at their own stamp and accumulated (the
non-repetitive Mid-360 pattern fills in over several scans).

ROS 2 environment (rosbag2_py, tf2_ros):

    source /opt/ros/humble/setup.bash
    python3 scripts/extract_frame_scans.py \\
        --bag data/real_run5/real_run5_20261028/robot_recording/bag \\
        --out outputs/perception_bench/run5_scans.npz
"""
import argparse
import bisect
import csv
from pathlib import Path

import numpy as np
import rclpy.duration
import rclpy.time
import rosbag2_py
import tf2_ros
import yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_msgs.msg import TFMessage

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/perception/search412_20261030_zip/search412_dataset_20261030"
CALIB = ROOT / "data/real_run5/real_run5_20261028/robot_recording/calibration.yaml"
W, H = 1280, 720


def matrix(t):
    q, p = t.transform.rotation, t.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def reader(bag, topics):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
           rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    r.set_filter(rosbag2_py.StorageFilter(topics=topics))
    while r.has_next():
        yield r.read_next()


def stamp_ns(header):
    return header.stamp.sec * 10**9 + header.stamp.nanosec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", type=Path, required=True)
    p.add_argument("--frames", type=Path, default=DATA / "frames/frames.csv")
    p.add_argument("--calib", type=Path, default=CALIB)
    p.add_argument("--cloud-topic", default="/cloud_registered_body")
    p.add_argument("--map-frame", default="map")
    p.add_argument("--body-frame", default="body")
    p.add_argument("--window", type=float, default=0.5, help="seconds either side of the frame stamp")
    p.add_argument("--before", type=float, help="seconds before the stamp (default --window)")
    p.add_argument("--after", type=float, help="seconds after the stamp (default --window); 0 = what a live node "
                   "has at capture time without waiting")
    p.add_argument("--max-range", type=float, default=8.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    c = yaml.safe_load(args.calib.read_text(encoding="utf-8"))
    k = c["camera"]["color_1280x720"]
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
    T_cam_body = np.array(c["extrinsics"]["T_cam_body"], dtype=float)
    frames = list(csv.DictReader(open(args.frames, encoding="utf-8")))
    stamps = [int(round(float(f["stamp"]) * 1e9)) for f in frames]
    order = sorted(stamps)
    before = int((args.window if args.before is None else args.before) * 1e9)
    after = int((args.window if args.after is None else args.after) * 1e9)

    buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=3600))
    for _, data, _ in reader(args.bag, ["/tf_static"]):
        for t in deserialize_message(data, TFMessage).transforms:
            buffer.set_transform_static(t, "bag")
    for _, data, _ in reader(args.bag, ["/tf"]):
        for t in deserialize_message(data, TFMessage).transforms:
            buffer.set_transform(t, "bag")

    def lookup(target, source, ns):
        return matrix(buffer.lookup_transform(target, source, rclpy.time.Time(nanoseconds=ns)))

    scans = []          # (stamp, points in map)
    for _, data, _ in reader(args.bag, [args.cloud_topic]):
        msg = deserialize_message(data, PointCloud2)
        ns = stamp_ns(msg.header)
        i = bisect.bisect_left(order, ns - after)
        if i == len(order) or order[i] > ns + before:
            continue
        P = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True).astype(float)
        T = lookup(args.map_frame, msg.header.frame_id, ns)
        scans.append((ns, P @ T[:3, :3].T + T[:3, 3]))
    scan_stamps = [s for s, _ in scans]

    out, missing = {}, []
    for f, ns in zip(frames, stamps):
        i = int(f["frame"])
        try:
            T_cam = lookup(args.map_frame, args.body_frame, ns) @ np.linalg.inv(T_cam_body)
        except tf2_ros.TransformException:
            missing.append(i)
            continue
        lo, hi = bisect.bisect_left(scan_stamps, ns - before), bisect.bisect_right(scan_stamps, ns + after)
        P = np.concatenate([scans[j][1] for j in range(lo, hi)]) if hi > lo else np.zeros((0, 3))
        Ti = np.linalg.inv(T_cam)
        Pc = P @ Ti[:3, :3].T + Ti[:3, 3]
        z = Pc[:, 2]
        ok = (z > 0.3) & (z < args.max_range)
        uv = np.zeros((len(P), 2))
        uv[ok] = (Pc[ok] @ K.T)[:, :2] / z[ok, None]
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        out[f"points_{i}"] = P[ok].astype(np.float32)
        out[f"T_cam_{i}"] = T_cam
        out[f"scans_{i}"] = np.array(hi - lo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, before_s=before / 1e9, after_s=after / 1e9, **out)
    n = [int(out[k]) for k in out if k.startswith("scans_")]
    pts = [len(out[k]) for k in out if k.startswith("points_")]
    print(f"{len(n)} frames (no TF: {missing}), scans per frame {min(n, default=0)}-{max(n, default=0)}, "
          f"points in view median {int(np.median(pts)) if pts else 0} -> {args.out}")


if __name__ == "__main__":
    main()
