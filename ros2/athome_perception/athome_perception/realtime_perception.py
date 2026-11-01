"""Real-time perception: each camera image goes to the detection server
(scripts/serve_detector.py, over an SSH tunnel) with the command's
vocabulary (base list + the query names of the targets the search publishes
on /athome/search/targets); each detection's mask is lifted to a 3D box with the LiDAR scans
of the last second, moved to the map frame with their own TF
(athome.perception.lifting); the result is published as
ObservedObjectArray, stamped with the image's capture time, like perception's
CG-D node (the search and the online scene graph read it unchanged).

The newest image is processed when the previous one is done (older ones are
dropped). No message is published while the server is unreachable, so the
search sees perception as lost and pauses; an image without detections
gives an empty message (liveness).

    ros2 run athome_perception realtime_perception --ros-args -p config:=configs/perception/realtime.yaml
"""
import base64
import json
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, Pose, Vector3
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener

import athome
from athome.perception.lifting import Camera, FrameView, ScanWindow, box_region, transform, unpack_mask
from athome_interfaces.msg import ObservedObject, ObservedObjectArray, SearchTargets

DEFAULT_CONFIG = Path(athome.__file__).resolve().parents[2] / "configs/perception/realtime.yaml"


def seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def matrix(t) -> np.ndarray:
    q, p = t.transform.rotation, t.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [p.x, p.y, p.z]
    return T


class RealtimePerception(Node):
    def __init__(self):
        super().__init__("athome_realtime_perception")
        path = Path(self.declare_parameter("config", str(DEFAULT_CONFIG)).value)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        cam = cfg["camera"]
        self._camera = Camera(np.array(cam["K"], float), cam["width"], cam["height"],
                              np.array(cam["T_cam_body"], float))
        self._T_body_cam = np.linalg.inv(self._camera.T_cam_body)
        self._url = cfg["server"]["url"].rstrip("/") + "/detect"
        self._timeout = cfg["server"]["timeout_s"]
        # The server re-encodes the vocabulary on the first image of a command.
        self._vocab_timeout = cfg["server"]["vocabulary_change_timeout_s"]
        self._sent_vocab = None
        self._map, self._body = cfg["frames"]["map"], cfg["frames"]["body"]
        lift = cfg["lifting"]
        self._before, self._after = lift["window_before_s"], lift["window_after_s"]
        self._max_range = lift["max_range_m"]
        self._dynamic = set(cfg["detections"]["dynamic_labels"])
        base = path.parent / cfg["vocabulary"]["base"]
        self._base = [l.strip() for l in base.read_text(encoding="utf-8").splitlines()
                      if l.strip() and not l.startswith("#")]
        self._targets = [t for t in self.declare_parameter("targets", [""]).value if t]

        self._tf = Buffer(cache_time=Duration(seconds=10.0))
        self._listener = TransformListener(self._tf, self)
        self._scans = ScanWindow(keep_s=max(3.0, self._before + self._after + 1.0))
        self._scan_lock = threading.Lock()
        self._latest = None
        self._wake = threading.Condition()
        self.stats = {"images": 0, "published": 0, "server_errors": 0, "tf_misses": 0, "unlifted": 0}
        # per published image: server round trip, server compute, lifting (s)
        self.timing = {"latency": [], "server": [], "detect": [], "lift": []}

        group = ReentrantCallbackGroup()
        topics = cfg["topics"]
        self.create_subscription(CompressedImage, topics["image"], self._on_image, qos_profile_sensor_data,
                                 callback_group=group)
        self.create_subscription(PointCloud2, topics["cloud"], self._on_cloud, qos_profile_sensor_data,
                                 callback_group=group)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(SearchTargets, topics["targets"], self._on_targets, latched, callback_group=group)
        self._pub = self.create_publisher(ObservedObjectArray, topics["output"], 10)
        self.create_timer(30.0, self._report, callback_group=group)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self.get_logger().info(f"{self._url} | base vocabulary {len(self._base)} | targets {self._targets}")

    def _report(self):
        """Every 30 s: counts and median stage times (ms) of the last images:
        capture -> publish, server round trip, server compute, lifting."""
        med = {k: round(1e3 * float(np.median(v[-60:])), 1) for k, v in self.timing.items() if v}
        self.get_logger().info(f"stats {self.stats} | median ms {med}")

    # ------------------------------------------------------------ inputs
    def _on_targets(self, msg):
        """The search's open targets: their query names (the labels its found
        check accepts) join the vocabulary; empty when no command runs."""
        targets = []
        for t in msg.targets:
            for n in t.query_names or [t.name]:
                if n not in targets:
                    targets.append(n)
        if targets != self._targets:
            self._targets = targets
            self.get_logger().info(f"targets {targets}")

    def _on_image(self, msg):
        with self._wake:
            self._latest = msg
            self._wake.notify()

    def _on_cloud(self, msg):
        try:
            T = matrix(self._tf.lookup_transform(self._map, msg.header.frame_id, Time.from_msg(msg.header.stamp),
                                                 timeout=Duration(seconds=0.2)))
        except TransformException:
            self.stats["tf_misses"] += 1
            return
        P = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True).astype(float)
        with self._scan_lock:
            self._scans.add(seconds(msg.header.stamp), transform(T, P))

    # ------------------------------------------------------------ worker
    def _run(self):
        while rclpy.ok():
            with self._wake:
                while self._latest is None and rclpy.ok():
                    self._wake.wait(timeout=0.5)
                msg, self._latest = self._latest, None
            if msg is not None:
                try:
                    self._process(msg)
                except Exception as e:  # noqa: BLE001 - keep the node alive, report
                    self.get_logger().error(f"{type(e).__name__}: {e}", throttle_duration_sec=5.0)

    def _detect(self, msg):
        targets = list(self._targets)
        vocab = self._base + [t for t in targets if t not in self._base]
        body = json.dumps({"image": base64.b64encode(bytes(msg.data)).decode("ascii"),
                           "vocab": vocab, "targets": targets}).encode()
        req = urllib.request.Request(self._url, data=body, headers={"Content-Type": "application/json"})
        timeout = self._timeout if vocab == self._sent_vocab else self._vocab_timeout
        with urllib.request.urlopen(req, timeout=timeout) as r:
            result = json.loads(r.read())
        self._sent_vocab = vocab
        return result

    def _process(self, msg):
        self.stats["images"] += 1
        stamp = seconds(msg.header.stamp)
        try:
            t0 = time.monotonic()
            result = self._detect(msg)
            t_server = time.monotonic() - t0
        except Exception as e:  # noqa: BLE001 - no message: the search pauses
            self.stats["server_errors"] += 1
            self.get_logger().warn(f"detection server: {e}", throttle_duration_sec=5.0)
            return
        if tuple(result["size"]) != (self._camera.width, self._camera.height):
            self.get_logger().error(f"image {result['size']} != camera config "
                                    f"{self._camera.width}x{self._camera.height}", throttle_duration_sec=5.0)
            return
        try:
            T_map_body = matrix(self._tf.lookup_transform(self._map, self._body, Time.from_msg(msg.header.stamp),
                                                          timeout=Duration(seconds=0.5)))
        except TransformException:
            self.stats["tf_misses"] += 1
            return
        deadline = time.monotonic() + self._after + 1.0
        while self._after > 0 and (self._scans.latest or 0) < stamp + self._after and time.monotonic() < deadline:
            time.sleep(0.02)
        with self._scan_lock:
            points = self._scans.points(stamp - self._before, stamp + self._after)
        t1 = time.monotonic()
        view = FrameView(self._camera, T_map_body @ self._T_body_cam, points, self._max_range)
        out = ObservedObjectArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._map
        out.clip_model = result.get("clip_model") or ""
        W, H = self._camera.width, self._camera.height
        for d in result["detections"]:
            region = unpack_mask(d["mask"], W, H) if d.get("mask") else box_region(d["box"], W, H)
            box = view.lift(region)
            if box is None:
                self.stats["unlifted"] += 1
                continue
            o = ObservedObject()
            o.object_id = -1                       # not linked to the pre-map; the online layer associates
            o.is_static = d["label"] not in self._dynamic
            o.label = d["label"]
            o.confidence = float(d["confidence"])
            o.centroid = Point(x=box.center[0], y=box.center[1], z=box.center[2])
            o.bbox_center = Pose()
            o.bbox_center.position = Point(x=box.center[0], y=box.center[1], z=box.center[2])
            o.bbox_center.orientation.w = 1.0
            o.bbox_size = Vector3(x=box.size[0], y=box.size[1], z=box.size[2])
            o.clip_feature = [float(v) for v in d.get("clip_feature", [])]
            out.objects.append(o)
        self._pub.publish(out)
        self.stats["published"] += 1
        latency = self.get_clock().now().nanoseconds * 1e-9 - stamp
        for k, v in (("latency", latency), ("server", t_server), ("detect", result["detect_ms"] / 1e3),
                     ("lift", time.monotonic() - t1)):
            self.timing[k] = (self.timing[k] + [v])[-500:]


def main():
    rclpy.init()
    node = RealtimePerception()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        med = {k: round(1e3 * float(np.median(v)), 1) for k, v in node.timing.items() if v}
        print(f"stats {node.stats} | median ms {med}", flush=True)   # the ROS logger is gone after SIGINT
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
