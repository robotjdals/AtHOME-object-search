"""Camera image of the moment a target was found, saved as a JPEG.

The found check runs after a visit on its observation frames
(SearchSession._confirmed); a frame's stamp is the capture time of the
camera image perception detected on. Images are kept briefly, and those of
frames with a possible target (SearchSession.is_candidate) until the
command ends, so the one of the confirming frame can be written when the
visit reports the target found.
"""

from __future__ import annotations

import io
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

from athome.execution.found_photo import draw_found
from athome_ros.debug_view import box_corners, project_box

# Perception results arrive within ~1 s of the capture (0.34 s median).
IMAGE_BUFFER_S = 10.0
MAX_CANDIDATE_IMAGES = 300


def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class FoundImages:
    def __init__(self, node: Node, topic: str, out_dir: Path):
        self._node = node
        self._out = Path(out_dir)
        self._recent = deque()                       # (stamp, jpeg bytes)
        self._kept: "OrderedDict[float, bytes]" = OrderedDict()
        # Camera pose of a kept image, taken when it was kept: the photo is
        # written when the visit ends, past the TF buffer (10 s).
        self._poses = {}
        node.create_subscription(CompressedImage, topic, self._on_image, qos_profile_sensor_data)

    def _on_image(self, msg: CompressedImage) -> None:
        t = _stamp(msg)
        self._recent.append((t, bytes(msg.data)))
        while self._recent and self._recent[0][0] < t - IMAGE_BUFFER_S:
            self._recent.popleft()

    def image_at(self, stamp: float) -> Optional[bytes]:
        """JPEG of the camera image captured at ``stamp`` (perception frames
        carry the capture time), if still buffered."""
        if stamp in self._kept:
            return self._kept[stamp]
        return next((data for t, data in reversed(self._recent) if abs(t - stamp) < 1e-3), None)

    def keep(self, stamp: float, camera_pose=None) -> None:
        """Keep the image of this frame (a possible target is in it), with
        the map -> camera transform at its capture."""
        if stamp in self._kept:
            return
        image = self.image_at(stamp)
        if image is None:
            return
        self._kept[stamp] = image
        if camera_pose is not None:
            self._poses[stamp] = camera_pose
        while len(self._kept) > MAX_CANDIDATE_IMAGES:
            old, _ = self._kept.popitem(last=False)
            self._poses.pop(old, None)

    def clear(self) -> None:
        self._kept.clear()
        self._poses.clear()

    def _box(self, stamp: float, obj, intrinsics, width: int, height: int):
        """Pixel rectangle of the object's 3D box in the kept image, None when
        it is unknown or out of view."""
        pose = self._poses.get(stamp)
        if pose is None or intrinsics is None or obj.bbox_center is None or obj.bbox_size is None:
            return None
        K, (cam_w, cam_h) = intrinsics
        K = K * np.array([[width / cam_w], [height / cam_h], [1.0]])
        return project_box(K, pose, box_corners(obj.bbox_center, obj.bbox_size, obj.bbox_yaw), width, height)

    def save(self, command_id: int, target: str, stamp: Optional[float], obj, intrinsics=None) -> Optional[Path]:
        """Write the image of ``stamp`` with a box on the found object, its
        label and confidence and the capture time; None without that image.
        ``intrinsics``: (K, (width, height)) of the camera (DebugView)."""
        image = self._kept.get(stamp) if stamp is not None else None
        if image is None:
            return None
        clock = time.strftime("%H:%M:%S", time.localtime(stamp))
        name = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(stamp))}_cmd{command_id}_{target.replace(' ', '_')}.jpg"
        self._out.mkdir(parents=True, exist_ok=True)
        path = self._out / name
        text = (f"{target}: {obj.label} {obj.confidence:.2f}  {clock}  "
                f"({obj.centroid[0]:.2f}, {obj.centroid[1]:.2f})")
        with Image.open(io.BytesIO(image)) as size:
            width, height = size.size
        rect = self._box(stamp, obj, intrinsics, width, height)
        if rect is None:
            text += "  (no box)"
        path.write_bytes(draw_found(image, text, rect, f"{obj.label} {obj.confidence:.2f}") or image)
        return path
