"""Camera image of the moment a target was found, saved as a JPEG.

The found check runs after a visit on its observation frames
(SearchSession._confirmed); a frame's stamp is the capture time of the
camera image perception detected on. Images are kept briefly, and those of
frames with a possible target (SearchSession.is_candidate) until the
command ends, so the one of the confirming frame can be written when the
visit reports the target found.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Optional

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

try:
    import cv2
    import numpy as np
except ImportError:  # the raw JPEG is still written
    cv2 = None

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

    def keep(self, stamp: float) -> None:
        """Keep the image of this frame (a possible target is in it)."""
        if stamp in self._kept:
            return
        image = self.image_at(stamp)
        if image is None:
            return
        self._kept[stamp] = image
        while len(self._kept) > MAX_CANDIDATE_IMAGES:
            self._kept.popitem(last=False)

    def clear(self) -> None:
        self._kept.clear()

    def save(self, command_id: int, target: str, stamp: Optional[float], obj) -> Optional[Path]:
        """Write the image of ``stamp`` with the target, its label and
        confidence and the capture time; None without that image."""
        image = self._kept.get(stamp) if stamp is not None else None
        if image is None:
            return None
        clock = time.strftime("%H:%M:%S", time.localtime(stamp))
        name = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(stamp))}_cmd{command_id}_{target.replace(' ', '_')}.jpg"
        self._out.mkdir(parents=True, exist_ok=True)
        path = self._out / name
        text = (f"{target}: {obj.label} {obj.confidence:.2f}  {clock}  "
                f"({obj.centroid[0]:.2f}, {obj.centroid[1]:.2f})")
        if cv2 is not None:
            img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
                cv2.putText(img, text, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imwrite(str(path), img)
                return path
        path.write_bytes(image)
        return path
