"""Debugging view of the search: the camera image of each perception frame
with its detections as the search judges them, and the search status on
top, published as /athome/search/debug_image/compressed (RViz Image display,
transport compressed).

Detections carry 3D boxes only (ObservedObjectArray), so the boxes are
projected into the image with the camera model of the real-time perception
node (configs/perception/realtime.yaml: K, T_cam_body) and the map -> body
pose at the capture time. Colours: green found, yellow possible target
(passes the found check in this frame; the visit still needs its
confirmation frames), red CLIP rank too low, orange below the confidence
floor or outside the search area, grey other objects.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage
from tf2_ros import Buffer, TransformException, TransformListener

from athome.perception.lifting import transform

FONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"   # Hangul status text
MIN_PERIOD_S = 0.2
STYLE = {  # reason -> (colour, width, caption)
    "found": ((0, 220, 0), 5, "발견"),
    "ok": ((255, 215, 0), 4, "후보"),
    "clip": ((230, 40, 40), 3, "CLIP"),
    "confidence": ((255, 140, 0), 2, "신뢰도 미달"),
    "outside": ((255, 140, 0), 2, "구역 밖"),
    "label": ((170, 170, 170), 1, ""),
}
PRIORITY = ["found", "ok", "clip", "outside", "confidence", "label"]


def _font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def box_corners(center, size, yaw) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy, dz = (np.asarray(size, float) / 2)
    local = np.array([[x, y, z] for x in (-dx, dx) for y in (-dy, dy) for z in (-dz, dz)])
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ R.T + np.asarray(center, float)


def project_box(K, T_cam_map, corners, width, height) -> Optional[tuple]:
    """Pixel rectangle of the box corners in front of the camera, clipped to
    the image; None if the box is behind the camera or outside the image."""
    P = transform(T_cam_map, corners)
    P = P[P[:, 2] > 0.1]
    if len(P) == 0:
        return None
    uv = (P @ K.T)[:, :2] / P[:, 2:3]
    u0, v0 = np.clip(uv.min(axis=0), 0, [width - 1, height - 1])
    u1, v1 = np.clip(uv.max(axis=0), 0, [width - 1, height - 1])
    if u1 - u0 < 2 or v1 - v0 < 2:
        return None
    return float(u0), float(v0), float(u1), float(v1)


class DebugView:
    def __init__(self, node: Node, camera_config: Path, image_at: Callable[[float], Optional[bytes]],
                 map_frame: str = "map", topic: str = "/athome/search/debug_image/compressed"):
        cfg = yaml.safe_load(Path(camera_config).read_text(encoding="utf-8"))
        cam = cfg["camera"]
        self._K = np.array(cam["K"], float)
        self._T_cam_body = np.array(cam["T_cam_body"], float)
        self._size = (int(cam["width"]), int(cam["height"]))
        self._body = cfg["frames"]["body"]
        self._map = map_frame
        self._image_at = image_at
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, node)
        self._pub = node.create_publisher(CompressedImage, topic, 1)
        self._node = node
        self._last = 0.0
        self._big, self._small = _font(26), _font(18)

    def render(self, frame, explain: Callable, status: dict, found: Sequence,
               save_to: Optional[Path] = None) -> bool:
        """Draw ``frame``'s detections on its camera image with the search
        ``status`` and publish it. ``explain(obj)`` -> reason (STYLE keys,
        "clip:<rank>"); ``found``: objects found in this command; ``save_to``:
        also write the view there (drawn even without a subscriber)."""
        now = time.monotonic()
        if save_to is None and (now - self._last < MIN_PERIOD_S or self._pub.get_subscription_count() == 0):
            return False
        data = self._image_at(frame.stamp)
        if data is None:
            return False
        try:
            tf = self._tf.lookup_transform(self._map, self._body, Time(nanoseconds=int(frame.stamp * 1e9)))
        except TransformException:
            return False
        self._last = now
        t, q = tf.transform.translation, tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        T_map_body = np.eye(4)
        T_map_body[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                              [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                              [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        T_map_body[:3, 3] = [t.x, t.y, t.z]
        T_cam_map = self._T_cam_body @ np.linalg.inv(T_map_body)

        img = Image.open(io.BytesIO(data)).convert("RGB")
        sx, sy = img.width / self._size[0], img.height / self._size[1]
        K = self._K * np.array([[sx], [sy], [1.0]])
        d = ImageDraw.Draw(img)
        found_c = [tuple(o.centroid) for o in found]
        drawn = []
        for o in frame.objects:
            if not o.is_static or o.bbox_center is None or o.bbox_size is None:
                continue
            if tuple(o.centroid) in found_c:
                reason = "found"
            else:
                reason = explain(o)
            kind = reason.split(":")[0]
            rect = project_box(K, T_cam_map, box_corners(o.bbox_center, o.bbox_size, o.bbox_yaw),
                               img.width, img.height)
            if rect is None:
                continue
            drawn.append((PRIORITY.index(kind), rect, o, reason, kind))
        for _, rect, o, reason, kind in sorted(drawn, key=lambda r: -r[0]):   # important ones on top
            colour, width, caption = STYLE[kind]
            d.rectangle(rect, outline=colour, width=width)
            note = (f"CLIP {reason.split(':')[1]}위" if kind == "clip" else caption)
            label = f"{o.label} {o.confidence:.2f}" + (f" {note}" if note else "")
            ty = max(rect[1] - 22, 0)
            d.rectangle((rect[0], ty, rect[0] + d.textlength(label, font=self._small) + 6, ty + 22), fill=colour)
            d.text((rect[0] + 3, ty + 1), label, fill=(0, 0, 0), font=self._small)

        lines = [status.get("text", "")]
        targets = ", ".join(f"{t['name']} {t['text']}" for t in status.get("targets", []))
        info = f"목표: {targets}" if targets else ""
        if status.get("last"):
            info += ("   |   " if info else "") + f"직전: {status['last']}"
        if info:
            lines.append(info)
        lines.append(f"영상 {time.strftime('%H:%M:%S', time.localtime(frame.stamp))}  "
                     "초록 발견 · 노랑 후보 · 빨강 CLIP 탈락 · 주황 신뢰도/구역 · 회색 기타")
        # Status above the image, so no detection is hidden under it.
        height = 46 + 28 * (len(lines) - 1)
        view = Image.new("RGB", (img.width, img.height + height), (0, 0, 0))
        view.paste(img, (0, height))
        d = ImageDraw.Draw(view)
        d.text((10, 8), lines[0], fill=(255, 255, 255), font=self._big)
        for i, line in enumerate(lines[1:]):
            d.text((10, 44 + 28 * i), line, fill=(200, 200, 200), font=self._small)

        out = io.BytesIO()
        view.save(out, format="JPEG", quality=75)
        msg = CompressedImage()
        msg.header.stamp = Time(nanoseconds=int(frame.stamp * 1e9)).to_msg()
        msg.header.frame_id = "camera"
        msg.format = "jpeg"
        msg.data = out.getvalue()
        self._pub.publish(msg)
        if save_to is not None:
            save_to.parent.mkdir(parents=True, exist_ok=True)
            save_to.write_bytes(msg.data)
        return True
