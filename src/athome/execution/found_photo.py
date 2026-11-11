"""The photo of a found target: the camera image with the found object's box
and a header line, so the operator can see what the search took for the target.
Without ROS so it can be tested (athome_ros.found_images saves it)."""

from __future__ import annotations

from typing import Optional, Sequence

try:
    import cv2
    import numpy as np
except ImportError:  # the raw JPEG is still written
    cv2 = None

BOX = (0, 220, 0)
TEXT = (0, 255, 255)
HEADER_PX = 44


def draw_found(jpeg: bytes, header: str, rect: Optional[Sequence[float]], label: str) -> Optional[bytes]:
    """JPEG with the header line and, given ``rect`` (u0, v0, u1, v1 in pixels),
    the box and its label; None if the image cannot be drawn on."""
    if cv2 is None:
        return None
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if rect is not None:
        u0, v0, u1, v1 = (int(round(v)) for v in rect)
        cv2.rectangle(img, (u0, v0), (u1, v1), BOX, 4)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        # Above the box, or inside it where the box starts at the header.
        top = max(v0 - h - 12, HEADER_PX)
        cv2.rectangle(img, (u0, top), (u0 + w + 10, top + h + 12), BOX, -1)
        cv2.putText(img, label, (u0 + 5, top + h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], HEADER_PX), (0, 0, 0), -1)
    cv2.putText(img, header, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT, 2)
    ok, out = cv2.imencode(".jpg", img)
    return out.tobytes() if ok else None
