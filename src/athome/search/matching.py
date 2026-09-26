"""Decide whether an observed object is a requested target."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol, Sequence

import numpy as np

from athome.scene_graph.query import normalize_category
from athome.schemas import ObservedObject


class TargetMatcher(Protocol):
    def matches(self, target: str, obj: ObservedObject) -> bool: ...


class LabelMatcher:
    """Exact category match on the perception label.

    Placeholder until the CLIP text-embedding matcher (proposal 5-3) is
    wired in; needs targets given in the perception label vocabulary.
    """

    def __init__(self, min_confidence: float = 0.0):
        self.min_confidence = min_confidence

    def matches(self, target, obj):
        return (
            obj.confidence >= self.min_confidence
            and normalize_category(obj.label) == normalize_category(target)
        )


class ClipMatcher:
    """Cosine similarity between the target text embedding and the observed
    object's CLIP feature (proposal 5-3). ``threshold`` comes from validation.

    Objects without a feature, or targets whose text embedding could not be
    computed (server down), fall back to ``fallback``.
    """

    def __init__(
        self,
        encode_text: Callable[[Sequence[str]], Sequence[Sequence[float]]],
        threshold: float,
        fallback: Optional[TargetMatcher] = None,
        template: str = "a photo of a {}",
    ):
        self._encode = encode_text
        self.threshold = threshold
        self._fallback = fallback if fallback is not None else LabelMatcher()
        self._template = template
        self._cache: Dict[str, Optional[np.ndarray]] = {}
        self.errors: Dict[str, str] = {}

    def _embedding(self, target: str) -> Optional[np.ndarray]:
        if target not in self._cache:
            try:
                v = np.asarray(self._encode([self._template.format(target)])[0], float)
                self._cache[target] = v / np.linalg.norm(v)
            except Exception as e:  # noqa: BLE001 - any failure -> fallback
                self._cache[target] = None
                self.errors[target] = str(e)
        return self._cache[target]

    def similarity(self, target: str, obj: ObservedObject) -> Optional[float]:
        text = self._embedding(target)
        if text is None or not obj.clip_feature:
            return None
        image = np.asarray(obj.clip_feature, float)
        if image.shape != text.shape:
            return None
        return float(text @ image / np.linalg.norm(image))

    def matches(self, target, obj):
        s = self.similarity(target, obj)
        if s is None:
            return self._fallback.matches(target, obj)
        return s >= self.threshold
