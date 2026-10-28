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
    ``category`` maps a label to its target category (Vocabulary.canonical:
    the training data's grouping of raw names, e.g. plurals).
    """

    def __init__(self, min_confidence: float = 0.0,
                 category: Callable[[str], str] = normalize_category,
                 target_keys: Optional[Callable[[str], Sequence[str]]] = None):
        """``target_keys``: categories that count as the target (the target
        and its subtypes, Vocabulary.target_keys); None: the target only."""
        self.min_confidence = min_confidence
        self._category = category
        self._target_keys = target_keys or (lambda t: (category(t),))

    def matches(self, target, obj):
        return (
            obj.confidence >= self.min_confidence
            and self._category(obj.label) in self._target_keys(target)
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


class ClipRankVerifier:
    """A detection labelled as the target counts only if CLIP agrees: the
    object's CLIP feature, classified zero-shot over a household vocabulary
    (the training target categories), puts the target within ``max_rank``.

    Open-vocabulary detectors given the target names as queries also return
    look-alikes under those names, and a look-alike is detected the same way
    at every frame, so seeing it again does not reject it; a second model
    does. Zero-shot ranking against other names is CLIP's own classification
    rule and needs no per-target similarity threshold.

    The target score is the best of its names (Vocabulary.target_keys); a
    target outside the vocabulary is ranked with its own name added. Without
    a feature or text embeddings (server down) the label decision stands
    (``unverified`` counts these).
    """

    def __init__(
        self,
        base: TargetMatcher,
        encode_text: Callable[[Sequence[str]], Sequence[Sequence[float]]],
        vocabulary: Sequence[str],
        max_rank: int,
        category: Callable[[str], str] = normalize_category,
        target_keys: Optional[Callable[[str], Sequence[str]]] = None,
        template: str = "a photo of a {}",
    ):
        if max_rank < 1:
            raise ValueError("max_rank는 1 이상")
        self._base = base
        self._encode = encode_text
        self._category = category
        self._target_keys = target_keys or (lambda t: (category(t),))
        self._names = sorted({category(n) for n in vocabulary})
        if not self._names:
            raise ValueError("CLIP 확인 어휘가 비어 있음")
        self.max_rank = max_rank
        self._template = template
        self._texts: Optional[np.ndarray] = None
        self._extra: Dict[str, Optional[np.ndarray]] = {}
        self.errors: Dict[str, str] = {}
        self.unverified = 0

    def _encode_names(self, names: Sequence[str]) -> np.ndarray:
        v = np.asarray(self._encode([self._template.format(n) for n in names]), float)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def _vocabulary(self) -> Optional[np.ndarray]:
        if self._texts is None:
            try:
                self._texts = self._encode_names(self._names)
            except Exception as e:  # noqa: BLE001 - retried at the next detection
                self.errors["vocabulary"] = str(e)
                return None
        return self._texts

    def rank(self, target: str, obj: ObservedObject) -> Optional[int]:
        """Rank of the target among the vocabulary for this object (1 = best)."""
        texts = self._vocabulary()
        if texts is None or not obj.clip_feature:
            return None
        image = np.asarray(obj.clip_feature, float)
        if image.shape != texts.shape[1:]:
            return None
        keys = set(self._target_keys(target))
        scores = texts @ image
        own = [i for i, n in enumerate(self._names) if n in keys]
        if own:
            best = max(scores[i] for i in own)
        else:
            if target not in self._extra:
                try:
                    self._extra[target] = self._encode_names(sorted(keys))
                except Exception as e:  # noqa: BLE001
                    self.errors[target] = str(e)
                    self._extra[target] = None
            if self._extra[target] is None:
                return None
            best = float(np.max(self._extra[target] @ image))
        others = np.array([s for n, s in zip(self._names, scores) if n not in keys])
        return 1 + int((others > best).sum())

    def matches(self, target, obj):
        if not self._base.matches(target, obj):
            return False
        r = self.rank(target, obj)
        if r is None:
            self.unverified += 1
            return True
        return r <= self.max_rank
