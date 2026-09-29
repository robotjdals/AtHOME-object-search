"""Object label vocabulary shared with the training data.

Target categories of the dataset (configs/data/target_categories.v2.json,
scripts/build_target_categories.py) group raw HM3DSem names: a category
covers its ``raw_names``, and plural names are merged into the singular
(``merged_plurals``). The robot resolves targets and perception labels with
the same grouping, so "bags" finds a bag as in the training GT. Scene graph
tags (planner input) keep the raw names, as in the training graphs.

Labels outside the HM3DSem category mapping cannot be classified by the
Search Location policy or the storage scope rule; ``unmapped`` lists them so
the perception vocabulary can be aligned.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from athome.scene_graph.query import normalize_category

REPO = Path(__file__).resolve().parents[3]


class Vocabulary:
    def __init__(self, aliases: Mapping[str, str] = None,
                 mapped: Optional[FrozenSet[str]] = None,
                 categories: Iterable[str] = (),
                 hyponyms: Mapping[str, Iterable[str]] = None):
        self._aliases: Dict[str, str] = {
            normalize_category(k): normalize_category(v) for k, v in (aliases or {}).items()}
        self._mapped = mapped
        # Target categories of the training data (the planner's vocabulary).
        self.categories: List[str] = sorted({normalize_category(c) for c in categories})
        # Subtypes that count as the target ("lamp" is found by a bedside lamp),
        # as in the training GT and masking (target_categories hyponyms).
        self._hyponyms: Dict[str, Tuple[str, ...]] = {
            self.canonical(parent): tuple(sorted({self.canonical(c) for c in children}))
            for parent, children in (hyponyms or {}).items()}

    @classmethod
    def load(cls, target_categories=None, search_locations=None) -> "Vocabulary":
        aliases, categories, hyponyms = {}, [], {}
        if target_categories is not None:
            data = json.loads(Path(target_categories).read_text(encoding="utf-8"))
            aliases.update(data.get("merged_plurals", {}))
            categories = list(data["categories"])
            hyponyms = data.get("hyponyms", {})
            for category, info in data["categories"].items():
                for raw in info.get("raw_names", ()):
                    aliases[raw] = category
        mapped = None
        if search_locations is not None:
            policy = json.loads(Path(search_locations).read_text(encoding="utf-8"))
            with (REPO / policy["category_mapping"]).open(encoding="utf-8", newline="") as f:
                mapped = frozenset(normalize_category(row["category"])
                                   for row in csv.DictReader(f, delimiter="\t"))
        return cls(aliases, mapped, categories, hyponyms)

    def canonical(self, label: str) -> str:
        """Target category of a raw label (normalized label if ungrouped)."""
        n = normalize_category(label)
        return self._aliases.get(n, n)

    def target_keys(self, target: str) -> Tuple[str, ...]:
        """Categories that count as finding ``target``: itself and its subtypes."""
        t = self.canonical(target)
        return (t,) + self._hyponyms.get(t, ())

    def is_target(self, target: str, label: str) -> bool:
        return self.canonical(label) in self.target_keys(target)

    def in_training_vocabulary(self, label: str) -> bool:
        return self.canonical(label) in self.categories

    def unmapped(self, labels: Iterable[str]) -> List[str]:
        """Labels the HM3DSem category mapping does not contain."""
        if self._mapped is None:
            return []
        return sorted({normalize_category(l) for l in labels} - self._mapped)
