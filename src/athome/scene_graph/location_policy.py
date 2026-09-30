"""Standalone objects that are not Search Locations (configs/data/search_locations.json).

Categories are classified with the official HM3DSem -> mpcat40 mapping
(Matterport3D taxonomy, shipped with habitat-sim). Building elements and
fixtures (walls, windows, curtains, pictures, lights, ...) are excluded:
objects are not put there, as HomeRobot restricts search to receptacle
furniture. Categories absent from the mapping are kept.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import FrozenSet, Optional

from athome.scene_graph.query import DEFAULT_EXCLUDED_CATEGORIES, normalize_category

REPO = Path(__file__).resolve().parents[3]


def excluded_categories(policy_path: Optional[Path]) -> FrozenSet[str]:
    """DEFAULT_EXCLUDED_CATEGORIES plus every HM3DSem category whose mpcat40
    class is excluded by the policy (none without a policy file)."""
    if policy_path is None:
        return frozenset(DEFAULT_EXCLUDED_CATEGORIES)
    policy_path = Path(policy_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    mapping = REPO / policy["category_mapping"]
    if hashlib.sha256(mapping.read_bytes()).hexdigest() != policy["category_mapping_sha256"]:
        raise ValueError("범주 매핑 파일이 기록과 다릅니다.")
    excluded = set(policy["excluded_mpcat40"])
    with mapping.open(encoding="utf-8", newline="") as f:
        by_category = {}
        for row in csv.DictReader(f, delimiter="\t"):
            by_category.setdefault(normalize_category(row["category"]), set()).add(row["mpcat40"])
    # A category is excluded only if all of its raw spellings map to excluded classes.
    return frozenset(DEFAULT_EXCLUDED_CATEGORIES) | frozenset(
        c for c, classes in by_category.items() if classes <= excluded)
