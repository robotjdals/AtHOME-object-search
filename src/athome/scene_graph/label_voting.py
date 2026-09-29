"""Majority vote over sampled room annotations (self-consistency).

Measured on HM3D (outputs/teacher_v3/label_stability): with the v2 prompt,
room labels were stable but some workspace sources were selected in only a
fraction of samples, and the single temperature-0 answer was sometimes the
minority one. The label is therefore the majority over ``n`` samples:
- room_label: most frequent label;
- a source is kept if selected in at least ``min_votes`` samples;
- its function_label is the most frequent among the samples selecting it.
Ties are broken deterministically (vocabulary order / lexicographic) and
flagged. Vote counts are kept for provenance.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence


def _top(counter: Counter, order: Sequence[str] = ()):
    rank = {v: i for i, v in enumerate(order)}
    best = max(counter.values())
    tied = sorted((k for k, v in counter.items() if v == best),
                  key=lambda k: (rank.get(k, len(rank)), k))
    return tied[0], len(tied) > 1


def aggregate(samples: List[dict], min_votes: int, room_labels: Sequence[str] = ()) -> Dict:
    """``samples``: validated labels of one room. Returns label + ``votes``."""
    if not samples:
        raise ValueError("투표할 샘플이 없습니다.")
    room_ids = {s["room_id"] for s in samples}
    if len(room_ids) != 1:
        raise ValueError(f"샘플 room_id 불일치: {sorted(room_ids)}")
    n = len(samples)
    if not 0 < min_votes <= n:
        raise ValueError("min_votes 범위 오류")
    room_label, label_tie = _top(Counter(s["room_label"] for s in samples), room_labels)
    selected = Counter()
    functions: Dict[str, Counter] = {}
    for sample in samples:
        ids = [x["source_object_id"] for x in sample["workspace_sources"]]
        if len(ids) != len(set(ids)):
            raise ValueError("샘플 내 Source 중복")
        for item in sample["workspace_sources"]:
            selected[item["source_object_id"]] += 1
            functions.setdefault(item["source_object_id"], Counter())[item["function_label"]] += 1
    sources, function_ties = [], []
    for oid in sorted(selected):
        if selected[oid] < min_votes:
            continue
        function, tie = _top(functions[oid])
        if tie:
            function_ties.append(oid)
        sources.append({"source_object_id": oid, "function_label": function})
    return {
        "room_id": room_ids.pop(),
        "room_label": room_label,
        "workspace_sources": sources,
        "votes": {
            "samples": n, "min_votes": min_votes,
            "room_label": dict(Counter(s["room_label"] for s in samples)),
            "room_label_tie": label_tie,
            "sources": dict(sorted(selected.items())),
            "function_label_ties": function_ties,
        },
    }
