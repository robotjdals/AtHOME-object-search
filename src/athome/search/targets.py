"""Targets still searched by a command, with the labels that count as
finding them, for perception's detection queries (/athome/search/targets)."""

from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

from athome.search.session import TargetStatus

OPEN = (TargetStatus.PENDING, TargetStatus.SEARCHING)


def search_targets(targets: Iterable, target_keys: Callable[[str], Iterable[str]]
                   ) -> List[Tuple[str, Tuple[str, ...]]]:
    """(name, query names) of the targets not found or failed yet; query
    names are the labels the found check accepts (Vocabulary.target_keys)."""
    return [(t.name, tuple(target_keys(t.name))) for t in targets if t.status in OPEN]
