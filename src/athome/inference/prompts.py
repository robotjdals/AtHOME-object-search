"""Planner prompt format (proposal 6-3).

Single source for Teacher data generation, SFT and online planning: the
same input format must be used in all three.

Candidates are shown with short aliases (C1, C2, ...) instead of graph IDs,
which cuts tokens and keeps the output space small.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from athome.search.policy import Candidate, PlanningContext, Stage

PROMPT_VERSION = "0.1"

SYSTEM = (
    "You are the search planner of a home service robot looking for an object "
    "whose location is unknown. Choose the single candidate where the target is "
    "most likely to be found, also considering the travel cost. "
    'Answer only with JSON: {"selected_id": "<candidate id>"}.'
)


def _join(items) -> str:
    items = [i for i in items if i]
    return ", ".join(items) if items else "none"


def _room_lines(c: Candidate) -> List[str]:
    info = c.info
    workspaces = [
        f"{w['category']} ({w['function_label']})" if w.get("function_label") else w["category"]
        for w in info.get("workspaces", [])
    ]
    return [
        f"Room label: {info.get('room_label', 'unknown')}",
        f"Workspaces: {_join(workspaces)}",
        f"Standalone objects: {_join(info.get('standalone_categories', []))}",
        f"Min A* path cost: {c.cost:.1f} m",
    ]


def _location_lines(c: Candidate) -> List[str]:
    info = c.info
    lines = [f"Object: {info.get('category', 'unknown')}"]
    if info.get("kind") == "workspace":
        lines.append(f"Function: {info.get('function_label') or 'unknown'}")
        lines.append(f"Objects on it: {_join(info.get('child_categories', []))}")
    lines.append(f"A* path cost: {c.cost:.1f} m")
    return lines


def build_messages(
    stage: Stage,
    target: str,
    candidates: Sequence[Candidate],
    context: PlanningContext,
) -> Tuple[List[dict], Dict[str, str]]:
    """Chat messages and alias -> candidate_id map."""
    if stage == Stage.KNOWN:
        raise ValueError("Known 단계는 Planner를 사용하지 않음")
    aliases = {f"C{i}": c.candidate_id for i, c in enumerate(candidates, 1)}

    lines = [f"Target: {target}"]
    if stage == Stage.ROOM:
        lines.append(f"Explored rooms: {_join(context.explored_room_labels)}")
        lines.append(f"Last searched: {context.last_location_label or 'none'}")
        lines.append("")
        lines.append("Candidate rooms:")
        describe = _room_lines
    else:
        kind = "workspaces" if stage == Stage.WORKSPACE else "standalone objects"
        lines.append(f"Current room: {context.room_label or 'unknown'}")
        lines.append(f"Last searched: {context.last_location_label or 'none'}")
        lines.append("")
        lines.append(f"Candidate {kind}:")
        describe = _location_lines

    for alias, c in zip(aliases, candidates):
        lines.append(f"- {alias}")
        lines.extend(f"  {line}" for line in describe(c))

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]
    return messages, aliases
