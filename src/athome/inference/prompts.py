"""Planner prompt format (proposal 6-3).

Single source for Teacher data generation, SFT and online planning: the
same input format must be used in all three.

Candidates are shown with short aliases (C1, C2, ...) instead of graph IDs,
which cuts tokens and keeps the output space small.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from athome.search.policy import Candidate, PlanningContext, Stage

# 0.2: the room stage marks the room the robot is in ("(current room)"), since
# the room is chosen again after every visit.
# 0.3: room candidates show how many of their locations were already searched
# without seeing the target (negative evidence, as MoMa-LLM shows unexplored
# areas and action history).
# 0.4: the room's observed floor fraction (Bayesian search: the searched
# region, not the number of visits) replaces the visited-location count when
# the coverage tracker is available.
# 0.5: repeated furniture and objects are shown once with a count
# ("shelf (general_storage) x5"), as MoMa-LLM summarizes matching nodes of a
# room with a counter; the count tells room size and storage capacity.
# 0.6: standalone candidates are grouped by category ("storage box x44"),
# with the path cost of the nearest instance, where the robot goes
# (athome.search.session.standalone_groups; MoMa-LLM navigates by object name).
PROMPT_VERSION = "0.6"

SYSTEM = (
    "You are the search planner of a home service robot looking for an object "
    "whose location is unknown. Choose the single candidate where the target is "
    "most likely to be found, also considering the travel cost. "
    'Answer only with JSON: {"selected_id": "<candidate id>"}.'
)

# Teacher (proposal 6-3). The Teacher sees the same planner input but judges
# only semantic likelihood per candidate, reasoning first; travel cost is
# combined outside the LLM (athome.training.teacher). This follows object-search
# planners that keep distance out of the LLM judgement and add it by formula
# (SG-Nav, LFG, ESC, VoroNav) and HSG-ON's likelihood ranking; asking one LLM
# to trade likelihood against metres made it pick nearby wrong rooms.
# 0.3: the Teacher judges the semantic prior only; search coverage is applied
# by Bayes' rule outside the LLM (athome.training.teacher), like travel cost.
# 0.4: the Teacher input leaves out the search coverage line, so the prior is
# not discounted twice (the LLM lowered it on its own in 0.3).
# 0.5: same input change as the Student prompt 0.5 (counts).
# 0.6: same input change as the Student prompt 0.6 (standalone groups).
TEACHER_PROMPT_VERSION = "0.6"
TEACHER_SYSTEM = (
    "You help a home service robot search for an object whose location is "
    "unknown. For every candidate, estimate the probability that the target "
    "will be found there, judging only from semantics: the room label, the "
    "furniture and its function, and the objects listed, with typical "
    "household placement of objects. Ignore the travel cost and how much of "
    "a room was already searched; both are taken into account separately. "
    "First write your reasoning, then give every "
    "candidate a likelihood between 0 and 1. Answer only with JSON."
)


def _join(items) -> str:
    """Comma list in first-seen order; repeats become one entry with a count."""
    counts: Dict[str, int] = {}
    for i in items:
        if i:
            counts[i] = counts.get(i, 0) + 1
    if not counts:
        return "none"
    return ", ".join(f"{i} x{n}" if n > 1 else i for i, n in counts.items())


def _coverage_line(info) -> str:
    if "observed_fraction" in info:
        return f"Observed: {round(100 * info['observed_fraction'])}% of this room, target not seen"
    return (f"Searched here: {info.get('searched_locations', 0)} of {info.get('total_locations', 0)} "
            "locations, target not seen")


def _room_lines(c: Candidate, coverage: bool = True) -> List[str]:
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
    ] + ([_coverage_line(info)] if coverage else [])


def _location_lines(c: Candidate) -> List[str]:
    info = c.info
    if info.get("kind") == "standalone_group":
        count = info.get("count", 1)
        return [f"Object: {info['category']}" + (f" x{count}" if count > 1 else ""),
                f"A* path cost: {c.cost:.1f} m" + (" (nearest)" if count > 1 else "")]
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
    coverage: bool = True,
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
        describe = lambda c: _room_lines(c, coverage)  # noqa: E731
    else:
        kind = "workspaces" if stage == Stage.WORKSPACE else "standalone objects"
        lines.append(f"Current room: {context.room_label or 'unknown'}")
        lines.append(f"Last searched: {context.last_location_label or 'none'}")
        lines.append("")
        lines.append(f"Candidate {kind}:")
        describe = _location_lines

    for alias, c in zip(aliases, candidates):
        current = stage == Stage.ROOM and c.candidate_id == context.current_room
        lines.append(f"- {alias} (current room)" if current else f"- {alias}")
        lines.extend(f"  {line}" for line in describe(c))

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]
    return messages, aliases


def build_teacher_messages(stage, target, candidates, context):
    """Teacher messages: the Student input without the search coverage line
    (applied by Bayes' rule in athome.training.teacher), teacher system prompt."""
    messages, aliases = build_messages(stage, target, candidates, context, coverage=False)
    messages[0] = {"role": "system", "content": TEACHER_SYSTEM}
    return messages, aliases


def teacher_schema(aliases) -> dict:
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "likelihoods": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate": {"type": "string", "enum": list(aliases)},
                        "likelihood": {"type": "number"},
                    },
                    "required": ["candidate", "likelihood"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["reasoning", "likelihoods"],
        "additionalProperties": False,
    }
