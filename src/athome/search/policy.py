"""Selection policies. The LLM planner implements the same interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, Sequence, Tuple


class Stage(Enum):
    KNOWN = "known"
    ROOM = "room"
    WORKSPACE = "workspace"
    STANDALONE = "standalone"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    cost: float                 # minimum A* path cost [m]
    # Semantic context for the planner (labels, child tags, ...).
    info: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class PlanningContext:
    # Room being searched (location stages) or None (room stage).
    room_id: Optional[str] = None
    room_label: Optional[str] = None
    # Last visited location, standing in for "current location".
    last_location: Optional[str] = None
    last_location_label: Optional[str] = None
    # Room labels/IDs with at least one visited location in this command.
    explored_rooms: Tuple[str, ...] = ()
    explored_room_labels: Tuple[str, ...] = ()
    visited_count: int = 0


class PolicyError(RuntimeError):
    """No usable selection (timeout, connection, unparsable output)."""


class Policy(Protocol):
    def select(
        self,
        stage: Stage,
        target: str,
        candidates: Sequence[Candidate],
        context: PlanningContext,
    ) -> str:
        """Return one ``candidate_id`` from ``candidates``."""
        ...


class MinCostPolicy:
    """Baseline: nearest candidate by path cost (ties broken by ID)."""

    def select(self, stage, target, candidates, context=None):
        return min(candidates, key=lambda c: (c.cost, c.candidate_id)).candidate_id
