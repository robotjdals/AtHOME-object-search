"""Selection policies. The LLM planner implements the same interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol, Sequence, Tuple


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
    # Room of the last visited location (where the robot is), None before the
    # first visit. Shown at the room stage so staying is an explicit option.
    current_room: Optional[str] = None


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


class RandomPolicy:
    """Uniformly random candidate (seeded): the uninformed baseline of
    object-search evaluations, next to the nearest-first MinCostPolicy."""

    def __init__(self, seed: int = 0):
        import random
        self._rng = random.Random(seed)

    def select(self, stage: Stage, target: str, candidates: Sequence[Candidate],
               context: PlanningContext) -> str:
        return self._rng.choice(sorted(c.candidate_id for c in candidates))


class ShuffledPolicy:
    """Presents the candidates to a wrapped policy in a seeded random order
    per query. LLM selectors depend on the option order (Zheng et al., ICLR
    2024; Pezeshkpour & Hruschka 2023); SearchSession lists candidates in a
    fixed ID order, so GRPO rollouts and evaluation shuffle here and the
    search logic stays deterministic. ``last_order`` is the presented order;
    other attributes (last_messages, last_output) come from the wrapped policy."""

    def __init__(self, policy, seed):
        import random
        self.policy = policy
        self._rng = random.Random(seed)
        self.last_order: Optional[List[str]] = None

    def select(self, stage: Stage, target: str, candidates: Sequence[Candidate],
               context: PlanningContext) -> str:
        order = list(candidates)
        self._rng.shuffle(order)
        self.last_order = [c.candidate_id for c in order]
        return self.policy.select(stage, target, order, context)

    def __getattr__(self, name):
        return getattr(self.policy, name)
