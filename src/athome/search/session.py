"""Search state for one user command (proposal 5-1, 5-3).

Pure logic: decides where to go next and updates the state from visit
outcomes. Motion and perception are handled by the execution layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

from athome.execution.visit import VisitOutcome, VisitStatus
from athome.navigation import LocationCost, NavigationPlanner
from athome.scene_graph.query import STANDALONE, WORKSPACE, SceneGraph
from athome.schemas import ObservedObject, Pose2D
from athome.search.matching import LabelMatcher, TargetMatcher
from athome.search.policy import (
    Candidate,
    MinCostPolicy,
    PlanningContext,
    Policy,
    PolicyError,
    Stage,
)


class TargetStatus(Enum):
    PENDING = "pending"
    SEARCHING = "searching"
    FOUND = "found"
    FAILED = "failed"


class SessionStatus(Enum):
    RUNNING = "running"
    DONE = "done"                  # every target found or failed
    MAX_STEPS = "max_steps"


@dataclass
class TargetRecord:
    name: str
    known: bool                    # present in the scene graph at start
    status: TargetStatus = TargetStatus.PENDING
    found_location: Optional[str] = None
    found_object: Optional[ObservedObject] = None
    # Known locations exhausted, now searching as unknown.
    switched_to_unknown: bool = False
    detail: str = ""


@dataclass(frozen=True)
class SearchDecision:
    step: int
    target: str
    stage: Stage
    room_id: str
    location_id: str
    cost: float
    goals: Tuple[Pose2D, ...]


@dataclass
class StepRecord:
    decision: SearchDecision
    status: VisitStatus
    found: List[str] = field(default_factory=list)
    covered: List[str] = field(default_factory=list)
    policy_fallback: bool = False
    fallback_reason: str = ""


class SearchSession:
    def __init__(
        self,
        graph: SceneGraph,
        navigation: NavigationPlanner,
        targets: Sequence[str],
        policy: Policy = MinCostPolicy(),
        matcher: TargetMatcher = LabelMatcher(),
        max_steps: int = 30,
        # Standalone locations within this distance of a completed
        # observation pose count as observed (0 disables).
        standalone_coverage_radius: float = 1.0,
    ):
        if not targets:
            raise ValueError("Target 없음")
        self.graph = graph
        self.navigation = navigation
        self.policy = policy
        self.matcher = matcher
        self.max_steps = max_steps
        self.coverage_radius = standalone_coverage_radius

        self.targets = [
            TargetRecord(t, known=bool(graph.known_locations(t)))
            for t in targets
        ]
        # Shared across targets, reset per command (a new session).
        self.visited: Set[str] = set()
        self.excluded: Set[str] = set()
        # Standalone locations seen from a nearby observation pose.
        self.covered: Set[str] = set()
        self.history: List[StepRecord] = []
        self.status = SessionStatus.RUNNING
        self._index = 0
        self._room: Optional[str] = None
        self._pending: Optional[SearchDecision] = None
        self._fallback = False
        self._fallback_reason = ""

        for lid, loc in graph.locations.items():
            navigation.add_location(lid, loc.bbox_min, loc.bbox_max)

    @property
    def steps(self) -> int:
        return len(self.history)

    @property
    def current_target(self) -> Optional[TargetRecord]:
        if self._index < len(self.targets):
            return self.targets[self._index]
        return None

    # --- planning -------------------------------------------------------

    def next_decision(self, pose: Pose2D) -> Optional[SearchDecision]:
        """Next location to visit, or None when the command is over."""
        if self._pending is not None:
            raise RuntimeError("이전 결정의 결과 보고 전")
        if self.status != SessionStatus.RUNNING:
            return None

        while self._index < len(self.targets):
            target = self.targets[self._index]
            if target.status in (TargetStatus.FOUND, TargetStatus.FAILED):
                self._advance_target()
                continue
            if self.steps >= self.max_steps:
                self._stop_max_steps()
                return None

            target.status = TargetStatus.SEARCHING
            decision = self._plan(target, pose)
            if decision is not None:
                self._pending = decision
                return decision
            target.status = TargetStatus.FAILED
            target.detail = "탐색 가능한 위치 소진"
            self._advance_target()

        self.status = SessionStatus.DONE
        return None

    def _plan(self, target: TargetRecord, pose: Pose2D):
        if target.known and not target.switched_to_unknown:
            ids = [
                l for l in self.graph.known_locations(target.name)
                if self._open(l)
            ]
            costs = self.navigation.evaluate(pose, ids)
            if costs:
                best = min(costs.values(), key=lambda c: (c.cost, c.location_id))
                return self._decision(target, Stage.KNOWN, best)
            # Not where the graph said: search it like an unknown target.
            target.switched_to_unknown = True

        if self._room is not None:
            decision = self._plan_in_room(target, pose, self._room)
            if decision is not None:
                return decision
            self._room = None

        costs = self.navigation.evaluate(
            pose, [l for l in self.graph.locations if self._open(l)]
        )
        by_room: Dict[str, List[LocationCost]] = {}
        for c in costs.values():
            by_room.setdefault(self.graph.locations[c.location_id].room_id, []).append(c)
        if not by_room:
            return None

        candidates = [
            self._room_candidate(rid, cs) for rid, cs in sorted(by_room.items())
        ]
        self._room = self._select(Stage.ROOM, target.name, candidates)
        return self._plan_in_room(target, pose, self._room, costs)

    def _plan_in_room(self, target, pose, room_id, costs=None):
        for kind, stage in ((WORKSPACE, Stage.WORKSPACE), (STANDALONE, Stage.STANDALONE)):
            ids = [
                l.location_id for l in self.graph.room_locations(room_id, kind)
                if self._open(l.location_id)
            ]
            if costs is None:
                room_costs = self.navigation.evaluate(pose, ids)
            else:
                room_costs = {l: costs[l] for l in ids if l in costs}
            if not room_costs:
                continue
            candidates = [
                self._location_candidate(c) for _, c in sorted(room_costs.items())
            ]
            chosen = self._select(stage, target.name, candidates)
            return self._decision(target, stage, room_costs[chosen])
        return None

    def _select(self, stage, target, candidates) -> str:
        context = self._context(stage)
        try:
            chosen = self.policy.select(stage, target, candidates, context)
        except PolicyError as e:
            chosen, self._fallback_reason = None, str(e)
        else:
            self._fallback_reason = f"후보 밖 출력: {chosen!r}"
        if chosen in {c.candidate_id for c in candidates}:
            return chosen
        # Failed or out-of-candidate output is never applied.
        self._fallback = True
        return MinCostPolicy().select(stage, target, candidates, context)

    def _context(self, stage) -> PlanningContext:
        last = self.history[-1].decision.location_id if self.history else None
        explored = sorted({
            self.graph.locations[l].room_id for l in self.visited
        })
        room = None if stage == Stage.ROOM else self._room
        return PlanningContext(
            room_id=room,
            room_label=self.graph.rooms[room].label if room else None,
            last_location=last,
            last_location_label=self.graph.locations[last].category if last else None,
            explored_rooms=tuple(explored),
            explored_room_labels=tuple(self.graph.rooms[r].label for r in explored),
            visited_count=len(self.visited),
        )

    def _room_candidate(self, room_id, costs: List[LocationCost]) -> Candidate:
        locs = [self.graph.locations[c.location_id] for c in costs]
        return Candidate(
            candidate_id=room_id,
            cost=min(c.cost for c in costs),
            info={
                "room_label": self.graph.rooms[room_id].label,
                "workspaces": [
                    {"category": l.category, "function_label": l.function_label}
                    for l in locs if l.kind == WORKSPACE
                ],
                "standalone_categories": sorted(
                    {l.category for l in locs if l.kind == STANDALONE}),
            },
        )

    def _location_candidate(self, cost: LocationCost) -> Candidate:
        loc = self.graph.locations[cost.location_id]
        info = {"kind": loc.kind, "category": loc.category}
        if loc.kind == WORKSPACE:
            info["function_label"] = loc.function_label
            info["child_categories"] = list(loc.child_categories)
        return Candidate(cost.location_id, cost.cost, info)

    def _decision(self, target, stage, cost: LocationCost) -> SearchDecision:
        return SearchDecision(
            step=self.steps + 1,
            target=target.name,
            stage=stage,
            room_id=self.graph.locations[cost.location_id].room_id,
            location_id=cost.location_id,
            cost=cost.cost,
            goals=cost.goals,
        )

    def _open(self, location_id: str) -> bool:
        return (
            location_id not in self.visited
            and location_id not in self.excluded
            and location_id not in self.covered
        )

    def _advance_target(self) -> None:
        self._index += 1
        # Each target starts from room selection.
        self._room = None

    def _stop_max_steps(self) -> None:
        self.status = SessionStatus.MAX_STEPS
        for t in self.targets:
            if t.status in (TargetStatus.PENDING, TargetStatus.SEARCHING):
                t.status = TargetStatus.FAILED
                t.detail = "최대 탐색 횟수 도달"

    # --- update ---------------------------------------------------------

    def report(self, decision: SearchDecision, outcome: VisitOutcome) -> StepRecord:
        """Apply a visit outcome. Only COMPLETED and FAILED count as a step;
        CANCELED and PAUSED leave the state unchanged for a later retry."""
        if decision is not self._pending:
            raise ValueError("현재 결정과 다른 결과 보고")
        self._pending = None
        record = StepRecord(
            decision, outcome.status, policy_fallback=self._fallback,
            fallback_reason=self._fallback_reason if self._fallback else "")
        self._fallback = False

        if outcome.status == VisitStatus.COMPLETED:
            self.visited.add(decision.location_id)
            record.found = self._match(decision.location_id, outcome)
            record.covered = self._cover(outcome.final_pose or decision.goals[0])
        elif outcome.status == VisitStatus.FAILED:
            # Not Visited: only skipped for the rest of this command.
            self.excluded.add(decision.location_id)
        else:
            return record

        self.history.append(record)
        return record

    def _cover(self, pose: Pose2D) -> List[str]:
        if self.coverage_radius <= 0:
            return []
        out = []
        for lid, loc in self.graph.locations.items():
            if loc.kind != STANDALONE or not self._open(lid):
                continue
            cx = (loc.bbox_min[0] + loc.bbox_max[0]) / 2
            cy = (loc.bbox_min[1] + loc.bbox_max[1]) / 2
            if math.hypot(cx - pose.x, cy - pose.y) <= self.coverage_radius:
                self.covered.add(lid)
                out.append(lid)
        return out

    def _match(self, location_id: str, outcome: VisitOutcome) -> List[str]:
        found = []
        for target in self.targets:
            if target.status == TargetStatus.FOUND:
                continue
            for frame in outcome.observations:
                obj = next(
                    (o for o in frame.objects if self.matcher.matches(target.name, o)),
                    None,
                )
                if obj is not None:
                    target.status = TargetStatus.FOUND
                    target.found_location = location_id
                    target.found_object = obj
                    target.detail = ""
                    found.append(target.name)
                    break
        return found
