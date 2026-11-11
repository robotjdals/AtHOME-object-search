"""Search state for one user command (proposal 5-1, 5-3).

Pure logic: decides where to go next and updates the state from visit
outcomes. Motion and perception are handled by the execution layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from athome.execution.visit import RelookPoint, VisitOutcome, VisitStatus
from athome.navigation import LocationCost, NavigationPlanner
from athome.scene_graph.query import STANDALONE, WORKSPACE, SceneGraph, SearchLocation
from athome.schemas import ObservedObject, Pose2D
from athome.search.matching import LabelMatcher, TargetMatcher, explain
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
    TIME_BUDGET = "time_budget"     # command time budget spent (GenMOS-style stop)


@dataclass
class TargetRecord:
    name: str
    known: bool                    # present in the scene graph at start
    status: TargetStatus = TargetStatus.PENDING
    found_location: Optional[str] = None
    found_object: Optional[ObservedObject] = None
    # Capture time of the frame found_object was seen in (perception stamp).
    found_stamp: Optional[float] = None
    # Known locations exhausted, now searching as unknown.
    switched_to_unknown: bool = False
    detail: str = ""
    # Entered the scene graph during the command (online object layer).
    became_known: bool = False


@dataclass(frozen=True)
class SearchDecision:
    step: int
    target: str
    stage: Stage
    room_id: str
    location_id: str
    cost: float
    goals: Tuple[Pose2D, ...]
    # Where the graph has the target at this location (Known target).
    look_at: Tuple[Tuple[float, float], ...] = ()


@dataclass
class StepRecord:
    decision: SearchDecision
    status: VisitStatus
    found: List[str] = field(default_factory=list)
    # Locations marked searched because their object was observed.
    covered: List[str] = field(default_factory=list)
    policy_fallback: bool = False
    fallback_reason: str = ""


GROUP_PREFIX = "standalone_group:"


def standalone_groups(graph: SceneGraph, room_id: str,
                      costs: Sequence[LocationCost]) -> Dict[str, List[LocationCost]]:
    """Group ID -> reachable open standalone locations of one category in the
    room, nearest first (ties by location ID)."""
    groups: Dict[str, List[LocationCost]] = {}
    for c in costs:
        category = graph.locations[c.location_id].category
        groups.setdefault(f"{GROUP_PREFIX}{room_id}:{category}", []).append(c)
    return {gid: sorted(members, key=lambda c: (c.cost, c.location_id))
            for gid, members in sorted(groups.items())}


def group_candidate(group_id: str, members: Sequence[LocationCost]) -> Candidate:
    """Planner candidate of a standalone group: its category, instance count
    and the cost of the nearest instance (where the robot would go)."""
    category = group_id.rsplit(":", 1)[1]
    return Candidate(group_id, members[0].cost,
                     {"kind": "standalone_group", "category": category, "count": len(members),
                      "nearest_location_id": members[0].location_id})


class SearchSession:
    def __init__(
        self,
        graph: SceneGraph,
        navigation: NavigationPlanner,
        targets: Sequence[str],
        policy: Policy = MinCostPolicy(),
        matcher: TargetMatcher = LabelMatcher(),
        max_steps: int = 30,
        # Resolves an observed static object's map ID to its scene-graph object
        # ID (perception's static object association, proposal 3-1). When set,
        # a Search Location whose defining object was observed during a visit
        # counts as searched (Visited), as observed space is ruled out in object
        # search (belief update of Wandzel et al. ICRA 2019; proposal 5-3).
        observed_object_ids: Optional[Callable[[int], Optional[str]]] = None,
        # Stop when elapsed_s() reaches time_budget_s: real robots stop a
        # search on found / every location visited / time budget (GenMOS:
        # 10 min on Spot). elapsed_s is wall-clock time on the robot and the
        # modelled travel + observation time in the symbolic environment.
        time_budget_s: Optional[float] = None,
        elapsed_s: Optional[Callable[[], float]] = None,
        # False (default): after every visit the next decision starts again
        # from room selection (Stage 1) with the updated Visited state, i.e.
        # re-planning after each observation as object-search systems on real
        # robots do (LFG, Inter-POMDP, POMDP object search; proposal scenario
        # step 6). True keeps a chosen room until its locations are exhausted
        # (earlier behaviour, for reproducing old results).
        commit_to_room: bool = False,
        # Observed fraction of each room (athome.search.coverage), updated at
        # every visit pose; shown at room selection as negative evidence.
        coverage=None,
        # Standalone candidates of a room are grouped by category: the planner
        # chooses a kind of object ("storage box x44") and the robot goes to
        # the nearest open instance, as MoMa-LLM lists a room's objects with
        # counts and navigates by object name. Same-name instances differ only
        # in distance, which the nearest rule already decides; dense HM3D
        # annotation gave up to ~500 per-instance candidates. False lists
        # every instance (prompt 0.5 and earlier).
        group_standalone: bool = True,
        # A target counts as found only when seen in this many observation
        # frames of one visit at the same place (within confirm_radius_m):
        # one false detection must not end the search. 1 (default): the first
        # match, as in the symbolic environment.
        confirmations: int = 1,
        confirm_radius_m: float = 0.5,
        # Look at a visited location again from another place (0: never): goal
        # poses at least this far from where the robot stood. Right after the
        # visit if it saw a likely candidate that did not pass the found check
        # (above the confidence floor: the target is more likely here than at
        # an unseen location); otherwise when no location is left, those with
        # any near miss first. After a plain miss an unseen location is the
        # better next look (about 28% vs 6% at the measured recall, 3 equal
        # locations). The planner is not asked: it never chooses a visited
        # location in training. Detections depend on the viewpoint and do not
        # change on a second look from the same place (412 records,
        # ViewpointDetectionModel).
        recheck_separation_m: float = 0.0,
    ):
        if not targets:
            raise ValueError("Target 없음")
        self.graph = graph
        self.navigation = navigation
        self.policy = policy
        self.matcher = matcher
        self.max_steps = max_steps
        self.observed_object_ids = observed_object_ids
        if (time_budget_s is None) != (elapsed_s is None):
            raise ValueError("time_budget_s와 elapsed_s는 함께 지정해야 합니다.")
        self.time_budget_s = time_budget_s
        self.elapsed_s = elapsed_s
        self.commit_to_room = commit_to_room
        self.coverage = coverage
        self.group_standalone = group_standalone
        if confirmations < 1:
            raise ValueError("confirmations는 1 이상")
        self.confirmations = confirmations
        self.confirm_radius_m = confirm_radius_m
        self.recheck_separation_m = recheck_separation_m

        self.targets = [
            TargetRecord(t, known=bool(graph.known_locations(t)))
            for t in targets
        ]
        # Shared across targets, reset per command (a new session).
        self.visited: Set[str] = set()
        self.history: List[StepRecord] = []
        self.status = SessionStatus.RUNNING
        self._index = 0
        self._room: Optional[str] = None
        self._pending: Optional[SearchDecision] = None
        self._fallback = False
        self._fallback_reason = ""
        # Locations that left the graph during the command (online object
        # layer), for the planning context of visits already made.
        self._retired: Dict[str, SearchLocation] = {}
        # Second look: visited location -> where the robot stood (in visit
        # order), locations with a near miss, and those looked at again.
        self._stood_at: Dict[str, Pose2D] = {}
        self._near_miss: Set[str] = set()
        self._rechecked: Set[str] = set()
        self._recheck_next: Optional[str] = None
        # Results of re-looks in the running visit: (target, x, y) that failed
        # and the object seen in a passed look at a found target.
        self._rejected: List[Tuple[str, float, float]] = []
        self._centred: Dict[str, ObservedObject] = {}

        for lid, loc in graph.locations.items():
            navigation.add_location(lid, loc.bbox_min, loc.bbox_max, loc.kind == STANDALONE, loc.room_id,
                                    graph.other_boxes(lid))

    def update_graph(self, graph: SceneGraph) -> None:
        """Plan on a changed scene graph from the next decision on (online
        object layer: objects added, moved or removed; rooms and Workspaces
        unchanged). Visited and a pending decision stay valid. A target whose
        category newly appears at an open location becomes Known (again)."""
        old = self.graph
        for lid in [*old.locations, *old.object_goals]:
            if lid not in graph.locations and lid not in graph.object_goals:
                self._retired[lid] = old.goal(lid)
        before = {t.name: set(old.known_locations(t.name)) for t in self.targets}
        self.graph = graph
        for lid, loc in [*graph.locations.items(), *graph.object_goals.items()]:
            prev = old.locations.get(lid) or old.object_goals.get(lid)
            if prev is not None and (prev.bbox_min, prev.bbox_max) != (loc.bbox_min, loc.bbox_max):
                self.navigation.remove_location(lid)
            if lid in graph.locations:
                self.navigation.add_location(lid, loc.bbox_min, loc.bbox_max, loc.kind == STANDALONE, loc.room_id,
                                             graph.other_boxes(lid))
        for t in self.targets:
            if t.status in (TargetStatus.FOUND, TargetStatus.FAILED):
                continue
            new = [l for l in graph.known_locations(t.name)
                   if l not in before[t.name] and self._open(l)]
            if new:
                t.became_known = not t.known
                t.switched_to_unknown = False

    def _goal(self, location_id: str) -> SearchLocation:
        """Location of the current graph, or one that left it."""
        if location_id in self.graph.locations or location_id in self.graph.object_goals:
            return self.graph.goal(location_id)
        return self._retired[location_id]

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
            if self.time_budget_s is not None and self.elapsed_s() >= self.time_budget_s:
                self._stop(SessionStatus.TIME_BUDGET, "시간 예산 초과")
                return None

            target.status = TargetStatus.SEARCHING
            decision = None
            if self._recheck_next is not None:
                location_id, self._recheck_next = self._recheck_next, None
                decision = self._recheck(target, pose, location_id)
            if decision is None:
                decision = self._plan(target, pose)
            if decision is None and self.recheck_separation_m > 0:
                decision = self._plan_recheck(target, pose)
            if decision is not None:
                self._pending = decision
                return decision
            target.status = TargetStatus.FAILED
            target.detail = "탐색 가능한 위치 소진"
            self._advance_target()

        self.status = SessionStatus.DONE
        return None

    def _plan(self, target: TargetRecord, pose: Pose2D):
        if (target.known or target.became_known) and not target.switched_to_unknown:
            ids = [
                l for l in self.graph.known_locations(target.name)
                if self._open(l)
            ]
            for lid in ids:  # object goals are added when a Known target needs them
                goal = self.graph.goal(lid)
                self.navigation.add_location(lid, goal.bbox_min, goal.bbox_max, goal.kind == STANDALONE, goal.room_id,
                                             self.graph.other_boxes(lid))
            costs = self.navigation.evaluate(pose, ids)
            if costs:
                best = min(costs.values(), key=lambda c: (c.cost, c.location_id))
                return self._decision(target, Stage.KNOWN, best)
            # Not where the graph said: search it like an unknown target.
            target.switched_to_unknown = True

        if self.commit_to_room and self._room is not None:
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

    def _plan_recheck(self, target: TargetRecord, pose: Pose2D):
        """The next visited location that has a goal pose away from where the
        robot stood."""
        order = sorted((l for l in self._stood_at if l not in self._rechecked),
                       key=lambda l: l not in self._near_miss)      # stable: visit order
        for lid in order:
            decision = self._recheck(target, pose, lid)
            if decision is not None:
                return decision
        return None

    def _recheck(self, target: TargetRecord, pose: Pose2D, location_id: str):
        """A look at ``location_id`` from elsewhere; a location with no such
        goal pose (against a wall) is done."""
        stood = self._stood_at[location_id]
        far = lambda p: math.hypot(p.x - stood.x, p.y - stood.y) >= self.recheck_separation_m  # noqa: E731
        cost = self.navigation.evaluate(pose, [location_id], keep=far).get(location_id)
        if cost is None:
            self._rechecked.add(location_id)
            return None
        return self._decision(target, Stage.RECHECK, cost)

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
            if stage == Stage.STANDALONE and self.group_standalone:
                groups = standalone_groups(self.graph, room_id, room_costs.values())
                candidates = [group_candidate(gid, members) for gid, members in groups.items()]
                chosen = self._select(stage, target.name, candidates)
                return self._decision(target, stage, groups[chosen][0])
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
            self._goal(l).room_id for l in self.visited
        })
        room = None if stage == Stage.ROOM else self._room
        return PlanningContext(
            room_id=room,
            room_label=self.graph.rooms[room].label if room else None,
            last_location=last,
            last_location_label=self._goal(last).category if last else None,
            explored_rooms=tuple(explored),
            explored_room_labels=tuple(self.graph.rooms[r].label for r in explored),
            visited_count=len(self.visited),
            current_room=self.history[-1].decision.room_id if self.history else None,
        )

    def _room_candidate(self, room_id, costs: List[LocationCost]) -> Candidate:
        locs = [self.graph.locations[c.location_id] for c in costs]
        # Search coverage of the room: visited locations out of visited + the
        # still open reachable ones (negative evidence for the room).
        searched = sum(1 for l in self.visited
                       if l in self.graph.locations and self.graph.locations[l].room_id == room_id)
        return Candidate(
            candidate_id=room_id,
            cost=min(c.cost for c in costs),
            info={
                "room_label": self.graph.rooms[room_id].label,
                "workspaces": [
                    {"category": l.category, "function_label": l.function_label}
                    for l in locs if l.kind == WORKSPACE
                ],
                # One entry per location (repeats kept, shown as counts).
                "standalone_categories": sorted(
                    l.category for l in locs if l.kind == STANDALONE),
                "searched_locations": searched,
                "total_locations": searched + len(locs),
                **({"observed_fraction": round(self.coverage.fraction(room_id), 2)}
                   if self.coverage is not None else {}),
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
            room_id=self._goal(cost.location_id).room_id,
            location_id=cost.location_id,
            cost=cost.cost,
            goals=cost.goals,
            look_at=(self.graph.object_points(target.name, cost.location_id)
                     if stage == Stage.KNOWN else ()),
        )

    def _open(self, location_id: str) -> bool:
        return location_id not in self.visited

    def _advance_target(self) -> None:
        self._index += 1
        # Each target starts from room selection.
        self._room = None

    def _stop_max_steps(self) -> None:
        self._stop(SessionStatus.MAX_STEPS, "최대 탐색 횟수 도달")

    def _stop(self, status: SessionStatus, detail: str) -> None:
        self.status = status
        for t in self.targets:
            if t.status in (TargetStatus.PENDING, TargetStatus.SEARCHING):
                t.status = TargetStatus.FAILED
                t.detail = detail

    # --- update ---------------------------------------------------------

    def report(self, decision: SearchDecision, outcome: VisitOutcome) -> StepRecord:
        """Apply valid observations regardless of visit completion.

        Only COMPLETED marks Visited. COMPLETED/FAILED count as steps;
        PAUSED/CANCELED keep the location open and do not consume a step.
        """
        if decision is not self._pending:
            raise ValueError("현재 결정과 다른 결과 보고")
        self._pending = None
        record = StepRecord(
            decision, outcome.status, policy_fallback=self._fallback,
            fallback_reason=self._fallback_reason if self._fallback else "")
        self._fallback = False

        if self.recheck_separation_m > 0:
            if self._rejected or any(p.kind == "near_miss" for p in self.relook_points(outcome.observations)):
                self._near_miss.add(decision.location_id)
            if outcome.status == VisitStatus.COMPLETED:
                if decision.stage == Stage.RECHECK:
                    self._rechecked.add(decision.location_id)
                elif decision.location_id in self.graph.locations:
                    self._stood_at[decision.location_id] = outcome.final_pose or decision.goals[0]
        record.found = self._match(decision.location_id, outcome)
        if (self.recheck_separation_m > 0 and decision.location_id in self._stood_at
                and decision.location_id not in self._rechecked
                and self._likely_candidate(outcome.observations)):
            self._recheck_next = decision.location_id
        self._rejected.clear()
        self._centred.clear()
        record.covered = self._cover(outcome)
        if self.coverage is not None and outcome.observations:
            pose = outcome.final_pose or decision.goals[0]
            self.coverage.observe(pose.x, pose.y)
        if outcome.status == VisitStatus.COMPLETED:
            self.visited.add(decision.location_id)
        elif outcome.status != VisitStatus.FAILED:
            return record

        self.history.append(record)
        return record

    def _cover(self, outcome: VisitOutcome) -> List[str]:
        if self.observed_object_ids is None:
            return []
        covered = []
        for frame in outcome.observations:
            for obj in frame.objects:
                if not obj.is_static:
                    continue
                object_id = self.observed_object_ids(obj.object_id)
                for lid in self.graph.locations_defined_by(object_id):
                    if self._open(lid) and lid not in covered:
                        covered.append(lid)
        self.visited.update(covered)
        return covered

    def _is_rejected(self, target: str, obj) -> bool:
        r2 = self.confirm_radius_m ** 2
        return any(t == target and (obj.centroid[0] - x) ** 2 + (obj.centroid[1] - y) ** 2 <= r2
                   for t, x, y in self._rejected)

    def is_candidate(self, obj) -> bool:
        """A static observation that matches a target not found yet."""
        return obj.is_static and any(
            self.matcher.matches(t.name, obj) and not self._is_rejected(t.name, obj)
            for t in self.targets if t.status != TargetStatus.FOUND)

    def relook_points(self, frames, passing=()) -> List[RelookPoint]:
        """What is worth centring in the view for the open targets, in order:
        confirmed targets, then detections named as a target that do not pass
        the found check (too low a confidence or CLIP rank) or were seen in
        too few frames, the most confident first. ``passing`` (frames taken
        while turning) only adds near misses."""
        points, near = [], []
        for target in self.targets:
            if target.status == TargetStatus.FOUND:
                continue
            found = self._confirmed(target.name, frames)
            if found is not None:
                points.append(RelookPoint(found.centroid[0], found.centroid[1], target.name, "found"))
                continue
            for frame in (*frames, *passing):
                for o in frame.objects:
                    if (o.is_static and explain(self.matcher, target.name, o) not in ("label", "outside")
                            and not self._is_rejected(target.name, o)):
                        near.append((o.confidence, RelookPoint(o.centroid[0], o.centroid[1],
                                                               target.name, "near_miss")))
        return points + [p for _, p in sorted(near, key=lambda n: -n[0])]

    def _likely_candidate(self, frames) -> bool:
        """An open target taken back by a re-look, or named above the
        confidence floor without passing the found check."""
        for target in self.targets:
            if target.status == TargetStatus.FOUND:
                continue
            if any(t == target.name for t, _, _ in self._rejected):
                return True
            if any(o.is_static and explain(self.matcher, target.name, o) not in ("label", "outside", "confidence")
                   for f in frames for o in f.objects):
                return True
        return False

    def relook_result(self, point: RelookPoint, frames) -> bool:
        """``frames``: the look at ``point``. The target there is confirmed in
        them alone, or a "found" point is taken back for this visit."""
        r2 = self.confirm_radius_m ** 2
        near = [o for f in frames for o in f.objects
                if o.is_static and self.matcher.matches(point.target, o)
                and (o.centroid[0] - point.x) ** 2 + (o.centroid[1] - point.y) ** 2 <= r2]
        hits = sum(any(o in near for o in f.objects) for f in frames)
        passed = hits >= self.confirmations
        if passed and point.kind == "found":
            self._centred[point.target] = max(near, key=lambda o: o.confidence)
        elif not passed and point.kind == "found":
            self._rejected.append((point.target, point.x, point.y))
        return passed

    def all_found_in(self, frames) -> bool:
        """Every target not found yet is confirmed in ``frames`` (a visit's
        observations so far): the command ends with this visit."""
        open_targets = [t for t in self.targets if t.status != TargetStatus.FOUND]
        return bool(open_targets) and all(
            self._confirmed(t.name, frames) is not None for t in open_targets)

    def _match(self, location_id: str, outcome: VisitOutcome) -> List[str]:
        found = []
        for target in self.targets:
            if target.status == TargetStatus.FOUND:
                continue
            # The view of a passed re-look shows the object whole.
            obj = self._centred.get(target.name) or self._confirmed(target.name, outcome.observations)
            if obj is not None:
                target.status = TargetStatus.FOUND
                target.found_location = location_id
                target.found_object = obj
                target.found_stamp = next((f.stamp for f in outcome.observations
                                           if any(o is obj for o in f.objects)), None)
                target.detail = ""
                found.append(target.name)
        return found

    def _confirmed(self, target: str, frames):
        """The matched observation of ``target`` seen in ``confirmations``
        frames at one place (the most confident of them), else None."""
        per_frame = [[o for o in f.objects if o.is_static and self.matcher.matches(target, o)
                      and not self._is_rejected(target, o)]
                     for f in frames]
        if self.confirmations == 1:
            return next((m[0] for m in per_frame if m), None)
        r2 = self.confirm_radius_m ** 2
        near = lambda a, b: sum((x - y) ** 2 for x, y in zip(a.centroid[:2], b.centroid[:2])) <= r2  # noqa: E731
        best = None
        for candidate in (o for m in per_frame for o in m):
            support = [next(o for o in m if near(o, candidate)) for m in per_frame
                       if any(near(o, candidate) for o in m)]
            if len(support) >= self.confirmations:
                top = max(support, key=lambda o: o.confidence)
                if best is None or top.confidence > best.confidence:
                    best = top
        return best
