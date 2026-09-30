"""Events of a search run as JSON records, for the run report.

The search server publishes them (/athome/search/events), so a rosbag of a
run holds the whole command: its start, every visit (decision, motion
outcome, what perception saw, matches) and the result, which a bag cannot
hold otherwise (an action result is a service reply). athome.run_report
reads them back. Planner queries are recorded separately
(athome.search.decision_log).

Recording must never disturb the search: a failing sink or record is
counted and reported through ``on_error``, never raised.
"""

from __future__ import annotations

from typing import Callable, List, Optional

SCHEMA_VERSION = 1
# Detections kept per visit (highest confidence first).
MAX_SEEN = 30


def _pose(p) -> Optional[List[float]]:
    return None if p is None else [round(p.x, 3), round(p.y, 3), round(p.yaw, 4)]


def _seen(frames) -> List[dict]:
    """Distinct (label, static) detections of a visit with their best confidence."""
    best = {}
    for frame in frames:
        for o in frame.objects:
            key = (o.label, bool(o.is_static))
            best[key] = max(best.get(key, 0.0), float(o.confidence))
    ordered = sorted(best.items(), key=lambda kv: -kv[1])[:MAX_SEEN]
    return [{"label": label, "static": static, "confidence": round(c, 3)}
            for (label, static), c in ordered]


class RunLog:
    """Observer of athome.execution.command.CommandExecutor."""

    def __init__(self, sink: Callable[[dict], None],
                 on_error: Optional[Callable[[Exception], None]] = None,
                 setup: Optional[dict] = None):
        self._sink = sink
        self._on_error = on_error
        # Fixed facts of the robot setup (map version, graph, planner),
        # recorded with every command start.
        self._setup = dict(setup or {})
        self.run = 0
        self.errors = 0

    def _emit(self, build: Callable[[], dict]) -> None:
        try:
            self._sink({"schema": SCHEMA_VERSION, "run": self.run, **build()})
        except Exception as e:  # noqa: BLE001 - logging must not stop the search
            self.errors += 1
            if self._on_error is not None:
                self._on_error(e)

    def command_started(self, session) -> None:
        self.run += 1
        self._emit(lambda: {
            "event": "command_start",
            "targets": [{"name": t.name, "known": t.known} for t in session.targets],
            "max_steps": session.max_steps,
            "time_budget_s": session.time_budget_s,
            "setup": self._setup,
        })

    def command_resumed(self) -> None:
        self._emit(lambda: {"event": "command_resume"})

    def visit_started(self, decision) -> None:
        self._emit(lambda: {
            "event": "visit_start",
            "step": decision.step,
            "target": decision.target,
            "stage": decision.stage.value,
            "room_id": decision.room_id,
            "location_id": decision.location_id,
            "cost_m": round(float(decision.cost), 3),
            "goals": [_pose(g) for g in decision.goals],
        })

    def visit_finished(self, decision, record, outcome) -> None:
        self._emit(lambda: {
            "event": "visit_end",
            "step": decision.step,
            "location_id": decision.location_id,
            "status": outcome.status.value,
            "reason": outcome.reason.value,
            "detail": outcome.detail,
            "nav_attempts": outcome.nav_attempts,
            "goal": _pose(outcome.goal),
            "final_pose": _pose(outcome.final_pose),
            "frames": len(outcome.observations),
            "seen": _seen(outcome.observations),
            "found": list(record.found),
            "policy_fallback": record.policy_fallback,
            "fallback_reason": record.fallback_reason,
        })

    def command_finished(self, result) -> None:
        self._emit(lambda: {
            "event": "command_end",
            "status": result.status.value,
            "reason": result.reason,
            "detail": result.detail,
            "location_id": result.location_id,
            "steps": len(result.history),
            "targets": [{
                "name": t.name,
                "known": t.known,
                "status": t.status.value,
                "found_location": t.found_location,
                "found_label": t.found_object.label if t.found_object is not None else None,
                "found_position": (None if t.found_object is None
                                   else [round(float(v), 3) for v in t.found_object.centroid]),
                "detail": t.detail,
            } for t in result.targets],
        })
