"""Run one user command: plan with SearchSession, execute with VisitExecutor."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol

from athome.execution.visit import (
    VisitExecutor,
    VisitOutcome,
    VisitReason,
    VisitRequest,
    VisitStatus,
)
from athome.navigation import StartNotFree
from athome.schemas import Pose2D
from athome.search.session import (
    SearchDecision,
    SearchSession,
    SessionStatus,
    StepRecord,
    TargetRecord,
)


class PoseSource(Protocol):
    def current_pose(self) -> Optional[Pose2D]: ...


class CommandStatus(Enum):
    COMPLETED = "completed"      # every target found or failed
    MAX_STEPS = "max_steps"
    CANCELED = "canceled"
    PAUSED = "paused"


class CommandPhase(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    VISITING = "visiting"
    ABORTING = "aborting"      # internal error: stopping the motion first
    DONE = "done"


@dataclass
class CommandResult:
    status: CommandStatus
    reason: str = ""
    detail: str = ""
    targets: List[TargetRecord] = field(default_factory=list)
    history: List[StepRecord] = field(default_factory=list)


class CommandExecutor:
    def __init__(
        self,
        visit: VisitExecutor,
        pose_source: PoseSource,
        map_version: str,
        on_step=None,
        # Pose unavailable / not in free space for this long -> PAUSED.
        pose_timeout: float = 5.0,
    ):
        self._visit = visit
        self._clock = visit.clock
        self._pose_timeout = pose_timeout
        self._waiting_since: Optional[float] = None
        self.error: Optional[BaseException] = None
        self._pose = pose_source
        self._map_version = map_version
        # Called with (decision, StepRecord) after each finished step.
        self._on_step = on_step
        self.phase = CommandPhase.IDLE
        self.result: Optional[CommandResult] = None
        self.decision: Optional[SearchDecision] = None
        self.session: Optional[SearchSession] = None

    def start(self, session: SearchSession) -> None:
        if self.phase in (CommandPhase.PLANNING, CommandPhase.VISITING):
            raise RuntimeError("이미 실행 중인 명령이 있음")
        self.session = session
        self.result = None
        self.decision = None
        self.error = None
        self._waiting_since = None
        self.phase = CommandPhase.PLANNING

    def resume(self) -> None:
        """Continue a PAUSED command with its search state (Visited etc.)."""
        if (
            self.phase != CommandPhase.DONE
            or self.result is None
            or self.result.status != CommandStatus.PAUSED
            or self.session.status != SessionStatus.RUNNING
        ):
            raise RuntimeError("재개할 수 있는 일시중지 명령 없음")
        self.result = None
        self.error = None
        self._waiting_since = None
        self.phase = CommandPhase.PLANNING

    def cancel(self) -> None:
        if self.phase == CommandPhase.VISITING:
            # Finishes as CANCELED once the stop is confirmed.
            self._visit.cancel()
        elif self.phase == CommandPhase.PLANNING:
            self._finish(CommandStatus.CANCELED, "canceled_by_request")

    def step(self) -> Optional[CommandResult]:
        try:
            if self.phase == CommandPhase.PLANNING:
                self._plan()
            elif self.phase == CommandPhase.VISITING:
                self._track_visit()
            elif self.phase == CommandPhase.ABORTING:
                self._abort_step()
        except Exception as e:  # noqa: BLE001 - never leave the robot moving
            self._on_internal_error(e)
        return self.result

    def _on_internal_error(self, e: BaseException) -> None:
        self.error = e
        if self.phase == CommandPhase.ABORTING or not self._visit.busy:
            self._finish(CommandStatus.PAUSED, "internal_error", _describe(e))
            return
        # Stop the robot before reporting.
        self._visit.cancel()
        self.phase = CommandPhase.ABORTING

    def _abort_step(self) -> None:
        outcome = self._visit.step()
        if outcome is not None:
            detail = _describe(self.error)
            if outcome.status != VisitStatus.CANCELED:
                detail += f" / 정지 처리: {outcome.reason.value}"
            self._finish(CommandStatus.PAUSED, "internal_error", detail)

    def _wait_or_pause(self, reason: str, detail: str = "") -> None:
        now = self._clock()
        if self._waiting_since is None:
            self._waiting_since = now
        elif now - self._waiting_since > self._pose_timeout:
            self._finish(CommandStatus.PAUSED, reason, detail)

    def _plan(self) -> None:
        pose = self._pose.current_pose()
        if pose is None:
            self._wait_or_pause("localization_unavailable")
            return
        try:
            decision = self.session.next_decision(pose)
        except StartNotFree as e:
            self._wait_or_pause("start_not_free", str(e))
            return
        self._waiting_since = None

        if decision is None:
            status = (
                CommandStatus.MAX_STEPS
                if self.session.status == SessionStatus.MAX_STEPS
                else CommandStatus.COMPLETED
            )
            self._finish(status)
            return

        try:
            self._visit.start(VisitRequest(
                visit_id=f"step{decision.step}:{decision.location_id}",
                goals=decision.goals,
                map_version=self._map_version,
            ))
        except RuntimeError as e:
            # Previous motion not confirmed stopped.
            self.session.report(decision, _paused_outcome(decision))
            self._finish(CommandStatus.PAUSED, "stop_unconfirmed", str(e))
            return
        self.decision = decision
        self.phase = CommandPhase.VISITING

    def _track_visit(self) -> None:
        outcome = self._visit.step()
        if outcome is None:
            return
        decision, self.decision = self.decision, None
        record = self.session.report(decision, outcome)
        if self._on_step is not None:
            self._on_step(decision, record, outcome)

        if outcome.status in (VisitStatus.COMPLETED, VisitStatus.FAILED):
            self.phase = CommandPhase.PLANNING
        elif outcome.status == VisitStatus.CANCELED:
            self._finish(CommandStatus.CANCELED, outcome.reason.value, outcome.detail)
        else:
            self._finish(CommandStatus.PAUSED, outcome.reason.value, outcome.detail)

    def _finish(self, status: CommandStatus, reason="", detail="") -> None:
        self.result = CommandResult(
            status=status,
            reason=reason,
            detail=detail,
            targets=list(self.session.targets),
            history=list(self.session.history),
        )
        self.phase = CommandPhase.DONE


def _describe(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _paused_outcome(decision: SearchDecision) -> VisitOutcome:
    return VisitOutcome(
        visit_id=decision.location_id,
        status=VisitStatus.PAUSED,
        reason=VisitReason.STOP_UNCONFIRMED,
    )
