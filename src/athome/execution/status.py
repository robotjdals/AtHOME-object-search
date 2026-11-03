"""What the search is doing right now, as one record (state + Korean
sentence) for status views and logs: waiting for the operator, parsing,
planner inference, driving, observing heading k of n, confirming a possible
target, and the result. Read-only: built from the executors' state each
time it is shown.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

from athome.execution.command import CommandPhase, CommandStatus
from athome.execution.visit import Phase
from athome.search import TargetStatus
from athome.search.policy import Stage

STAGE_TEXT = {Stage.KNOWN: "아는 위치", Stage.ROOM: "방 선택", Stage.WORKSPACE: "워크스페이스 선택",
              Stage.STANDALONE: "단독 위치 선택"}
RESULT_TEXT = {CommandStatus.COMPLETED: "완료", CommandStatus.MAX_STEPS: "최대 단계 도달",
               CommandStatus.TIME_BUDGET: "시간 예산 소진", CommandStatus.CANCELED: "취소됨",
               CommandStatus.PAUSED: "일시정지"}
TARGET_TEXT = {TargetStatus.PENDING: "대기", TargetStatus.SEARCHING: "탐색 중",
               TargetStatus.FOUND: "발견", TargetStatus.FAILED: "못 찾음"}


def location_name(graph, location_id: str) -> str:
    """'computer desk (kitchen)' for a Search Location of ``graph``."""
    loc = graph.locations.get(location_id) if graph is not None else None
    if loc is None:
        return location_id
    room = graph.rooms.get(loc.room_id)
    return f"{loc.category} ({room.label if room else loc.room_id})"


def describe(command, visit_progress: dict, graph=None, query=None,
             clock: Callable[[], float] = None, pose=None,
             parsing: Optional[tuple] = None, awaiting: Optional[Sequence[str]] = None) -> dict:
    """``command``: CommandExecutor; ``visit_progress``: VisitExecutor.progress();
    ``query``: DecisionLog.active (stage, target, start on ``clock``);
    ``parsing``: (instruction, start on ``clock``) while the command parser
    runs; ``awaiting``: parsed targets the operator has not confirmed yet."""
    session = command.session
    status = {"state": "idle", "text": "대기 중", "targets": [], "location": None,
              "location_name": None, "stage": None, "heading": None, "last": None}
    if session is not None:
        status["targets"] = [{"name": t.name, "status": t.status.value,
                              "text": TARGET_TEXT.get(t.status, t.status.value)} for t in session.targets]
        if session.history:
            r = session.history[-1]
            found = f", 발견 {', '.join(r.found)}" if r.found else ", 목표 없음"
            status["last"] = (f"step {r.decision.step} {location_name(graph, r.decision.location_id)}: "
                              f"{r.status.value}{found}")

    phase = command.phase
    if phase == CommandPhase.IDLE or session is None:
        if parsing is not None:
            status.update(state="parsing", text=f"명령 해석 중: '{parsing[0]}' ({clock() - parsing[1]:.1f}초)")
        elif awaiting:
            status.update(state="awaiting_confirmation", text=f"사용자 확인 대기: {', '.join(awaiting)}")
        return status

    if phase == CommandPhase.DONE:
        result = command.result
        found = [t.name for t in result.targets if t.status == TargetStatus.FOUND]
        missed = [t.name for t in result.targets if t.status != TargetStatus.FOUND]
        text = RESULT_TEXT.get(result.status, result.status.value)
        if found:
            text += f": 발견 {', '.join(found)}"
        if missed:
            text += f"{',' if found else ':'} 못 찾음 {', '.join(missed)}"
        if result.reason:
            text += f" ({result.reason}{': ' + result.detail if result.detail else ''})"
        status.update(state="done", result=result.status.value, text=text)
        return status

    if phase == CommandPhase.PLANNING:
        if query is not None:
            stage, target, start = query
            status.update(state="inferring", stage=stage.value,
                          text=f"추론 중: {STAGE_TEXT.get(stage, stage.value)} ({target}) {clock() - start:.1f}초")
        else:
            status.update(state="planning", text="다음 탐색 위치 계산 중")
        return status

    if phase == CommandPhase.ABORTING:
        status.update(state="stopping", text="중단: 정지 확인 대기")
        return status

    decision = command.decision
    if decision is not None:
        name = location_name(graph, decision.location_id)
        status.update(location=decision.location_id, location_name=name, stage=decision.stage.value,
                      step=decision.step, target=decision.target)
    else:
        name = "?"
    p = visit_progress
    vphase = p.get("phase")
    heading = (p.get("heading", 0) + 1, p.get("heading_count", 0))
    if vphase == Phase.NAVIGATING.value:
        goal, extra = p.get("goal"), ""
        if goal is not None and pose is not None:
            extra = f", 남은 직선 {math.hypot(goal.x - pose.x, goal.y - pose.y):.1f} m"
        if p.get("goal_index", 0) > 0 or p.get("nav_attempts", 1) > 1:
            extra += f", 목표 자세 {p['goal_index'] + 1}/{p['goal_count']} 시도 {p['nav_attempts']}"
        stage = STAGE_TEXT.get(decision.stage, "") if decision is not None else ""
        status.update(state="navigating", text=f"이동 중 → {name} [{stage}]{extra}")
    elif vphase == Phase.ROTATING.value:
        status.update(state="rotating", heading=heading,
                      text=f"관측 회전 {heading[0]}/{heading[1]}: {name}")
    elif vphase == Phase.OBSERVING.value:
        hits, need = p.get("candidate_frames", 0), p.get("confirm_frames", 1)
        if 0 < hits < need:
            status.update(state="confirming", heading=heading,
                          text=f"관측 {heading[0]}/{heading[1]}: 목표 후보 확인 중 ({hits}/{need}프레임)")
        else:
            status.update(state="observing", heading=heading,
                          text=f"관측 {heading[0]}/{heading[1]}: 인지 결과 대기 "
                               f"{p.get('observing_s', 0.0):.1f}초 (프레임 {p.get('frames', 0)})")
    elif vphase == Phase.WAITING.value:
        status.update(state="retry_wait", text=f"주행 재시도 대기: {name}")
    elif vphase == Phase.CANCELING.value:
        status.update(state="stopping", text="정지 중")
    elif decision is None:          # the visit just ended, its result being applied
        status.update(state="planning", text="방문 결과 처리 중")
    else:
        status.update(state="visiting", text=f"방문 중: {name}")
    return status
