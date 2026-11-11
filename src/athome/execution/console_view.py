"""What the operator console shows of a command beyond the status sentence:
elapsed time, the visits so far, where each target was found and its photo,
how the planner is doing and what a pause means. Built from the search
server's events and status (any console's command, not only its own), without
ROS so it can be tested.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable, Optional

from athome.execution.status import STAGE_TEXT

STAGE_NAMES = {stage.value: text for stage, text in STAGE_TEXT.items()}
VISIT_TEXT = {"completed": "관측 완료", "failed": "방문 실패", "canceled": "취소", "paused": "일시정지"}
# VisitReason values and the command's own: what happened, what to do.
REASONS = {
    "navigation_failed": ("이동 실패", "경로의 장애물과 주행 모듈을 확인한 뒤 재개하세요."),
    "rotation_failed": ("제자리 회전 실패", "로봇 주변 공간과 주행 모듈을 확인한 뒤 재개하세요."),
    "motion_rejected": ("주행 모듈이 목표를 거절함", "주행 모듈 상태를 확인한 뒤 재개하세요."),
    "motion_unresponsive": ("주행 모듈 응답 없음", "주행 모듈 실행 여부를 확인한 뒤 재개하세요."),
    "unexpected_cancel": ("주행이 예상 밖으로 취소됨", "다른 곳에서 취소했는지 확인한 뒤 재개하세요."),
    "stop_unconfirmed": ("로봇 정지를 확인하지 못함", "로봇이 멈췄는지 직접 확인하세요."),
    "localization_lost": ("위치 추정 끊김", "측위 상태를 확인하고 필요하면 초기 위치를 다시 찍은 뒤 재개하세요."),
    "map_mismatch": ("지도 버전이 다름", "탐색과 주행이 같은 지도를 쓰는지 확인하세요."),
    "motion_internal_error": ("주행 모듈 내부 오류", "주행 모듈 로그를 확인한 뒤 재개하세요."),
    "perception_lost": ("인지 결과가 끊김", "인지 노드와 GPU 서버 연결을 확인한 뒤 재개하세요."),
    "internal_error": ("탐색 서버 내부 오류", "탐색 서버 로그를 확인하세요."),
    "canceled_by_request": ("정지 요청으로 취소됨", ""),
}
_PHOTO_NAME = re.compile(r"[^/\\]+\.jpg")


def perception_state(age_s: Optional[float], stale_s: float) -> dict:
    """Can a command start? ``age_s``: since the last perception result (None:
    none yet since the console started). The detector loads for 2-3 minutes
    after a launch and a command sent before pauses at its first stop."""
    if age_s is None:
        return {"ready": False, "short": "인지 준비 중",
                "text": "인지 결과를 기다리는 중입니다. 검출 모델을 불러오는 데 2~3분 걸리고, "
                        "첫 결과가 오면 명령을 시작할 수 있습니다."}
    if age_s > stale_s:
        return {"ready": False, "short": f"인지 {round(age_s)}초 끊김",
                "text": f"인지 결과가 {round(age_s)}초째 없습니다. 카메라, 측위, 인지 노드, 검출 서버를 "
                        "확인하세요. 결과가 다시 오면 명령을 시작할 수 있습니다."}
    return {"ready": True, "short": "", "text": ""}


def explain_reason(reason: str) -> dict:
    text, action = REASONS.get(reason, (reason, ""))
    return {"text": text, "action": action}


def found_photo(directory: Path, command_id, target: str, since: float) -> Optional[str]:
    """File name of the newest photo of ``target`` saved by this command
    (found_images.FoundImages). Command numbers start again with every launch
    of the search server, so older files of the same number do not count."""
    if command_id is None or not directory.is_dir():
        return None
    best = None
    for path in directory.glob(f"*_cmd{command_id}_{target.replace(' ', '_')}.jpg"):
        when = path.stat().st_mtime
        if when >= since and (best is None or when > best[0]):
            best = (when, path.name)
    return best[1] if best else None


def photo_path(directory: Path, name: str) -> Optional[Path]:
    """The photo called ``name`` directly in ``directory``, nothing else."""
    if not _PHOTO_NAME.fullmatch(name) or name.startswith("."):
        return None
    path = directory / name
    return path if path.is_file() else None


class SearchTimeline:
    def __init__(self, clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time):
        self._clock = clock
        self._wall = wall
        self._names = {}                # location id -> name, from the status
        self._reset()

    def _reset(self) -> None:
        self._visits = []
        self._found = {}
        self._elapsed = 0.0
        self._running_since = None
        self._ended = None
        self.started_wall = None
        self._planner = {"calls": 0, "fallbacks": 0, "last_fallback": False, "error": ""}

    def _place(self, location_id: str) -> str:
        # Until the status names it: "workspace:room_1:computer desk_3" -> "computer desk".
        return self._names.get(location_id) or location_id.rsplit(":", 1)[-1].rsplit("_", 1)[0]

    def on_status(self, status: dict) -> None:
        if status.get("location") and status.get("location_name"):
            self._names[status["location"]] = status["location_name"]

    def on_decision(self, record: dict) -> None:
        planner = self._planner
        planner["calls"] += 1
        planner["last_fallback"] = bool(record.get("fallback"))
        if record.get("fallback"):
            planner["fallbacks"] += 1
            planner["error"] = record.get("error", "")

    def on_event(self, record: dict) -> None:
        kind = record.get("event")
        if kind == "command_start":
            self._reset()
            self.started_wall = self._wall()
            self._running_since = self._clock()
        elif kind == "command_resume":
            self._ended = None
            if self._running_since is None:
                self._running_since = self._clock()
        elif kind == "visit_start":
            self._visits.append({
                "step": record["step"], "target": record["target"],
                "stage": STAGE_NAMES.get(record["stage"], record["stage"]),
                "location": record["location_id"], "cost_m": record.get("cost_m"),
                "started": record.get("stamp"), "duration_s": None, "status": None, "status_text": "진행 중",
                "reason": "", "found": [], "frames": None, "relooks": 0, "fallback": False})
        elif kind == "visit_end":
            visit = next((v for v in reversed(self._visits) if v["step"] == record["step"]), None)
            if visit is not None:
                reason = record.get("reason", "none")
                stamp, started = record.get("stamp"), visit["started"]
                visit.update(
                    status=record["status"], status_text=VISIT_TEXT.get(record["status"], record["status"]),
                    reason="" if reason == "none" else explain_reason(reason)["text"],
                    found=list(record.get("found", [])), frames=record.get("frames"),
                    relooks=len(record.get("relooks", [])), fallback=bool(record.get("policy_fallback")),
                    duration_s=None if stamp is None or started is None else round(stamp - started, 1))
            for name in record.get("found", []):
                self._found.setdefault(name, {})["location"] = record["location_id"]
        elif kind == "command_end":
            if self._running_since is not None:
                self._elapsed += self._clock() - self._running_since
                self._running_since = None
            self._ended = {"status": record["status"], "reason": record.get("reason", ""),
                           "detail": record.get("detail", "")}
            for t in record.get("targets", []):
                if t["status"] == "found":
                    self._found.setdefault(t["name"], {}).update(
                        location=t.get("found_location") or self._found.get(t["name"], {}).get("location"),
                        label=t.get("found_label"), position=t.get("found_position"))

    def view(self) -> dict:
        elapsed = self._elapsed
        if self._running_since is not None:
            elapsed += self._clock() - self._running_since
        ended = self._ended
        if ended is not None:
            ended = dict(ended, explained=explain_reason(ended["reason"]) if ended["reason"] else None)
        return {
            "started_at": self.started_wall,
            "elapsed_s": None if self.started_wall is None else round(elapsed, 1),
            "visits": [dict(v, place=self._place(v["location"])) for v in reversed(self._visits)],
            "found": {name: dict(f, place=self._place(f.get("location") or ""))
                      for name, f in self._found.items()},
            "ended": ended, "planner": dict(self._planner)}
