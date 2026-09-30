"""Report of recorded search runs: an image, a table and metrics per command.

Inputs are plain records, read from a rosbag by athome_ros.run_report:
events of athome.execution.run_log (command start/resume/end, visits),
planner queries of athome.search.decision_log and the robot trajectory
(time, x, y, yaw in the map frame). Every record carries the search
server's clock ("stamp"). Metrics follow the symbolic evaluation
(scripts/evaluate_student.py): distance, visits (steps) and
cost = distance + step_cost x visits.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from athome.scene_graph.query import SceneGraph

Sample = Tuple[float, float, float, float]          # t, x, y, yaw
STAGE_NAMES = {"room": "방", "workspace": "워크스페이스", "standalone": "단독 물체"}
STATUS_NAMES = {"completed": "완료", "failed": "실패", "paused": "일시정지",
                "canceled": "취소"}
VISIT_COLORS = {"completed": "tab:green", "failed": "red", "paused": "darkorange",
                "canceled": "gray"}
SEEN_IN_TABLE = 5


@dataclass
class Visit:
    step: int
    target: str
    stage: str
    room_id: str
    location_id: str
    cost_m: float
    goals: list
    start_s: float
    decisions: List[dict] = field(default_factory=list)
    end: Optional[dict] = None           # visit_end event
    distance_m: float = 0.0

    @property
    def status(self) -> str:
        return self.end["status"] if self.end else "unfinished"

    @property
    def duration_s(self) -> Optional[float]:
        return self.end["stamp"] - self.start_s if self.end else None

    @property
    def pose(self) -> Optional[list]:
        """Where the robot observed from (reached goal), else the first goal."""
        if self.end is not None:
            for key in ("final_pose", "goal"):
                if self.end.get(key):
                    return self.end[key]
        return self.goals[0] if self.goals else None


@dataclass
class Run:
    run: int
    start_s: float
    targets: List[dict]                  # command_start targets
    partial: bool = False                # recording started after the command
    setup: dict = field(default_factory=dict)          # map version, graph, planner
    visits: List[Visit] = field(default_factory=list)
    ends: List[dict] = field(default_factory=list)      # command_end: pauses and the last
    resumes: List[float] = field(default_factory=list)
    decisions: List[dict] = field(default_factory=list)
    samples: List[Sample] = field(default_factory=list)
    last_s: float = 0.0

    @property
    def final(self) -> Optional[dict]:
        return self.ends[-1] if self.ends else None


def path_length(samples: Sequence[Sample]) -> float:
    return sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(samples, samples[1:]))


def build_runs(events: Sequence[dict], decisions: Sequence[dict] = (),
               trajectory: Sequence[Sample] = ()) -> List[Run]:
    """Group events into runs (one per command, resumes included) and
    attach the planner queries and trajectory of each."""
    runs: Dict[int, Run] = {}
    for e in sorted(events, key=lambda e: e["stamp"]):
        run = runs.get(e["run"])
        if e["event"] == "command_start":
            run = runs[e["run"]] = Run(e["run"], e["stamp"], e["targets"],
                                       setup=e.get("setup") or {})
        elif run is None:
            run = runs[e["run"]] = Run(e["run"], e["stamp"], [], partial=True)
        run.last_s = e["stamp"]
        if e["event"] == "command_resume":
            run.resumes.append(e["stamp"])
        elif e["event"] == "visit_start":
            run.visits.append(Visit(e["step"], e["target"], e["stage"], e["room_id"],
                                    e["location_id"], e["cost_m"], e["goals"], e["stamp"]))
        elif e["event"] == "visit_end":
            for v in reversed(run.visits):
                if v.end is None and v.step == e["step"] and v.location_id == e["location_id"]:
                    v.end = e
                    break
        elif e["event"] == "command_end":
            run.ends.append(e)

    ordered = sorted(runs.values(), key=lambda r: r.start_s)
    queries = sorted(decisions, key=lambda d: d["stamp"])
    for run in ordered:
        run.decisions = [d for d in queries if run.start_s <= d["stamp"] <= run.last_s]
        # A query belongs to the first visit starting at or after it (the
        # session plans, then the visit starts).
        previous = None
        for v in run.visits:
            v.decisions = [d for d in run.decisions if d["stamp"] <= v.start_s
                           and (previous is None or d["stamp"] > previous)]
            previous = v.start_s
        run.samples = [s for s in trajectory if run.start_s <= s[0] <= run.last_s]
        for v in run.visits:
            end = v.end["stamp"] if v.end else run.last_s
            v.distance_m = path_length([s for s in run.samples if v.start_s <= s[0] <= end])
    return ordered


def paused_s(run: Run) -> float:
    """Time between a paused result and the next resume (waiting for a person)."""
    total = 0.0
    for end in run.ends:
        if end["status"] != "paused":
            continue
        later = [r for r in run.resumes if r >= end["stamp"]]
        if later:
            total += min(later) - end["stamp"]
    return total


def run_metrics(run: Run, step_cost_m: float) -> dict:
    final = run.final or {}
    targets = final.get("targets") or [
        {"name": t["name"], "known": t["known"], "status": "pending"} for t in run.targets]
    steps = final.get("steps", sum(v.status in ("completed", "failed") for v in run.visits))
    distance = path_length(run.samples)
    duration = run.last_s - run.start_s
    answered = [d for d in run.decisions if d.get("raw") is not None]
    latencies = [d["latency_s"] for d in answered]
    found = sum(t["status"] == "found" for t in targets)
    return {
        "run": run.run,
        "setup": run.setup,
        "partial_recording": run.partial,
        "status": final.get("status", "unfinished"),
        "reason": final.get("reason", ""),
        "targets": targets,
        "found": found,
        "success": bool(targets) and found == len(targets),
        "duration_s": round(duration, 1),
        "active_s": round(duration - paused_s(run), 1),
        "distance_m": round(distance, 2),
        "visits": steps,
        "step_cost_m": step_cost_m,
        "cost_m": round(distance + step_cost_m * steps, 2),
        "pauses": sum(e["status"] == "paused" for e in run.ends),
        "planner_queries": len(run.decisions),
        "llm_answers": len(answered),
        "fallbacks": sum(bool(d.get("fallback")) for d in run.decisions),
        "median_latency_s": round(statistics.median(latencies), 3) if latencies else None,
        "trajectory_samples": len(run.samples),
    }


def summarize(metrics: Sequence[dict]) -> dict:
    """Experiment summary over runs (e.g. several report.json files)."""
    done = [m for m in metrics if m["status"] != "unfinished"]

    def mean(key):
        return round(statistics.mean(m[key] for m in done), 2) if done else None

    return {
        "runs": len(metrics),
        "finished": len(done),
        "success_rate": round(sum(m["success"] for m in done) / len(done), 3) if done else None,
        **{f"mean_{k}": mean(k) for k in ("active_s", "distance_m", "visits", "cost_m")},
        "pauses": sum(m["pauses"] for m in metrics),
        "fallbacks": sum(m["fallbacks"] for m in metrics),
        "planner_queries": sum(m["planner_queries"] for m in metrics),
    }


# --- presentation -------------------------------------------------------------

def _location_name(graph: Optional[SceneGraph], location_id: str) -> str:
    if graph is not None:
        loc = graph.locations.get(location_id) or graph.object_goals.get(location_id)
        if loc is not None:
            return loc.category
    return location_id


def _room_name(graph: Optional[SceneGraph], room_id: str) -> str:
    if graph is not None and room_id in graph.rooms:
        return f"{graph.rooms[room_id].label} ({room_id})"
    return room_id


def _decision_text(visit: Visit) -> str:
    if visit.stage == "known":
        return "그래프에 있는 물체로 바로 이동"
    parts = []
    for d in visit.decisions:
        stage = STAGE_NAMES.get(d["stage"], d["stage"])
        if len(d["candidates"]) == 1:          # nothing to choose (the LLM is not asked)
            parts.append(f"{stage} 후보 1개")
            continue
        how = ("대체(최소 비용)" if d.get("fallback")
               else "LLM" if d.get("raw") is not None else "규칙")
        parts.append(f"{stage} {how} ({len(d['candidates'])}개 중)")
    return " → ".join(parts) if parts else "후보 1개"


def _result_text(visit: Visit) -> str:
    if visit.end is None:
        return "기록 끝남"
    text = STATUS_NAMES.get(visit.status, visit.status)
    if visit.end["reason"] not in ("", "none"):
        text += f" ({visit.end['reason']})"
    if visit.end["found"]:
        text += f", 발견: {', '.join(visit.end['found'])}"
    return text


def _seen_text(visit: Visit) -> str:
    if visit.end is None or not visit.end["seen"]:
        return "-"
    return ", ".join(f"{s['label']} {s['confidence']:.2f}"
                     for s in visit.end["seen"][:SEEN_IN_TABLE])


def _target_text(t: dict, graph: Optional[SceneGraph]) -> str:
    if t["status"] == "found":
        where = _location_name(graph, t["found_location"]) if t.get("found_location") else ""
        return f"{t['name']} 찾음" + (f" ({where})" if where else "")
    return f"{t['name']} {({'failed': '못 찾음', 'pending': '미완료'}).get(t['status'], t['status'])}"


def markdown(runs: Sequence[Run], metrics: Sequence[dict], graph: Optional[SceneGraph],
             images: Dict[int, str], title: str, notes: Sequence[str] = ()) -> str:
    lines = [f"# 탐색 기록: {title}", ""]
    if notes:
        lines += [f"> {n}" for n in notes] + [""]
    for run, m in zip(runs, metrics):
        names = ", ".join(t["name"] for t in m["targets"]) or "(명령 시작 기록 없음)"
        lines += [f"## 명령 {run.run}: {names}", ""]
        if run.partial:
            lines.append("- 기록이 명령 도중에 시작됨: 앞부분 없음")
        lines += [
            f"- 결과: {m['status']}" + (f" ({m['reason']})" if m["reason"] else "")
            + " — " + ", ".join(_target_text(t, graph) for t in m["targets"]),
            f"- 시간 {m['duration_s']} s (일시정지 제외 {m['active_s']} s), 이동 {m['distance_m']} m, "
            f"방문 {m['visits']}회, 비용 {m['cost_m']} m (방문당 {m['step_cost_m']} m)",
            f"- 판단 질의 {m['planner_queries']}회: LLM 응답 {m['llm_answers']}회, "
            f"대체 {m['fallbacks']}회"
            + (f", 지연 중앙값 {m['median_latency_s']} s" if m["median_latency_s"] is not None
               else ""),
        ]
        for end in run.ends:
            if end["status"] == "paused":
                where = _location_name(graph, end["location_id"]) if end["location_id"] else "-"
                lines.append(f"- 일시정지: {end['reason']} ({where}) {end['detail']}".rstrip())
        if run.run in images:
            lines += ["", f"![명령 {run.run}]({images[run.run]})"]
        lines += ["", "| # | 목표 | 방 | 탐색 위치 | 판단 | A* 비용 [m] | 이동 [m] | 시간 [s] "
                      "| 결과 | 본 것 (신뢰도) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for v in run.visits:
            duration = f"{v.duration_s:.1f}" if v.duration_s is not None else "-"
            lines.append(
                f"| {v.step} | {v.target} | {_room_name(graph, v.room_id)} | "
                f"{_location_name(graph, v.location_id)} | {_decision_text(v)} | {v.cost_m:.1f} | "
                f"{v.distance_m:.1f} | {duration} | {_result_text(v)} | {_seen_text(v)} |")
        lines.append("")
    return "\n".join(lines)


def render_run(path, run: Run, metrics: dict, graph: SceneGraph, occupancy, grid,
               rooms=None, title: str = "") -> None:
    """The run on the scene graph overview: trajectory colored by time,
    numbered visits (color = result), found targets."""
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    from athome.scene_graph.overview import (
        base_legend, draw_locations, draw_map, draw_rooms, finish_axes, location_corners,
        map_axes)

    found = [t for t in metrics["targets"] if t.get("found_position")]
    points = (location_corners(graph) + [(s[1], s[2]) for s in run.samples]
              + [tuple(t["found_position"][:2]) for t in found])
    fig, ax, bounds = map_axes(occupancy, points)
    draw_map(ax, occupancy, grid)
    draw_rooms(ax, graph, rooms)
    draw_locations(ax, graph)

    handles = base_legend(rooms is not None)
    if len(run.samples) > 1:
        xy = np.array([(s[1], s[2]) for s in run.samples])
        lines = LineCollection(np.stack([xy[:-1], xy[1:]], axis=1), cmap="viridis",
                               linewidths=2.2, zorder=8)
        lines.set_array(np.array([s[0] - run.start_s for s in run.samples[:-1]]))
        ax.add_collection(lines)
        fig.colorbar(lines, ax=ax, orientation="horizontal", fraction=0.035, pad=0.07,
                     label="time since command start [s]")
        ax.plot(*xy[0], marker="^", color="black", markersize=10, zorder=9)
        handles += [Line2D([], [], color=colormaps["viridis"](0.5), lw=2.2,
                           label="robot path (color = time)"),
                    Line2D([], [], marker="^", color="black", linestyle="", markersize=9,
                           label="start")]

    for v in run.visits:
        pose = v.pose
        if pose is None:
            continue
        color = VISIT_COLORS.get(v.status, "gray")
        loc = graph.locations.get(v.location_id)
        if loc is not None:
            center = ((loc.bbox_min[0] + loc.bbox_max[0]) / 2, (loc.bbox_min[1] + loc.bbox_max[1]) / 2)
            ax.plot([pose[0], center[0]], [pose[1], center[1]], ":", color=color, lw=1.2, zorder=8)
        ax.plot(pose[0], pose[1], "o", markersize=15, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=2.2, zorder=9)
        ax.text(pose[0], pose[1], str(v.step), ha="center", va="center", fontsize=8,
                fontweight="bold", color=color, zorder=10)
    handles += [Line2D([], [], marker="o", linestyle="", markersize=10, markerfacecolor="white",
                       markeredgecolor=c, markeredgewidth=2,
                       label=f"visit {s} (number = step)")
                for s, c in VISIT_COLORS.items() if any(v.status == s for v in run.visits)]

    for t in found:
        x, y = t["found_position"][:2]
        ax.plot(x, y, marker="*", markersize=18, color="magenta", markeredgecolor="black", zorder=11)
        ax.text(x, y + 0.25, f"FOUND {t['name']}", ha="center", fontsize=8, color="magenta",
                fontweight="bold", zorder=11,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))
    if found:
        handles.append(Line2D([], [], marker="*", linestyle="", markersize=12, color="magenta",
                              markeredgecolor="black", label="found target"))

    m = metrics
    head = (f"run {run.run}: {', '.join(t['name'] for t in m['targets'])} -> {m['status']}"
            + (f" ({m['reason']})" if m["reason"] else "")
            + f", found {m['found']}/{len(m['targets'])}")
    stats = (f"{m['duration_s']} s (active {m['active_s']} s), {m['distance_m']} m, "
             f"visits {m['visits']}, cost {m['cost_m']} m (step cost {m['step_cost_m']} m); "
             f"planner queries {m['planner_queries']}, fallbacks {m['fallbacks']}")
    finish_axes(ax, bounds, "\n".join(s for s in (title, head, stats) if s), handles)
    fig.savefig(path, dpi=150, bbox_inches="tight")
