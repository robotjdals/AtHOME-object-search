"""Planner comparison on the robot: the plan, the run order and the records,
without ROS (athome_ros.experiment runs it; the operator console shows it).

A plan (experiments/<name>/scenarios.csv) lists placements of objects the scene
graph does not hold. Every placement is searched once by each planner variant,
back to back with the object placed once; odd scenarios run the trained planner
first, even ones the base model, so order effects cancel. A run the operator
marks invalid (a cause unrelated to the system) is run again.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

VARIANTS = ("trained", "base")
SHORTEST_PATH_M = (1.0, 30.0)       # ObjectNav start rule, as the simulation tables


def load_plan(path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["scenario"] = int(r["scenario"])
    return rows


def pair_order(scenario: int) -> Tuple[str, str]:
    """Odd scenarios run the trained planner first, even ones the base model."""
    return VARIANTS if scenario % 2 == 1 else VARIANTS[::-1]


def start_names(plan: Sequence[dict]) -> List[str]:
    return list(dict.fromkeys(r["start"] for r in plan))


def remaining(plan: Sequence[dict], runs: Sequence[dict]) -> List[Tuple[dict, str]]:
    """(scenario row, variant) still without a valid run, in run order."""
    done = {(r["scenario"], r["variant"]) for r in runs if r.get("valid")}
    return [(row, v) for row in plan for v in pair_order(row["scenario"]) if (row["scenario"], v) not in done]


def progress(plan: Sequence[dict], runs: Sequence[dict]) -> dict:
    valid = {(r["scenario"], r["variant"]) for r in runs if r.get("valid")}
    return {"total": 2 * len(plan), "valid": len(valid),
            "invalid": sum(1 for r in runs if r.get("valid") is False),
            "aborted": sum(1 for r in runs if r.get("aborted"))}


def load_runs(path) -> List[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_run(path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_lines(path, records: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(events: Sequence[dict], decisions: Sequence[dict]) -> dict:
    """The search server's records of one command, reduced to the run record.
    steps_completed: visits whose move and observation finished (proposal
    Search Step); rechecks at the same location are part of the count only when
    the search server logged them as visits."""
    start = next((e for e in events if e.get("event") == "command_start"), {})
    end = next((e for e in events if e.get("event") == "command_end"), {})
    visits = [e for e in events if e.get("event") == "visit_end"]
    target = (end.get("targets") or [{}])[0]
    return {
        "planner_variant_used": (start.get("setup") or {}).get("planner_variant"),
        "status": end.get("status"), "reason": end.get("reason"), "detail": end.get("detail"),
        "steps_logged": end.get("steps"),
        "steps_completed": sum(1 for v in visits if v.get("status") == "completed"),
        "nav_attempts": sum(int(v.get("nav_attempts") or 0) for v in visits),
        "visited": [v.get("location_id") for v in visits],
        "target_status": target.get("status"),
        "found_location": target.get("found_location"),
        "found_label": target.get("found_label"),
        "found_position": target.get("found_position"),
        "decisions": len(decisions),
        "fallbacks": sum(1 for d in decisions if d.get("fallback")),
    }


def finish_run(record: dict, events: Sequence[dict], decisions: Sequence[dict], status, out_dir,
               found_dir) -> dict:
    """Fill a run record from the search's records (from ``record["t_start"]``)
    and save those records beside runs.jsonl as events_<run>/decisions_<run>."""
    out_dir, index = Path(out_dir), record["run"]
    record["t_end"] = time.time()
    record["duration_s"] = round(record["t_end"] - record["t_start"], 1)
    record["action_status"] = status
    record.update(summarize(events, decisions))
    record["found_photo"] = latest_found_photo(found_dir, record["object"], record["t_start"] - 1.0)
    write_lines(out_dir / f"events_{index}.jsonl", events)
    write_lines(out_dir / f"decisions_{index}.jsonl", decisions)
    record["events_file"], record["decisions_file"] = f"events_{index}.jsonl", f"decisions_{index}.jsonl"
    return record


def abort_run(record: dict, events: Sequence[dict], decisions: Sequence[dict], out_dir, found_dir) -> dict:
    """A run aborted before its verdict, kept without one (valid None) so
    remaining() runs it again; a search under way is summarized from what it logged."""
    if "t_start" in record and "t_end" not in record:
        finish_run(record, events, decisions, None, out_dir, found_dir)
    record.update(valid=None, aborted=True, note="판정 전 중단")
    return record


def latest_found_photo(found_dir, target: str, since_wall: float) -> Optional[str]:
    """Name of the newest found photo of ``target`` written since ``since_wall``
    (athome_ros.found_images: <time>_cmd<N>_<target>.jpg)."""
    found_dir = Path(found_dir)
    if not found_dir.is_dir():
        return None
    suffix = "_" + target.replace(" ", "_") + ".jpg"
    photos = [p for p in found_dir.glob("*.jpg") if p.name.endswith(suffix) and p.stat().st_mtime >= since_wall]
    return max(photos, key=lambda p: p.stat().st_mtime).name if photos else None


def check_starts(config, plan: Sequence[dict], starts: Dict[str, Sequence[float]]) -> List[dict]:
    """Shortest path from each scenario's start to its location with the search
    server's own planner (grid, goal rule, Dijkstra); ok when 1-30 m."""
    from athome.navigation import load_map_server
    from athome.navigation.planner import NavigationPlanner
    from athome.scene_graph.location_policy import excluded_categories
    from athome.scene_graph.query import STANDALONE, SceneGraph
    from athome.schemas import Pose2D

    graph = SceneGraph(json.loads(config.graph_path.read_text(encoding="utf-8")),
                       excluded_categories=excluded_categories(config.search_locations))
    grid = load_map_server(config.map_yaml, config.map_inflation, config.unknown_as_occupied)
    nav = NavigationPlanner(grid, config.navigation_config(), config.sight_grid(), config.goal_rooms())
    for lid, loc in graph.locations.items():
        nav.add_location(lid, loc.bbox_min, loc.bbox_max, loc.kind == STANDALONE, loc.room_id,
                         graph.other_boxes(lid))
    rows = []
    for r in plan:
        row = {"scenario": r["scenario"], "object": r["object"], "start": r["start"],
               "location_id": r["location_id"], "l_star_m": None, "ok": False, "problem": ""}
        pose = starts.get(r["start"])
        if pose is None:
            row["problem"] = "시작 위치 저장 안 됨"
        else:
            try:
                cost = nav.evaluate(Pose2D(*pose), [r["location_id"]]).get(r["location_id"])
            except Exception as e:  # noqa: BLE001 - reported per row (start not free etc.)
                cost, row["problem"] = None, str(e)
            if cost is not None:
                row["l_star_m"] = round(cost.cost, 2)
                lo, hi = SHORTEST_PATH_M
                row["ok"] = lo <= cost.cost <= hi
                if not row["ok"]:
                    row["problem"] = f"최단 경로 {cost.cost:.2f} m (1-30 m 밖)"
            elif not row["problem"]:
                row["problem"] = "목표 지점까지 경로 없음"
        rows.append(row)
    return rows
