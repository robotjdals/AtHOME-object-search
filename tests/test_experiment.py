import os
import time

from athome.execution.experiment import (
    latest_found_photo,
    load_plan,
    load_runs,
    append_run,
    pair_order,
    progress,
    remaining,
    start_names,
    summarize,
)

PLAN = [{"scenario": 1, "object": "laptop", "start": "bedroom"},
        {"scenario": 2, "object": "book", "start": "hall"},
        {"scenario": 3, "object": "plant", "start": "bedroom"}]


def test_pairs_alternate_which_planner_goes_first():
    assert pair_order(1) == ("trained", "base")
    assert pair_order(2) == ("base", "trained")
    order = [(row["scenario"], v) for row, v in remaining(PLAN, [])]
    assert order == [(1, "trained"), (1, "base"), (2, "base"), (2, "trained"), (3, "trained"), (3, "base")]


def test_remaining_skips_valid_runs_and_keeps_invalid_ones():
    runs = [{"scenario": 1, "variant": "trained", "valid": True},
            {"scenario": 1, "variant": "base", "valid": False},
            {"scenario": 2, "variant": "base", "valid": True}]
    order = [(row["scenario"], v) for row, v in remaining(PLAN, runs)]
    assert order == [(1, "base"), (2, "trained"), (3, "trained"), (3, "base")]
    assert progress(PLAN, runs) == {"total": 6, "valid": 2, "invalid": 1}


def test_start_names_in_plan_order():
    assert start_names(PLAN) == ["bedroom", "hall"]


def test_summarize_reads_the_search_records():
    events = [
        {"event": "command_start", "setup": {"planner_variant": "base"}},
        {"event": "visit_end", "location_id": "workspace:room_1:desk_44", "status": "completed", "nav_attempts": 1},
        {"event": "visit_end", "location_id": "workspace:room_2:desk_18", "status": "canceled", "nav_attempts": 2},
        {"event": "command_end", "status": "completed", "reason": "", "detail": "", "steps": 2,
         "targets": [{"name": "laptop", "status": "found", "found_location": "workspace:room_2:desk_18",
                      "found_label": "laptop", "found_position": [1.0, 2.0, 0.4]}]},
    ]
    decisions = [{"fallback": False}, {"fallback": True}]
    s = summarize(events, decisions)
    assert s["planner_variant_used"] == "base"
    assert s["steps_completed"] == 1 and s["steps_logged"] == 2 and s["nav_attempts"] == 3
    assert s["visited"] == ["workspace:room_1:desk_44", "workspace:room_2:desk_18"]
    assert s["target_status"] == "found" and s["found_location"] == "workspace:room_2:desk_18"
    assert (s["decisions"], s["fallbacks"]) == (2, 1)


def test_summarize_without_records_is_empty_not_an_error():
    s = summarize([], [])
    assert s["status"] is None and s["steps_completed"] == 0


def test_latest_found_photo_of_the_target_since_the_start(tmp_path):
    old = tmp_path / "20261106-200000_cmd1_vacuum_cleaner.jpg"
    old.write_bytes(b"x")
    os.utime(old, (time.time() - 100, time.time() - 100))
    since = time.time() - 10
    new = tmp_path / "20261111-120000_cmd3_vacuum_cleaner.jpg"
    new.write_bytes(b"x")
    (tmp_path / "20261111-120001_cmd3_cup.jpg").write_bytes(b"x")
    assert latest_found_photo(tmp_path, "vacuum cleaner", since) == new.name
    assert latest_found_photo(tmp_path, "laptop", since) is None
    assert latest_found_photo(tmp_path / "missing", "cup", since) is None


def test_plan_and_runs_files(tmp_path):
    plan = tmp_path / "scenarios.csv"
    plan.write_text("scenario,object,start\n2,book,hall\n", encoding="utf-8")
    assert load_plan(plan) == [{"scenario": 2, "object": "book", "start": "hall"}]
    runs = tmp_path / "runs.jsonl"
    assert load_runs(runs) == []
    append_run(runs, {"run": 1, "valid": True})
    append_run(runs, {"run": 2, "valid": False})
    assert [r["run"] for r in load_runs(runs)] == [1, 2]
