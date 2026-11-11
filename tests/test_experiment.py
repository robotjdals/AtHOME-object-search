import os
import time

from athome.execution.experiment import (
    abort_run,
    latest_found_photo,
    load_plan,
    load_runs,
    append_run,
    finish_run,
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
    assert progress(PLAN, runs) == {"total": 6, "valid": 2, "invalid": 1, "aborted": 0}


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


def _record(run=1):
    return {"run": run, "scenario": 1, "variant": "trained", "object": "laptop"}


def test_finish_run_saves_the_search_records(tmp_path):
    events = [{"event": "command_start", "setup": {"planner_variant": "trained"}},
              {"event": "command_end", "status": "completed", "steps": 0, "targets": [{"status": "not_found"}]}]
    r = finish_run(dict(_record(3), t_start=time.time() - 5), events, [], 4, tmp_path, tmp_path / "found")
    assert r["action_status"] == 4 and r["status"] == "completed" and r["duration_s"] >= 5
    assert (r["events_file"], r["decisions_file"]) == ("events_3.jsonl", "decisions_3.jsonl")
    assert load_runs(tmp_path / "events_3.jsonl") == events and load_runs(tmp_path / "decisions_3.jsonl") == []


def test_run_aborted_before_the_verdict_is_kept_and_run_again(tmp_path):
    runs = tmp_path / "runs.jsonl"
    # Aborted on the way to the start pose: no search, nothing to summarize.
    moving = abort_run(dict(_record(1), move_to_start={"status": "canceled"}), [], [], tmp_path, tmp_path)
    assert moving["valid"] is None and moving["aborted"] and "t_end" not in moving
    # Aborted during the search: what it logged until the cancel is kept.
    events = [{"event": "command_start", "setup": {"planner_variant": "trained"}},
              {"event": "command_end", "status": "canceled", "steps": 1, "targets": [{"status": "searching"}]}]
    searching = abort_run(dict(_record(2), t_start=time.time()), events, [{"fallback": False}], tmp_path, tmp_path)
    assert searching["status"] == "canceled" and searching["decisions"] == 1
    assert searching["events_file"] == "events_2.jsonl" and (tmp_path / "events_2.jsonl").is_file()
    # Aborted at the verdict: the finished record is kept as it was, without a verdict.
    finished = finish_run(dict(_record(3), t_start=time.time()), events[:1], [], 4, tmp_path, tmp_path)
    t_end = finished["t_end"]
    finished = abort_run(finished, [{"event": "later"}], [], tmp_path, tmp_path)
    assert finished["t_end"] == t_end and finished["action_status"] == 4 and finished["valid"] is None
    assert load_runs(tmp_path / "events_3.jsonl") == events[:1]
    for r in (moving, searching, finished):
        append_run(runs, r)
    saved = load_runs(runs)
    assert [r["note"] for r in saved] == ["판정 전 중단"] * 3
    assert [(row["scenario"], v) for row, v in remaining(PLAN, saved)][:2] == [(1, "trained"), (1, "base")]
    assert progress(PLAN, saved) == {"total": 6, "valid": 0, "invalid": 0, "aborted": 3}
