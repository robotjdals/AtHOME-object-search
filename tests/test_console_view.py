import importlib.util
import os
import re
from pathlib import Path

from athome.execution.console_view import SearchTimeline, explain_reason, found_photo, photo_path


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def timeline():
    clock = Clock()
    return SearchTimeline(clock=clock, wall=clock), clock


def visit_start(step=1, stamp=10.0):
    return {"event": "visit_start", "step": step, "target": "cup", "stage": "room", "location_id": "workspace:a",
            "cost_m": 2.5, "stamp": stamp}


def visit_end(step=1, found=(), status="completed", reason="none", stamp=22.0):
    return {"event": "visit_end", "step": step, "status": status, "reason": reason, "location_id": "workspace:a",
            "found": list(found), "frames": 12, "relooks": [{"kind": "found"}], "policy_fallback": True, "stamp": stamp}


def test_a_visit_shows_its_place_time_and_result_newest_first():
    t, _ = timeline()
    t.on_event({"event": "command_start"})
    t.on_event(visit_start(1, 10.0))
    t.on_status({"location": "workspace:a", "location_name": "desk (kitchen)"})
    assert t.view()["visits"][0]["status_text"] == "진행 중"
    t.on_event(visit_end(1, found=["cup"]))
    t.on_event(visit_start(2, 30.0))
    visits = t.view()["visits"]
    assert [v["step"] for v in visits] == [2, 1]
    assert visits[1]["place"] == "desk (kitchen)"
    assert visits[1]["duration_s"] == 12.0
    assert visits[1]["found"] == ["cup"]
    assert visits[1]["relooks"] == 1
    assert visits[1]["fallback"] is True
    assert visits[1]["stage"] == "방 선택"


def test_a_place_without_status_shows_its_category():
    t, _ = timeline()
    t.on_event({"event": "command_start"})
    t.on_event(dict(visit_start(), location_id="workspace:room_1:computer desk_3"))
    assert t.view()["visits"][0]["place"] == "computer desk"


def test_elapsed_stops_while_paused_and_continues_on_resume():
    t, clock = timeline()
    assert t.view()["elapsed_s"] is None and t.view()["started_at"] is None
    t.on_event({"event": "command_start"})
    assert t.view()["started_at"] == 100.0
    clock.now += 10
    assert t.view()["elapsed_s"] == 10.0
    t.on_event({"event": "command_end", "status": "paused", "reason": "perception_lost", "targets": []})
    clock.now += 100
    assert t.view()["elapsed_s"] == 10.0
    assert t.view()["ended"]["explained"]["text"] == "인지 결과가 끊김"
    t.on_event({"event": "command_resume"})
    assert t.view()["ended"] is None
    clock.now += 5
    assert t.view()["elapsed_s"] == 15.0


def test_a_new_command_clears_the_last_one():
    t, _ = timeline()
    t.on_event({"event": "command_start"})
    t.on_event(visit_start())
    t.on_event(visit_end(found=["cup"]))
    t.on_decision({"fallback": True, "error": "timeout"})
    t.on_event({"event": "command_start"})
    view = t.view()
    assert view["visits"] == [] and view["found"] == {} and view["planner"]["calls"] == 0


def test_found_place_during_the_command_and_details_at_its_end():
    t, _ = timeline()
    t.on_event({"event": "command_start"})
    t.on_event(visit_start())
    t.on_status({"location": "workspace:a", "location_name": "desk (kitchen)"})
    t.on_event(visit_end(found=["cup"]))
    assert t.view()["found"]["cup"]["place"] == "desk (kitchen)"
    t.on_event({"event": "command_end", "status": "completed", "reason": "", "targets": [
        {"name": "cup", "status": "found", "found_location": "workspace:a", "found_label": "mug",
         "found_position": [1.0, 2.0, 0.5]},
        {"name": "book", "status": "failed"}]})
    view = t.view()
    assert view["found"]["cup"]["label"] == "mug"
    assert "book" not in view["found"]
    assert view["ended"]["explained"] is None


def test_planner_shows_its_last_call_and_counts_fallbacks():
    t, _ = timeline()
    t.on_decision({"fallback": True, "error": "connection refused"})
    assert t.view()["planner"] == {"calls": 1, "fallbacks": 1, "last_fallback": True, "error": "connection refused"}
    t.on_decision({"fallback": False, "error": ""})
    planner = t.view()["planner"]
    assert planner["last_fallback"] is False and planner["fallbacks"] == 1


def test_unknown_reason_stays_readable():
    assert explain_reason("something_new") == {"text": "something_new", "action": ""}


def test_every_visit_reason_is_explained():
    from athome.execution.console_view import REASONS
    from athome.execution.visit import VisitReason

    missing = [r.value for r in VisitReason if r is not VisitReason.NONE and r.value not in REASONS]
    assert missing == []


def _photo(directory, name, when):
    path = directory / name
    path.write_bytes(b"jpg")
    os.utime(path, (when, when))


def test_found_photo_takes_this_commands_newest(tmp_path):
    _photo(tmp_path, "a_cmd1_trash_can.jpg", 50.0)      # the same number before the search server restarted
    assert found_photo(tmp_path, 1, "trash can", 100.0) is None
    _photo(tmp_path, "b_cmd1_trash_can.jpg", 120.0)
    _photo(tmp_path, "c_cmd1_trash_can.jpg", 130.0)
    _photo(tmp_path, "d_cmd2_trash_can.jpg", 140.0)
    assert found_photo(tmp_path, 1, "trash can", 100.0) == "c_cmd1_trash_can.jpg"


def test_found_photo_without_a_folder_or_command(tmp_path):
    assert found_photo(tmp_path / "none", 1, "cup", 0.0) is None
    assert found_photo(tmp_path, None, "cup", 0.0) is None


def test_photo_path_serves_only_jpegs_directly_in_the_folder(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"jpg")
    (tmp_path / "a.txt").write_bytes(b"x")
    (tmp_path / ".hidden.jpg").write_bytes(b"x")
    outside = tmp_path.parent / "outside_console_view.jpg"
    outside.write_bytes(b"x")
    try:
        assert photo_path(tmp_path, "a.jpg") == tmp_path / "a.jpg"
        for name in ("a.txt", ".hidden.jpg", "../outside_console_view.jpg", "..\\x.jpg", "none.jpg", ""):
            assert photo_path(tmp_path, name) is None
    finally:
        outside.unlink()


def test_the_page_uses_only_ids_it_has():
    path = Path(__file__).resolve().parents[1] / "ros2/athome_ros/athome_ros/operator_console_page.py"
    spec = importlib.util.spec_from_file_location("operator_console_page", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    page = module.PAGE
    used = set(re.findall(r'\$\("([\w-]+)"\)', page)) | set(re.findall(r'put\("([\w-]+)"', page))
    have = set(re.findall(r'\bid="([\w-]+)"', page))
    assert used - have == set()
    # Works without internet on the robot's network.
    assert not re.search(r'(src|href)="https?:', page)


def test_a_command_waits_for_perception():
    from athome.execution.console_view import perception_state
    loading = perception_state(None, 15.0)
    assert not loading["ready"] and "2~3분" in loading["text"]
    assert perception_state(3.0, 15.0) == {"ready": True, "short": "", "text": ""}
    assert perception_state(15.0, 15.0)["ready"]
    lost = perception_state(40.0, 15.0)
    assert not lost["ready"] and "40초" in lost["short"]
