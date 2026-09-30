import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from athome.config import load_robot_config, map_version_of
from athome.navigation import GridMap
from athome.preflight import (
    Check,
    Status,
    config_checks,
    free_space_agreement,
    report,
    server_checks,
)
from athome.scene_graph.location_policy import REPO

CLIP_MODEL = "ViT-H-14/laion2b_s32b_b79k"


def _obj(oid, tag, lo, hi, role):
    return {"object_id": oid, "room_id": "room_1", "semantic_tag": tag,
            "bbox": {"min": [*lo, 0.0], "max": [*hi, 0.7]}, "role": role}


def _environment(tmp_path, hfov=69.4, provenance="same", rooms=True, graph=True):
    """A 5 x 3 m room with a table and a chair, built like build_scene_graph.py."""
    image = np.full((60, 100), 254, np.uint8)
    image[0, :] = image[-1, :] = 0
    image[:, 0] = image[:, -1] = 0
    (tmp_path / "m.pgm").write_bytes(b"P5\n100 60\n255\n" + image.tobytes())
    (tmp_path / "m.yaml").write_text(
        "image: m.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    version = map_version_of(tmp_path / "m.yaml") if provenance == "same" else provenance
    if graph:
        (tmp_path / "g.json").write_text(json.dumps({
            "coordinate_frame": "athome_z_up",
            "rooms": [{"room_id": "room_1", "room_label": "living room", "floor_z_m": 0.0}],
            "workspaces": [{"workspace_id": "ws_table", "room_id": "room_1",
                            "source_object_id": "table_1", "child_object_ids": []}],
            "objects": [_obj("table_1", "table", (2.0, 0.8), (2.8, 1.2), "source"),
                        _obj("chair_1", "chair", (3.4, 1.8), (3.8, 2.2), "standalone")],
            "provenance": {"map_version": version},
        }))
    if rooms:
        np.savez_compressed(tmp_path / "g.rooms.npz", labels=np.ones((60, 100), np.int32),
                            origin_xy_m=np.zeros(2), resolution_m=np.asarray(0.05))
    data = REPO / "configs" / "data"
    (tmp_path / "robot.yaml").write_text(
        "map: {yaml: m.yaml, floor_z_m: 0.0}\nrobot: {inflation_radius: 0.2}\n"
        "navigation: {goal_offset: 0.45, path_planner: none}\n"
        f"sensors: {{camera_hfov_deg: {hfov}}}\n"
        "search: {observation_range_m: 3.0}\n"
        f"scene_graph: {{path: g.json, target_categories: {data / 'target_categories.v5.json'}, "
        f"search_locations: {data / 'search_locations.json'}, "
        f"scene_scope: {data / 'scene_scope.json'}}}\n")
    return load_robot_config(tmp_path / "robot.yaml")


def _by_name(checks):
    return {c.name: c for c in checks}


def test_config_checks_pass_for_a_consistent_setup(tmp_path):
    checks = config_checks(_environment(tmp_path))
    assert [c.status for c in checks] == [Status.PASS] * len(checks), checks
    assert {"지도", "씬 그래프", "그래프와 지도", "목표 자세", "바닥 높이", "방 관측 비율",
            "관측 회전", "학습 데이터 규칙"} == set(_by_name(checks))


def test_config_checks_catch_mismatches(tmp_path):
    checks = _by_name(config_checks(
        _environment(tmp_path, hfov=56.0, provenance="sha256:other", rooms=False)))
    assert checks["그래프와 지도"].status == Status.FAIL
    # 640x480 D435i RGB: 6 headings of 60 deg leave gaps.
    assert checks["관측 회전"].status == Status.FAIL
    assert checks["방 관측 비율"].status == Status.WARN
    assert checks["목표 자세"].status == Status.PASS


def test_missing_graph_fails_without_dependent_checks(tmp_path):
    checks = _by_name(config_checks(_environment(tmp_path, graph=False)))
    assert checks["씬 그래프"].status == Status.FAIL
    assert "그래프와 지도" not in checks and "목표 자세" not in checks
    assert checks["지도"].status == Status.PASS


class _FakeServers(BaseHTTPRequestHandler):
    """/v1/models (vLLM / OpenAI) and /embed_text (serve_clip_text.py)."""
    models = ["room", "search_location", "workspace", "qwen3-4b"]

    def do_GET(self):
        if self.headers.get("Authorization") != "Bearer test-key":
            return self._send(401, {"error": "unauthorized"})
        self._send(200, {"data": [{"id": m} for m in self.models]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["model"] != CLIP_MODEL:
            return self._send(400, {"error": f"model is {CLIP_MODEL}"})
        self._send(200, {"model": CLIP_MODEL, "embeddings": [[0.6, 0.8]]})

    def _send(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeServers)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _with_servers(config, url, models=None, clip_model=CLIP_MODEL):
    llm = {"base_url": url, "api_key_env": "TEST_KEY", "timeout": 2.0,
           "models": models or {"room": "room", "workspace": "workspace",
                                "standalone": "search_location"}}
    return dataclasses.replace(
        config,
        planner={"type": "llm", "llm": llm},
        command_parser={"base_url": url, "model": "qwen3-4b", "api_key_env": "TEST_KEY",
                        "timeout": 2.0},
        matcher={"type": "clip", "clip": {"base_url": url, "model": clip_model, "timeout": 2.0}})


def test_server_checks_pass(tmp_path, server_url):
    config = _with_servers(_environment(tmp_path), server_url)
    checks = server_checks(config, environ={"TEST_KEY": "test-key"})
    assert [c.status for c in checks] == [Status.PASS] * 3, checks


def test_server_checks_report_missing_key_model_and_clip_mismatch(tmp_path, server_url):
    config = _with_servers(_environment(tmp_path), server_url,
                           models={"room": "room", "workspace": "workspace_v2"},
                           clip_model="ViT-B-32/openai")
    checks = _by_name(server_checks(config, environ={"TEST_KEY": "test-key"}))
    assert checks["플래너 LLM 서버"].status == Status.FAIL
    assert "workspace_v2" in checks["플래너 LLM 서버"].detail
    assert checks["CLIP 매칭 서버"].status == Status.FAIL
    assert checks["명령 해석 서버"].status == Status.PASS
    # The key is looked up in the process environment only (as search_server).
    no_key = _by_name(server_checks(config, environ={}))
    assert no_key["명령 해석 서버"].status == Status.FAIL
    assert "TEST_KEY" in no_key["명령 해석 서버"].detail


def test_server_checks_without_servers_configured(tmp_path):
    config = dataclasses.replace(_environment(tmp_path), planner={"type": "min_cost"},
                                 command_parser={}, matcher={"type": "label"})
    checks = _by_name(server_checks(config, environ={}))
    assert checks["플래너 LLM 서버"].status == Status.WARN
    assert checks["명령 해석 서버"].status == Status.SKIP
    assert checks["CLIP 매칭 서버"].status == Status.SKIP


def test_unreachable_server_is_a_failed_check(tmp_path):
    config = dataclasses.replace(_environment(tmp_path), planner={"type": "llm", "llm": {
        "base_url": "http://127.0.0.1:9", "timeout": 1.0, "models": {"room": "room"}}})
    check = _by_name(server_checks(config, environ={}))["플래너 LLM 서버"]
    assert check.status == Status.FAIL and "연결 실패" in check.detail


def test_free_space_agreement():
    free = np.zeros((40, 40), bool)
    free[10:30, 10:30] = True
    grid = GridMap(free, (0.0, 0.0), 0.05)
    assert free_space_agreement(grid, free, (0.0, 0.0), 0.05) == 1.0
    # The same free square on a coarser grid.
    coarse = np.zeros((20, 20), bool)
    coarse[5:15, 5:15] = True
    assert free_space_agreement(grid, coarse, (0.0, 0.0), 0.1) == 1.0
    # Shifted by a quarter of the square: overlap 15x20 of union 25x20 cells.
    assert free_space_agreement(grid, free, (0.25, 0.0), 0.05) == pytest.approx(15 / 25)


def test_report_counts_and_exit_code():
    lines, code = report([("a", [Check("x", Status.PASS, "ok"), Check("y", Status.WARN, "w")])])
    assert code == 0 and lines[-1].startswith("결과: 통과 1, 경고 1, 실패 0")
    lines, code = report([("a", [Check("x", Status.FAIL, "bad")])])
    assert code == 1 and "[실패] x: bad" in "\n".join(lines)
