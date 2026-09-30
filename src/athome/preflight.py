"""Checks before a search run: robot config and files, then servers.

    ros2 run athome_ros preflight --ros-args -p robot_config:=$PWD/configs/robot/demo.yaml

runs these and the checks of the running system (athome_ros.preflight).
Server checks use GET /v1/models and one CLIP text embedding: no LLM tokens.
API keys are looked up in this process's environment only, as the search
server does; a .env file is deliberately not read here.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from athome.execution.visit import VisitConfig, observation_yaw_tolerance
from athome.inference.chat import served_models
from athome.inference.http import post_json
from athome.navigation import GridMap, NavigationConfig, load_map_server, load_occupancy
from athome.scene_graph.location_policy import excluded_categories
from athome.scene_graph.overview import check_locations
from athome.scene_graph.query import SceneGraph


class Status(Enum):
    PASS = "통과"
    WARN = "경고"
    FAIL = "실패"
    SKIP = "건너뜀"


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str = ""


def guarded(name: str, check: Callable[[], Check]) -> Check:
    """A check that raises is reported as failed, so one broken part does not
    hide the others."""
    try:
        return check()
    except Exception as e:  # noqa: BLE001 - reported, not handled
        return Check(name, Status.FAIL, f"{type(e).__name__}: {e}")


def config_checks(config) -> List[Check]:
    """Files of ``config`` (athome.config.RobotConfig) and their agreement
    with the planner's training data. No network, no ROS."""
    out = [guarded("지도", lambda: _map(config))]
    graph_check, loaded = _graph(config)
    out.append(graph_check)
    if loaded is not None:
        raw, graph = loaded
        out += [
            _provenance(config, raw),
            guarded("목표 자세", lambda: _goals(config, graph)),
            _floor(graph),
        ]
    out += [
        _coverage(config),
        _rotation(config),
        _training_alignment(config),
    ]
    return out


def server_checks(config, environ: Mapping[str, str] = os.environ) -> List[Check]:
    return [
        guarded("플래너 LLM 서버", lambda: _planner_server(config, environ)),
        guarded("명령 해석 서버", lambda: _command_server(config, environ)),
        guarded("CLIP 매칭 서버", lambda: _clip_server(config)),
    ]


def free_space_agreement(grid: GridMap, other_free: np.ndarray, other_origin,
                         other_resolution: float) -> float:
    """Jaccard index of the robot-center free cells of ``grid`` and of another
    grid (e.g. the Nav2 costmap), compared at the other grid's cell centers.
    Cells outside ``grid`` count as not free there."""
    rows, cols = np.indices(other_free.shape)
    x = other_origin[0] + (cols + 0.5) * other_resolution
    y = other_origin[1] + (rows + 0.5) * other_resolution
    r = np.floor((y - grid.origin[1]) / grid.resolution).astype(int)
    c = np.floor((x - grid.origin[0]) / grid.resolution).astype(int)
    inside = (r >= 0) & (r < grid.shape[0]) & (c >= 0) & (c < grid.shape[1])
    ours = np.zeros(other_free.shape, bool)
    ours[inside] = grid.free[r[inside], c[inside]]
    union = ours | other_free
    return float((ours & other_free).sum() / union.sum()) if union.any() else 1.0


def report(sections: Sequence[Tuple[str, Sequence[Check]]]) -> Tuple[List[str], int]:
    """Printable lines and the exit code (1 if any check failed)."""
    lines, counts = [], {s: 0 for s in Status}
    for title, checks in sections:
        lines.append(f"== {title} ==")
        for c in checks:
            counts[c.status] += 1
            lines.append(f"  [{c.status.value}] {c.name}: {c.detail}")
    lines.append("결과: " + ", ".join(f"{s.value} {n}" for s, n in counts.items()))
    if counts[Status.FAIL]:
        lines.append("실패 항목을 해결한 뒤 탐색을 시작하세요.")
    return lines, int(counts[Status.FAIL] > 0)


def _map(config) -> Check:
    m = load_occupancy(config.map_yaml)
    known = int((m.occupancy >= 0).sum()) * m.resolution ** 2
    return Check("지도", Status.PASS,
                 f"{config.map_yaml.name} ({config.map_version}), 알려진 영역 {known:.0f} m²")


def _graph(config) -> Tuple[Check, Optional[tuple]]:
    name, path = "씬 그래프", config.graph_path
    if path is None:
        return Check(name, Status.FAIL, "scene_graph.path 없음"), None
    if not path.is_file():
        return Check(name, Status.FAIL, f"{path} 없음 (scripts/build_scene_graph.py)"), None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Same Search Locations as the search server.
        graph = SceneGraph(raw, excluded_categories=excluded_categories(config.search_locations))
    except Exception as e:  # noqa: BLE001 - reported
        return Check(name, Status.FAIL, f"{type(e).__name__}: {e}"), None
    return Check(name, Status.PASS, f"{path.name}: 방 {len(graph.rooms)}, "
                                    f"탐색 위치 {len(graph.locations)}"), (raw, graph)


def _provenance(config, raw: dict) -> Check:
    name = "그래프와 지도"
    built_on = (raw.get("provenance") or {}).get("map_version")
    if built_on is None:
        return Check(name, Status.WARN, "그래프에 지도 버전 기록 없음 "
                                        "(build_scene_graph.py로 만들면 기록됨)")
    if built_on != config.map_version:
        return Check(name, Status.FAIL,
                     f"그래프는 지도 {built_on}에서 만들어짐, 설정 지도는 {config.map_version}")
    return Check(name, Status.PASS, "같은 지도에서 만든 그래프")


def _goals(config, graph: SceneGraph) -> Check:
    grid = load_map_server(config.map_yaml, config.inflation_radius, config.unknown_as_occupied)
    checks = check_locations(graph, grid, NavigationConfig(
        goal_offset=config.goal_offset, goal_clearance=config.goal_clearance,
        goal_max_offset=config.goal_max_offset))
    bad = [c.location_id for c in checks.values() if c.problem]
    if bad:
        shown = ", ".join(bad[:5]) + (f" 외 {len(bad) - 5}개" if len(bad) > 5 else "")
        return Check("목표 자세", Status.WARN,
                     f"방문할 수 없는 탐색 위치 {len(bad)}개: {shown} "
                     "(scripts/render_graph_overview.py 그림 확인)")
    return Check("목표 자세", Status.PASS, f"탐색 위치 {len(checks)}개 모두 설 자리 있음 (지도 기준)")


def _floor(graph: SceneGraph) -> Check:
    if not graph.room_floor_z:
        return Check("바닥 높이", Status.WARN, "그래프에 방 바닥 높이 없음: 1.5 m 탐색 높이 범위 "
                                              "미적용 (map.floor_z_m 설정 후 그래프 다시 만들기)")
    return Check("바닥 높이", Status.PASS, f"방 {len(graph.room_floor_z)}개")


def _coverage(config) -> Check:
    name = "방 관측 비율"
    rooms = (config.graph_path.with_name(config.graph_path.stem + ".rooms.npz")
             if config.graph_path else None)
    missing = [what for what, ok in (
        ("search.observation_range_m", config.observation_range_m is not None),
        (rooms.name if rooms else "방 지도", rooms is not None and rooms.is_file())) if not ok]
    if missing:
        return Check(name, Status.WARN,
                     f"비활성 ({', '.join(missing)} 없음): 플래너 입력이 학습 데이터와 다름")
    return Check(name, Status.PASS, f"{rooms.name}, 관측 거리 {config.observation_range_m} m")


def _rotation(config) -> Check:
    name = "관측 회전"
    if config.camera_hfov_deg is None:
        return Check(name, Status.WARN, "sensors.camera_hfov_deg 없음: 방향 사이 사각지대 가능")
    heading_count = VisitConfig().heading_count
    try:
        tolerance = observation_yaw_tolerance(math.radians(config.camera_hfov_deg), heading_count)
    except ValueError as e:
        return Check(name, Status.FAIL, str(e))
    return Check(name, Status.PASS, f"시야 {config.camera_hfov_deg}°, {heading_count}방향, "
                                    f"회전 yaw 허용 오차 {math.degrees(tolerance):.1f}°")


def _training_alignment(config) -> Check:
    name = "학습 데이터 규칙"
    missing, absent = [], []
    for key, path in (("scene_graph.target_categories", config.target_categories),
                      ("scene_graph.search_locations", config.search_locations),
                      ("scene_graph.scene_scope", config.scene_scope)):
        if path is None:
            missing.append(key)
        elif not path.is_file():
            absent.append(str(path))
    if absent:
        return Check(name, Status.FAIL, f"파일 없음: {', '.join(absent)}")
    if missing:
        return Check(name, Status.WARN, f"설정 없음: {', '.join(missing)} (학습 데이터와 다른 규칙)")
    return Check(name, Status.PASS, "범주 묶음, 탐색 위치 정책, 수납 범위 규칙")


def _api_key(env_name: Optional[str], environ: Mapping[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """(key, problem). No variable configured: no key, no problem."""
    if not env_name:
        return None, None
    key = environ.get(env_name)
    if not key:
        return None, f"환경 변수 {env_name} 없음 (search_server를 실행할 셸에 설정)"
    return key, None


def _planner_server(config, environ) -> Check:
    name = "플래너 LLM 서버"
    kind = config.planner.get("type", "min_cost")
    if kind != "llm":
        return Check(name, Status.WARN, f"planner.type {kind}: LLM 대신 규칙으로 탐색")
    llm = config.planner["llm"]
    key, problem = _api_key(llm.get("api_key_env"), environ)
    if problem:
        return Check(name, Status.FAIL, problem)
    served = served_models(llm["base_url"], key, float(llm.get("timeout", 5.0)))
    wanted = sorted(set(llm.get("models", {}).values()))
    missing = [m for m in wanted if m not in served]
    if missing:
        return Check(name, Status.FAIL, f"{llm['base_url']}에 없는 모델 {missing} (있음: {served})")
    return Check(name, Status.PASS, f"{llm['base_url']}: 모델 {wanted}")


def _command_server(config, environ) -> Check:
    name = "명령 해석 서버"
    section = config.command_parser
    if not section:
        return Check(name, Status.SKIP, "command_parser 없음: 명령은 targets로만 받음")
    key, problem = _api_key(section.get("api_key_env"), environ)
    if problem:
        return Check(name, Status.FAIL, problem)
    served = served_models(section["base_url"], key, float(section.get("timeout", 10.0)))
    if section["model"] not in served:
        return Check(name, Status.FAIL, f"{section['base_url']}에 모델 {section['model']} 없음")
    return Check(name, Status.PASS, f"{section['base_url']}: {section['model']}")


def _clip_server(config) -> Check:
    name = "CLIP 매칭 서버"
    kind = config.matcher.get("type", "label")
    if kind != "clip":
        return Check(name, Status.SKIP, f"matcher.type {kind}: CLIP 안 씀")
    clip = config.matcher["clip"]
    # The server rejects requests for another model than the one it serves.
    reply = post_json(clip["base_url"].rstrip("/") + "/embed_text",
                      {"model": clip["model"], "texts": ["cup"]}, float(clip.get("timeout", 3.0)))
    return Check(name, Status.PASS,
                 f"{clip['base_url']}: {clip['model']}, {len(reply['embeddings'][0])}차원")
