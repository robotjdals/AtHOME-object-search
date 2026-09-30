import json
import math

import numpy as np
import pytest

from athome.config import load_robot_config
from athome.navigation import GridMap, NavigationConfig, OccupancyMap, traversable_from_occupancy
from athome.schemas import Pose2D
from athome.scene_graph.location_policy import excluded_categories
from athome.scene_graph.overview import (
    DISCONNECTED,
    FREE,
    NEAR_OBSTACLE,
    NO_GOAL,
    OCCUPIED,
    UNKNOWN,
    RoomMap,
    check_locations,
    describe_problems,
    example_goals,
    map_image,
    render_overview,
    write_overview,
)
from athome.scene_graph.query import DEFAULT_EXCLUDED_CATEGORIES, SceneGraph

RES = 0.05


def _obj(oid, room, tag, lo, hi, role):
    return {"object_id": oid, "room_id": room, "semantic_tag": tag,
            "bbox": {"min": [*lo, 0.0], "max": [*hi, 0.7]}, "role": role}


# 5 x 3 m. A wall at x = 3.5 m cuts room_2 off from room_1; clutter fills
# x < 1.5, y >= 1.5 around the cabinet.
GRAPH = {
    "coordinate_frame": "athome_z_up",
    "rooms": [{"room_id": "room_1", "room_label": "living room", "floor_z_m": 0.0},
              {"room_id": "room_2", "room_label": "bedroom", "floor_z_m": 0.0}],
    "workspaces": [
        {"workspace_id": "ws_table", "room_id": "room_1", "source_object_id": "table_1",
         "child_object_ids": []},
        {"workspace_id": "ws_cabinet", "room_id": "room_1", "source_object_id": "cabinet_1",
         "child_object_ids": []},
        {"workspace_id": "ws_desk", "room_id": "room_2", "source_object_id": "desk_1",
         "child_object_ids": []},
    ],
    "objects": [
        _obj("table_1", "room_1", "table", (2.0, 0.8), (2.8, 1.2), "source"),
        _obj("cabinet_1", "room_1", "cabinet", (0.2, 2.2), (0.6, 2.8), "source"),
        _obj("desk_1", "room_2", "desk", (4.0, 1.2), (4.4, 1.6), "source"),
        _obj("chair_1", "room_1", "chair", (2.5, 2.2), (2.9, 2.6), "standalone"),
        _obj("door_1", "room_1", "door", (3.3, 1.0), (3.5, 2.0), "standalone"),
    ],
}


def _occupancy():
    occ = np.zeros((60, 100), np.int16)
    occ[:, 70] = 100
    occ[30:, :30] = 100
    occ[0, 0] = -1
    return occ


def _scene():
    occ = _occupancy()
    grid = GridMap(traversable_from_occupancy(occ, RES, 0.1), (0.0, 0.0), RES)
    labels = np.zeros(occ.shape, np.int32)
    labels[:, :70] = 1
    labels[:, 71:] = 2
    return (SceneGraph(GRAPH), OccupancyMap(occ, (0.0, 0.0), RES), grid,
            RoomMap(labels, (0.0, 0.0), RES))


def test_check_locations_flags_unvisitable_locations():
    graph, _, grid, _ = _scene()
    checks = check_locations(graph, grid, NavigationConfig())
    # The door is a fixture, not a Search Location.
    assert set(checks) == {"ws_table", "ws_cabinet", "ws_desk", "standalone:chair_1"}
    assert checks["ws_table"].problem is None and checks["ws_table"].goals
    assert checks["standalone:chair_1"].problem is None
    assert checks["ws_cabinet"].problem == NO_GOAL and not checks["ws_cabinet"].goals
    assert checks["ws_desk"].problem == DISCONNECTED and checks["ws_desk"].goals
    lines = describe_problems(checks)
    assert len(lines) == 3 and "ws_cabinet" in lines[1] and "ws_desk" in lines[2]


def test_example_goals_spread_around_location():
    goals = [Pose2D(math.cos(a), math.sin(a), 0.0)
             for a in np.linspace(-math.pi, math.pi, 36, endpoint=False)]
    picked = example_goals(goals, (0.0, 0.0), k=3)
    bearings = sorted(math.degrees(math.atan2(g.y, g.x)) for g in picked)
    assert len(picked) == 3
    assert all(b - a == pytest.approx(120) for a, b in zip(bearings, bearings[1:]))
    assert example_goals(goals[:2], (0.0, 0.0)) == goals[:2]


def test_map_image_colors_and_rows():
    _, occupancy, grid, _ = _scene()
    image = map_image(occupancy, grid)
    assert tuple(image[0, 70]) == OCCUPIED
    assert tuple(image[0, 0]) == UNKNOWN          # row 0 = lowest y, as the grid
    assert tuple(image[10, 69]) == NEAR_OBSTACLE  # free on the map, next to the wall
    assert tuple(image[10, 50]) == FREE
    with pytest.raises(ValueError):
        map_image(occupancy, GridMap(grid.free, (1.0, 0.0), RES))


def test_render_overview_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    graph, occupancy, grid, rooms = _scene()
    checks = check_locations(graph, grid, NavigationConfig())
    render_overview(tmp_path / "rooms.png", graph, occupancy, grid, checks, rooms, title="t")
    # Without a room map the room names go to the middle of their locations.
    render_overview(tmp_path / "plain.png", graph, occupancy, grid, checks)
    for name in ("rooms.png", "plain.png"):
        assert (tmp_path / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def _write_environment(tmp_path, graph):
    occ = _occupancy()
    image = np.where(occ == 100, 0, np.where(occ < 0, 205, 254)).astype(np.uint8)[::-1]
    (tmp_path / "m.pgm").write_bytes(b"P5\n100 60\n255\n" + image.tobytes())
    (tmp_path / "m.yaml").write_text(
        "image: m.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    (tmp_path / "g.json").write_text(json.dumps(graph))
    (tmp_path / "robot.yaml").write_text(
        "map: {yaml: m.yaml, floor_z_m: 0.0}\nrobot: {inflation_radius: 0.1}\n"
        "navigation: {goal_offset: 0.45, path_planner: none}\nscene_graph: {path: g.json}\n")
    return load_robot_config(tmp_path / "robot.yaml")


def test_write_overview_uses_robot_config_and_room_map(tmp_path):
    pytest.importorskip("matplotlib")
    config = _write_environment(tmp_path, GRAPH)
    _, _, _, rooms = _scene()
    np.savez_compressed(tmp_path / "g.rooms.npz", labels=rooms.labels,
                        origin_xy_m=np.asarray(rooms.origin), resolution_m=np.asarray(RES))
    checks = write_overview(config, tmp_path / "g.json", tmp_path / "g.overview.png")
    assert checks["ws_cabinet"].problem == NO_GOAL
    assert checks["ws_desk"].problem == DISCONNECTED
    assert checks["ws_table"].problem is None
    assert (tmp_path / "g.overview.png").is_file()


def test_write_overview_rejects_graph_of_another_map(tmp_path):
    config = _write_environment(tmp_path, {**GRAPH, "provenance": {"map_version": "sha256:0"}})
    with pytest.raises(ValueError, match="지도"):
        write_overview(config, tmp_path / "g.json", tmp_path / "g.overview.png")
    assert not (tmp_path / "g.overview.png").exists()


def test_excluded_categories_without_policy():
    assert excluded_categories(None) == DEFAULT_EXCLUDED_CATEGORIES
