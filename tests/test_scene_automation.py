import importlib.util
from pathlib import Path

import numpy as np

from athome.data.hm3d.navmesh_levels import LevelRule, segment_levels


def grid_mesh(x0, x1, y0, y1, z_of, n=4):
    """Triangulated strip; z_of(x, y) gives the height."""
    xs, ys = np.linspace(x0, x1, n + 1), np.linspace(y0, y1, 3)
    verts = [(x, y, z_of(x, y)) for y in ys for x in xs]
    faces = []
    for j in range(2):
        for i in range(n):
            a, b = j * (n + 1) + i, j * (n + 1) + i + 1
            c, d = a + n + 1, b + n + 1
            faces += [(a, b, d), (a, d, c)]
    return np.array(verts, float), np.array(faces)


def merge(*meshes):
    verts, faces, off = [], [], 0
    for v, f in meshes:
        verts.append(v)
        faces.append(f + off)
        off += len(v)
    return np.vstack(verts), np.vstack(faces)


def test_two_levels_joined_by_stairs_are_split_and_first_steps_excluded():
    lower = grid_mesh(0, 2, 0, 1, lambda x, y: 0.0)
    stairs = grid_mesh(2, 4, 0, 1, lambda x, y: (x - 2) * 0.5, n=8)   # rises 1 m, sloped
    upper = grid_mesh(4, 6, 0, 1, lambda x, y: 1.0)
    v, f = merge(lower, stairs, upper)
    out = segment_levels(v, f, max_climb=0.2, rule=LevelRule(min_platform_area_m2=1.0))
    assert len(out["components"]) == 2
    heights = sorted(c["platform_heights_m"][0] for c in out["components"])
    assert heights == [0.0, 1.0]
    n_low, n_st = len(lower[1]), len(stairs[1])
    stair_ids = set(range(n_low, n_low + n_st))
    assert stair_ids <= set(out["excluded_triangle_ids"])   # incl. steps within 0.2 m


def test_small_flat_piece_is_not_a_platform():
    tiny = grid_mesh(0, 0.2, 0, 0.2, lambda x, y: 0.0, n=1)
    out = segment_levels(*tiny, max_climb=0.2)
    assert out["components"] == [] and len(out["excluded_triangle_ids"]) == len(tiny[1])


def load_decider():
    path = Path(__file__).resolve().parents[1] / "scripts/decide_component_membership.py"
    spec = importlib.util.spec_from_file_location("decide_component_membership", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ev(c, overlap, dz, floor="f"):
    return {"component": c, "floor_object_id": floor, "bbox_xy_overlap_m2": overlap,
            "object_bottom_minus_floor_center_m": dz}


def test_membership_rule():
    decide = load_decider().decide
    assert decide({"floor_bbox_evidence": [ev("A", 0.1, 0.0), ev("B", 0.0, -0.2)]}, 0.1)[0] == "A"
    assert decide({"floor_bbox_evidence": [ev("A", 0.1, 0.0), ev("B", 0.2, 0.0)]}, 0.1)[0] is None
    assert decide({"floor_bbox_evidence": [ev("A", 0.0, 0.0), ev("B", 0.0, 0.0)]}, 0.1)[0] is None
    assert decide({"floor_bbox_evidence": [ev("A", 0.3, -0.2), ev("B", 0.0, 0.0)]}, 0.1)[0] is None
    assert decide({"floor_bbox_evidence": [ev("B", 0.4, 2.2), ev("A", 0.0, 2.4)]}, 0.1)[0] == "B"


def load_inspector():
    path = Path(__file__).resolve().parents[1] / "scripts/inspect_component_rooms.py"
    spec = importlib.util.spec_from_file_location("inspect_component_rooms", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_component_names_continue_past_z():
    name = load_inspector().component_name
    assert [name(i) for i in (0, 1, 25, 26, 27, 51, 52)] == ["A", "B", "Z", "AA", "AB", "AZ", "BA"]


def test_nearest_floor_within_half_gap():
    nearest = load_inspector().nearest_floor
    envs = [{"floor_id": "floor_0", "floor_center_height_range_m": [-2.5, -2.3]},
            {"floor_id": "floor_1", "floor_center_height_range_m": [-0.4, 0.0]}]
    assert nearest(0.1, envs, 0.5)[0]["floor_id"] == "floor_1"
    assert nearest(-2.2, envs, 0.5)[0]["floor_id"] == "floor_0"
    assert nearest(1.2, envs, 0.5)[0] is None   # no floor level near this height


def test_selection_islands_accepts_both_schemas():
    from athome.data.hm3d.navmesh_levels import selection_islands
    single = {"island_id": 0, "retained_triangle_ids": [1]}
    assert selection_islands(single) == [single]
    multi = {"islands": [{"island_id": 0}, {"island_id": 3}]}
    assert [i["island_id"] for i in selection_islands(multi)] == [0, 3]


def test_room_split_across_levels():
    from athome.data.hm3d.floors import level_of, on_floor, split_room
    levels = [("floor_1", -12.54), ("floor_2", -10.64)]   # stair landing + hall of one region
    assert level_of(-12.5, levels) == "floor_1"
    assert level_of(-10.70, levels) == "floor_2"          # 6 cm below the floor centre: still on it
    assert level_of(-13.0, levels) == "floor_1"           # below every floor: lowest level
    box = lambda oid, z: {"object_id": oid, "room_id": "_4", "bbox": {"min": [0, 0, z]}}
    split = split_room([box("floor_a", -12.6), box("chair_1", -10.6), box("plant_2", -12.5)],
                       levels, {"floor_a": "floor_1"})
    assert split == {"floor_1": ["floor_a", "plant_2"], "floor_2": ["chair_1"]}
    env = {"room_ids": ["_4", "_5"], "partial_room_object_ids": {"_4": split["floor_2"]}}
    assert on_floor(env, {"object_id": "chair_1", "room_id": "_4"})
    assert not on_floor(env, {"object_id": "plant_2", "room_id": "_4"})
    assert on_floor(env, {"object_id": "x", "room_id": "_5"})


def test_degenerate_triangle_is_excluded():
    v, f = grid_mesh(0, 2, 0, 1, lambda x, y: 0.0)
    n = len(v)
    v = np.vstack([v, [[0.0, 1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 1.0, 0.0]]])   # collinear, on the edge
    f = np.vstack([f, [[n, n + 1, n + 2]]])
    out = segment_levels(v, f, max_climb=0.2)
    assert out["degenerate_triangle_ids"] == [len(f) - 1]
    assert len(f) - 1 in out["excluded_triangle_ids"]
    assert len(f) - 1 not in out["retained_triangle_ids"]


def test_robot_navmesh_islands_keep_everything_connected():
    from athome.data.hm3d.navmesh_levels import navmesh_components
    left = grid_mesh(0, 2, 0, 1, lambda x, y: 0.0)
    right = grid_mesh(3, 5, 0, 1, lambda x, y: 0.01)        # separate piece, 1 cm higher
    v, f = merge(left, right)
    out = navmesh_components(v, f)
    assert len(out["components"]) == 2 and out["excluded_triangle_ids"] == []
