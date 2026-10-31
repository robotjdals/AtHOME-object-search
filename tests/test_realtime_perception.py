import numpy as np

from athome.perception.lifting import Camera, FrameView, ScanWindow, box_region, pack_mask, unpack_mask

W, H = 640, 360
K = np.array([[450.0, 0, 320], [0, 450.0, 180], [0, 0, 1]])
# camera looking along map +x from the origin: map x -> cam z, map -y -> cam x, map -z -> cam y
T_MAP_CAM = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0.8], [0, 0, 0, 1]], dtype=float)
CAMERA = Camera(K, W, H, np.eye(4))
RNG = np.random.default_rng(0)


def cube(center, half, n):
    return RNG.uniform(-half, half, (n, 3)) + np.asarray(center)


def wall(x=4.0, n=4000):
    return np.c_[np.full(n, x), RNG.uniform(-2, 2, n), RNG.uniform(0.0, 2.0, n)]


def project_box(P):
    Pc = (P - T_MAP_CAM[:3, 3]) @ T_MAP_CAM[:3, :3]
    uv = (Pc @ K.T)[:, :2] / Pc[:, 2:3]
    return [*uv.min(0), *uv.max(0)]


def test_mask_round_trip():
    m = np.zeros((H, W), dtype=bool)
    m[100:140, 200:260] = True
    m[120, 230] = False
    assert (unpack_mask(pack_mask(m), W, H) == m).all()
    assert not unpack_mask(pack_mask(np.zeros((H, W), bool)), W, H).any()


def test_small_object_in_front_of_a_wall_is_lifted_not_the_wall():
    cup = cube((1.5, 0.0, 0.8), 0.04, 40)
    view = FrameView(CAMERA, T_MAP_CAM, np.r_[cup, wall()])
    box = view.lift(box_region(project_box(cup), W, H))
    assert box is not None
    assert abs(box.center[0] - 1.5) < 0.06 and abs(box.center[2] - 0.8) < 0.06
    assert max(box.size) < 0.12


def test_a_small_occluding_fragment_is_not_the_object():
    table = cube((2.0, 0.0, 0.5), 0.3, 600)
    sliver = cube((1.2, 0.0, 0.5), 0.01, 5)          # e.g. a chair edge in front, < 10 % of the points
    view = FrameView(CAMERA, T_MAP_CAM, np.r_[table, sliver])
    box = view.lift(box_region(project_box(table), W, H))
    # only the table's visible front part is kept; the sliver at 1.2 m is not chosen
    assert box is not None and 1.6 < box.center[0] < 2.4 and box.size[1] > 0.4


def test_no_points_in_the_region_gives_no_box():
    view = FrameView(CAMERA, T_MAP_CAM, wall())
    region = np.zeros((H, W), dtype=bool)
    region[:5, :5] = True                              # sky above the wall's top
    assert view.lift(region) is None


def test_scan_window_keeps_recent_scans_and_selects_by_time():
    w = ScanWindow(keep_s=1.0)
    for k in range(20):
        w.add(10.0 + 0.1 * k, np.full((2, 3), k, dtype=float))
    assert w.latest == 11.9
    assert len(w.points(11.45, 11.75)) == 6            # scans 15, 16, 17
    assert len(w.points(0.0, 10.5)) == 0               # dropped beyond keep_s
