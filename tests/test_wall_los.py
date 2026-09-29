import math

import numpy as np
import pytest

from athome.symbolic.wall_los import WallGeometry, WallLosObserver, WallLosSpec, clip_to_z_band

SPEC = WallLosSpec(range_m=3.0, occluder_categories=("wall",), band_above_floor_m=(0.3, 1.8),
                   contact_tolerance_m=0.02)


def wall_quad(x, y0, y1, z0=0.0, z1=2.5):
    """Vertical wall surface at x = const as two triangles."""
    a, b, c, d = (x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)
    return [np.array([a, b, c]), np.array([a, c, d])]


def test_clip_to_band():
    tri = np.array([(0, 0, 0), (1, 0, 0), (0, 0, 3)], dtype=float)
    poly = clip_to_z_band(tri, 1.0, 2.0)
    assert poly[:, 2].min() == pytest.approx(1.0) and poly[:, 2].max() == pytest.approx(2.0)
    assert len(clip_to_z_band(tri, 4.0, 5.0)) == 0


def test_far_side_blocked_near_side_visible_other_storey_ignored():
    # Wall with two surfaces (x = 2.00 and 2.06) and an upper-storey wall at x = 1.
    tris = wall_quad(2.0, 0.0, 4.0) + wall_quad(2.06, 0.0, 4.0) + wall_quad(1.0, 0.0, 4.0, 3.0, 5.0)
    observer = WallLosObserver(WallGeometry(tris, (0.3, 1.8)), {
        1: ((0.5, 1.0, 0.0), (0.7, 1.2, 0.5)),     # open floor, near
        2: ((2.5, 1.0, 0.0), (2.7, 1.2, 0.5)),     # behind the wall
        3: ((1.9, 2.0, 1.0), (2.0, 2.2, 1.2)),     # mounted on the near side, flush
        4: ((2.03, 2.5, 1.0), (2.2, 2.7, 1.2)),    # against the far side, bbox overlaps wall
        5: ((1.999, 3.0, 1.0), (2.0, 3.2, 1.2)),   # 1 mm thin annotation flush with the wall
    }, SPEC)
    seen = observer.observe(0.55, 1.55, 0.0, 0.0)
    assert set(seen) == {1, 3, 5}
    assert seen[1] == pytest.approx(0.35, abs=1e-3)  # nearest footprint point (0.55, 1.2)


def test_range_is_measured_to_nearest_footprint_point():
    observer = WallLosObserver(WallGeometry([], (0.3, 1.8)), {7: ((2.9, 0.0, 0.0), (3.9, 0.2, 0.2))}, SPEC)
    assert observer.observe(0.0, 0.1, 0.0, 0.0) == {7: pytest.approx(2.9)}


def test_out_of_range_and_no_walls():
    observer = WallLosObserver(WallGeometry([], (0.3, 1.8)),
                               {5: ((8.0, 8.0, 0.0), (8.2, 8.2, 0.2)),
                                6: ((2.0, 0.0, 0.0), (2.2, 0.2, 0.2))}, SPEC)
    assert set(observer.observe(0.5, 0.5, 0.0, 0.0)) == {6}
