import numpy as np
from shapely.geometry import box

from athome.search.coverage import RoomCoverage, floor_points


def test_floor_points_sample_the_floor_polygon():
    pts = floor_points([box(0, 0, 1, 1), box(1, 0, 2, 1)], spacing=0.5)
    assert len(pts) == 8 and pts[:, 0].max() < 2 and pts[:, 1].max() < 1


def test_observed_fraction_grows_with_what_is_seen():
    room = {"R": np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [9.0, 0.0]])}
    cov = RoomCoverage(room, lambda x, y, px, py: abs(px - x) <= 1.5)   # 1.5 m range, no walls
    assert cov.fraction("R") == 0.0
    cov.observe(0.5, 0.0)
    assert cov.fraction("R") == 0.5
    cov.observe(5.0, 0.0)
    assert cov.fraction("R") == 0.75
    assert cov.fraction("unknown") == 0.0


def test_room_points_from_segmentation_labels():
    from athome.search.coverage import room_points_from_labels
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[:, :5], labels[:, 5:] = 1, 2
    pts = room_points_from_labels(labels, (0.0, 0.0), 0.1, spacing=0.2)
    assert set(pts) == {"room_1", "room_2"}
    assert pts["room_1"][:, 0].max() < 0.5 <= pts["room_2"][:, 0].min()


def test_occupancy_line_of_sight_blocks_behind_walls():
    from athome.search.coverage import occupancy_line_of_sight
    occupied = np.zeros((20, 20), dtype=bool)
    occupied[:, 10] = True                            # wall at x = 1.0 .. 1.1
    sees = occupancy_line_of_sight(occupied, (0.0, 0.0), 0.1, range_m=3.0)
    assert sees(0.25, 1.0, 0.85, 1.0)                 # same side
    assert not sees(0.25, 1.0, 1.55, 1.0)             # behind the wall
    assert not sees(0.05, 0.05, 1.95, 1.95)            # diagonal through the wall
    assert not occupancy_line_of_sight(np.zeros((5, 5), bool), (0, 0), 0.1, 0.2)(0.0, 0.0, 0.4, 0.0)
