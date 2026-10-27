import numpy as np
import pytest

from athome.navigation import OccupancyMap
from athome.scene_graph.room_segmentation import segment_rooms
from athome.scene_graph.wall_map import (
    WallBand,
    read_pcd_xyz,
    segmentation_free_space,
    wall_cells,
)

RES, FLOOR = 0.05, -0.4


def _box_points(x0, y0, x1, y1, top, step=0.05):
    """Points on the outline of a box footprint from the floor up to ``top``."""
    xs, ys = np.arange(x0, x1 + 1e-9, step), np.arange(y0, y1 + 1e-9, step)
    outline = ([(x, y0) for x in xs] + [(x, y1) for x in xs]
               + [(x0, y) for y in ys] + [(x1, y) for y in ys])
    zs = np.arange(FLOOR, FLOOR + top + 1e-9, 0.1)
    return np.array([(x, y, z) for x, y in outline for z in zs])


def _scene(divider_height):
    """8 x 4 m room. A divider across it at x = 4 m leaves a 0.6 m gap: desks
    (0.8 m tall) or a partition (2.2 m tall)."""
    occ = np.full((80, 160), -1, np.int16)
    occ[1:79, 1:159] = 0
    occ[[0, 79], :] = 100
    occ[:, [0, 159]] = 100
    occ[6:68, 76:84] = 100                   # divider footprint, gap y in [3.4, 3.95)
    occupancy = OccupancyMap(occ, (0.0, 0.0), RES)
    points = np.vstack([_box_points(0.0, 0.0, 8.0, 4.0, 2.5),
                        _box_points(3.8, 0.3, 4.2, 3.4, divider_height)])
    return occupancy, points


def _rooms(free):
    return len(segment_rooms(free, (0.0, 0.0), RES).room_ids)


def test_furniture_does_not_split_a_room_but_a_partition_does():
    occupancy, desks = _scene(divider_height=0.8)
    navigation_free = occupancy.occupancy == 0
    assert _rooms(navigation_free) == 2          # desks look like a wall with a door
    free, walls = segmentation_free_space(occupancy, desks, FLOOR)
    assert _rooms(free) == 1
    assert not walls[30, 76]                     # desk outline (x = 3.8 m) is not a wall

    occupancy, partition = _scene(divider_height=2.2)
    free, walls = segmentation_free_space(occupancy, partition, FLOOR)
    assert walls[30, 76]                         # partition outline is
    assert _rooms(free) == 2


def test_wall_cells_need_points_in_the_band():
    occupancy = OccupancyMap(np.zeros((20, 20), np.int16), (0.0, 0.0), 0.1)
    pts = np.array([[0.55, 0.55, FLOOR + 2.0], [0.55, 0.55, FLOOR + 2.1],   # wall
                    [1.05, 1.05, FLOOR + 2.0],                              # one stray point
                    [1.55, 1.55, FLOOR + 0.7], [1.55, 1.55, FLOOR + 0.8],   # furniture
                    [-5.0, 0.5, FLOOR + 2.0], [-5.0, 0.5, FLOOR + 2.0]])    # outside the map
    walls = wall_cells(occupancy, pts, FLOOR, WallBand(1.6, 2.3, min_points=2))
    assert walls[5, 5] and walls.sum() == 1


@pytest.mark.parametrize("kind", ["binary", "ascii"])
def test_read_pcd(tmp_path, kind):
    pts = np.array([[1.0, 2.0, 3.0, 9.0], [-1.5, 0.25, 2.5, 7.0]], np.float32)
    header = ("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
              "TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
              f"POINTS 2\nDATA {kind}\n").encode()
    body = (pts.tobytes() if kind == "binary"
            else "\n".join(" ".join(str(v) for v in row) for row in pts).encode())
    (tmp_path / "m.pcd").write_bytes(header + body)
    np.testing.assert_allclose(read_pcd_xyz(tmp_path / "m.pcd"), pts[:, :3])
