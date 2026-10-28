"""Room segmentation input from the 3D map: walls only, not furniture.

The navigation map marks every obstacle between the floor and about 1 m
(furniture included), so the gaps between desks look like doors and one
room is cut into pieces (distance-transform segmentation is unstable under
clutter: Bormann et al., ICRA 2016, recall 96.9% -> 76.1% with furniture).
As in HOV-SG (Werby et al., RSS 2024; walls from the point cloud, then a
distance field and watershed), walls are taken from the 3D points in a
height band above the furniture and below the ceiling. Segmentation then
runs on the known cells of the navigation map minus these walls.

Rooms bounded by something the band does not see (temporary partitions of a
demo, lower than the band) come from a room map instead: a map_server map on
the navigation map's grid with walls and partitions only, no furniture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import ndimage

from athome.navigation import OccupancyMap


@dataclass(frozen=True)
class WallBand:
    # Height above the floor [m]: above furniture, below the ceiling.
    low: float = 1.6
    high: float = 2.3
    # Points per cell to count as wall (one stray point is noise).
    min_points: int = 2
    # Walls are thickened by this many cells so thin wall lines stay closed.
    dilate_cells: int = 1


def read_pcd_xyz(path) -> np.ndarray:
    """x, y, z of a PCD file (v0.7, ascii or binary; float x y z fields)."""
    path = Path(path)
    data = path.read_bytes()
    header, offset = {}, 0
    while True:
        end = data.index(b"\n", offset)
        line = data[offset:end].decode("ascii").strip()
        offset = end + 1
        if not line or line.startswith("#"):
            continue
        key, *values = line.split()
        header[key.upper()] = values
        if key.upper() == "DATA":
            break
    fields = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
    n = int(header["POINTS"][0])
    kind = header["DATA"][0]
    if not all(f in fields for f in ("x", "y", "z")):
        raise ValueError(f"{path}: x y z 필드 없음")
    if kind == "ascii":
        rows = np.loadtxt(data[offset:].decode("ascii").splitlines(), ndmin=2)
        cols = np.cumsum([0] + counts)
        return np.stack([rows[:, cols[fields.index(f)]] for f in "xyz"], axis=1)
    if kind != "binary":
        raise ValueError(f"{path}: DATA {kind} 미지원 (ascii, binary만)")
    dtype = np.dtype([(f"{f}{i}" if c > 1 else f, f"<{t.lower()}{s}", (c,) if c > 1 else ())
                      for i, (f, s, t, c) in enumerate(zip(fields, sizes, types, counts))])
    points = np.frombuffer(data, dtype=dtype, count=n, offset=offset)
    return np.stack([points[f].astype(np.float64) for f in "xyz"], axis=1)


def wall_cells(occupancy: OccupancyMap, points: np.ndarray, floor_z: float,
               band: WallBand = WallBand()) -> np.ndarray:
    """Cells of the map grid (row 0 = lowest y) with wall points."""
    height = points[:, 2] - floor_z
    p = points[(height >= band.low) & (height <= band.high)]
    rows, cols = occupancy.occupancy.shape
    r = np.floor((p[:, 1] - occupancy.origin[1]) / occupancy.resolution).astype(int)
    c = np.floor((p[:, 0] - occupancy.origin[0]) / occupancy.resolution).astype(int)
    inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    count = np.zeros((rows, cols), dtype=np.int32)
    np.add.at(count, (r[inside], c[inside]), 1)
    return count >= band.min_points


def same_grid(a: OccupancyMap, b: OccupancyMap) -> bool:
    """Same cells in the same frame: shape, origin and resolution."""
    return (a.occupancy.shape == b.occupancy.shape
            and np.allclose(a.origin, b.origin, atol=1e-6)
            and bool(np.isclose(a.resolution, b.resolution)))


def room_map_free_space(room_map: OccupancyMap,
                        navigation: OccupancyMap) -> Tuple[np.ndarray, np.ndarray]:
    """(free space for room segmentation, room-map walls that are free in the
    navigation map). The room map's occupied cells (walls, partitions) bound
    the rooms; its unknown cells are left out, as in segmentation_free_space.
    Walls the navigation map has free were drawn in one map only: the robot
    would plan through a partition the rooms are bounded by."""
    if not same_grid(room_map, navigation):
        raise ValueError(
            f"방 경계 지도와 주행 지도의 격자가 다름: {room_map.occupancy.shape[::-1]} 칸, "
            f"원점 {room_map.origin}, {room_map.resolution} m / "
            f"{navigation.occupancy.shape[::-1]} 칸, 원점 {navigation.origin}, "
            f"{navigation.resolution} m")
    walls = room_map.occupancy >= 50          # occupied, as traversable_from_occupancy
    return (room_map.occupancy >= 0) & ~walls, walls & (navigation.occupancy == 0)


def inside_room_map(room_map: OccupancyMap):
    """(x, y) -> whether the point is on a known cell of the room map: the
    area the scene graph keeps objects from (unknown cells are beyond the
    rooms, build_scene_graph.py)."""
    h, w = room_map.occupancy.shape

    def inside(x: float, y: float) -> bool:
        r = int(np.floor((y - room_map.origin[1]) / room_map.resolution))
        c = int(np.floor((x - room_map.origin[0]) / room_map.resolution))
        return 0 <= r < h and 0 <= c < w and room_map.occupancy[r, c] >= 0

    return inside


def segmentation_free_space(occupancy: OccupancyMap, points: np.ndarray, floor_z: float,
                            band: WallBand = WallBand()) -> Tuple[np.ndarray, np.ndarray]:
    """(free space for room segmentation, wall cells): the known cells of
    the navigation map minus the thickened walls. Furniture is free here; it
    still blocks the robot in the navigation map."""
    walls = wall_cells(occupancy, points, floor_z, band)
    if band.dilate_cells:
        walls = ndimage.binary_dilation(walls, iterations=band.dilate_cells)
    return (occupancy.occupancy >= 0) & ~walls, walls
