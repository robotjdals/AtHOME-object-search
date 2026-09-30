"""2D traversability grid shared by goal generation and A*.

Row index follows +y and column index follows +x from ``origin`` (the corner
of cell (0, 0)), which matches both ROS OccupancyGrid and the HM3D grids.
``free`` marks cells where the robot *center* may be, i.e. obstacles are
already inflated by the robot radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

Cell = Tuple[int, int]


@dataclass(frozen=True)
class GridMap:
    free: np.ndarray
    origin: Tuple[float, float]
    resolution: float

    def __post_init__(self):
        if self.free.ndim != 2 or self.free.dtype != bool:
            raise ValueError("free는 2차원 bool 배열이어야 함")
        if not self.resolution > 0:
            raise ValueError("resolution은 양수여야 함")

    @property
    def shape(self) -> Tuple[int, int]:
        return self.free.shape

    def to_cell(self, x: float, y: float) -> Cell:
        return (
            int(np.floor((y - self.origin[1]) / self.resolution)),
            int(np.floor((x - self.origin[0]) / self.resolution)),
        )

    def to_xy(self, cell: Cell) -> Tuple[float, float]:
        row, col = cell
        return (
            float(self.origin[0] + (col + 0.5) * self.resolution),
            float(self.origin[1] + (row + 0.5) * self.resolution),
        )

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.shape[0] and 0 <= cell[1] < self.shape[1]

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and bool(self.free[cell])

    def cell_centers(self):
        """XY of every free cell, with their row/col indices."""
        rows, cols = np.nonzero(self.free)
        x = self.origin[0] + (cols + 0.5) * self.resolution
        y = self.origin[1] + (rows + 0.5) * self.resolution
        return rows, cols, x, y

    def clearance(self) -> np.ndarray:
        """Distance [m] from each cell center to the nearest non-free cell's
        boundary (0 for non-free cells). Cached; ``free`` is read-only."""
        cached = self.__dict__.get("_clearance")
        if cached is None:
            padded = np.pad(self.free, 1, constant_values=False)
            edt = distance_transform_edt(padded)[1:-1, 1:-1] * self.resolution
            cached = np.where(self.free, edt - self.resolution / 2, 0.0)
            object.__setattr__(self, "_clearance", cached)
        return cached

    def nearest_free(self, cell: Cell, max_distance: float) -> Optional[Cell]:
        """Closest free cell within ``max_distance`` meters, if any."""
        if self.is_free(cell):
            return cell
        r = int(np.ceil(max_distance / self.resolution))
        r0, c0 = max(cell[0] - r, 0), max(cell[1] - r, 0)
        window = self.free[r0:cell[0] + r + 1, c0:cell[1] + r + 1]
        rows, cols = np.nonzero(window)
        if rows.size == 0:
            return None
        rows, cols = rows + r0, cols + c0
        d = np.hypot(rows - cell[0], cols - cell[1])
        i = int(np.argmin(d))
        if d[i] * self.resolution > max_distance:
            return None
        return int(rows[i]), int(cols[i])


def traversable_from_occupancy(
    occupancy: np.ndarray,
    resolution: float,
    inflation_radius: float,
    occupied_threshold: int = 50,
    unknown_as_occupied: bool = True,
) -> np.ndarray:
    """ROS-style occupancy (-1 unknown, 0..100) -> free cells for the robot center."""
    blocked = occupancy >= occupied_threshold
    if unknown_as_occupied:
        blocked |= occupancy < 0
    clearance = distance_transform_edt(~blocked) * resolution
    return clearance > inflation_radius


def traversable_from_costmap(values: np.ndarray) -> np.ndarray:
    """Nav2 costmap as OccupancyGrid values (-1 unknown, 99 inscribed,
    100 lethal) -> free cells for the robot center, as the planner sees them."""
    return (values >= 0) & (values < 99)


@dataclass(frozen=True)
class OccupancyMap:
    """A map_server map as ROS occupancy values (-1 unknown, 0 free,
    100 occupied), rows ordered like GridMap (row 0 = lowest y)."""
    occupancy: np.ndarray
    origin: Tuple[float, float]
    resolution: float


def load_occupancy(yaml_path) -> OccupancyMap:
    """Read a ROS map_server map (yaml + binary pgm)."""
    import yaml

    yaml_path = Path(yaml_path)
    meta = yaml.safe_load(yaml_path.read_text())
    image = _read_pgm(yaml_path.parent / meta["image"])
    if meta.get("negate", 0):
        image = 255 - image
    p = (255.0 - image) / 255.0
    occupancy = np.zeros(image.shape, dtype=np.int16)
    occupancy[p > meta["occupied_thresh"]] = 100
    unknown = (p >= meta["free_thresh"]) & (p <= meta["occupied_thresh"])
    occupancy[unknown] = -1

    origin = meta["origin"]
    if len(origin) > 2 and abs(origin[2]) > 1e-9:
        raise ValueError("회전된 map origin은 지원하지 않음")
    # Image row 0 is the top of the map; grid row 0 is the bottom.
    return OccupancyMap(occupancy[::-1], (float(origin[0]), float(origin[1])),
                        float(meta["resolution"]))


def load_map_server(yaml_path, inflation_radius: float, unknown_as_occupied=True):
    """Load a ROS map_server map (yaml + binary pgm) as an inflated GridMap."""
    m = load_occupancy(yaml_path)
    free = traversable_from_occupancy(
        m.occupancy, m.resolution, inflation_radius,
        unknown_as_occupied=unknown_as_occupied,
    )
    return GridMap(free, m.origin, m.resolution)


def _read_pgm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if not data.startswith(b"P5"):
        raise ValueError(f"{path}: binary PGM(P5)만 지원")
    # Header: magic, width, height, maxval, separated by whitespace/comments.
    fields, pos = [], 2
    while len(fields) < 3:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            pos = data.index(b"\n", pos) + 1
            continue
        end = pos
        while not data[end:end + 1].isspace():
            end += 1
        fields.append(int(data[pos:end]))
        pos = end
    width, height, maxval = fields
    if maxval > 255:
        raise ValueError(f"{path}: 16-bit PGM 미지원")
    pixels = np.frombuffer(data, np.uint8, width * height, pos + 1)
    return pixels.reshape(height, width).astype(np.float64)
