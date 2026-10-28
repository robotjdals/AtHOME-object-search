"""Draw demo partitions into a map_server map: navigation map + room map.

    python3 scripts/make_demo_maps.py --base <412_test2.yaml> --pcd <map.pcd> \
        --floor-z -0.398 --layout configs/robot/room412_partitions.json \
        --out data/real_run5/demo/412_demo

Writes <out>.yaml/.png (navigation map: the base map with the partitions
drawn, the ``clear`` boxes' map-only walls removed and free cells beyond the
demo area unknown) and <out>_rooms.yaml/.png (room map for
scene_graph.room_map and search.visibility_map: walls and partitions only,
no furniture). Beyond the demo area means outside ``area`` or in its
``exclude`` boxes: objects centered there are left out of the graph. Occupied
cells stay occupied in the navigation map, so planners that take unknown for
free (Nav2 track_unknown_space: false) still meet the same walls. Both maps
keep the base grid, so Static Features built on the base map stay valid
(map.edited_from).

Walls of the room map: cells of the base map that are occupied and either
have wall points in the PCD wall band (athome.scene_graph.wall_map) or no
obstacle points at all (walls drawn into the base map only), plus the
layout's walls. Occupied cells with obstacle points below the band are
furniture and left out.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from athome.navigation import load_occupancy  # noqa: E402
from athome.scene_graph.wall_map import WallBand, read_pcd_xyz, wall_cells  # noqa: E402

OCCUPIED, FREE, UNKNOWN = 0, 254, 205          # map_server trinary pixels
OBSTACLE_BAND = (0.05, 1.6)                    # [m] above the floor: furniture and walls
POINT_TOLERANCE_CELLS = 2                      # drawn-only wall: no point within 10 cm


def box_mask(shape, origin, resolution, x, y):
    """Cells whose centers lie in [x0, x1) x [y0, y1) (row 0 = lowest y)."""
    rows, cols = np.indices(shape)
    cx = origin[0] + (cols + 0.5) * resolution
    cy = origin[1] + (rows + 0.5) * resolution
    return (cx >= x[0]) & (cx < x[1]) & (cy >= y[0]) & (cy < y[1])


def write_map(path: Path, pixels: np.ndarray, base_meta: dict) -> None:
    """``pixels`` rows ordered like the grid (row 0 = lowest y)."""
    Image.fromarray(pixels[::-1].astype(np.uint8), mode="L").save(path.with_suffix(".png"))
    meta = dict(base_meta, image=path.with_suffix(".png").name)
    path.with_suffix(".yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="map_server yaml the objects were built on")
    parser.add_argument("--pcd", type=Path, required=True, help="3D map of the same frame")
    parser.add_argument("--floor-z", type=float, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output stem (no suffix)")
    args = parser.parse_args()

    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    base = load_occupancy(args.base)
    shape, origin, res = base.occupancy.shape, base.origin, base.resolution
    box = lambda item: box_mask(shape, origin, res, item["x"], item["y"])

    points = read_pcd_xyz(args.pcd)
    height = points[:, 2] - args.floor_z
    near = points[(height > OBSTACLE_BAND[0]) & (height < OBSTACLE_BAND[1])]
    r = np.floor((near[:, 1] - origin[1]) / res).astype(int)
    c = np.floor((near[:, 0] - origin[0]) / res).astype(int)
    inside = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    seen = np.zeros(shape, bool)
    seen[r[inside], c[inside]] = True
    seen = ndimage.binary_dilation(seen, iterations=POINT_TOLERANCE_CELLS)

    occupied = base.occupancy >= 50
    drawn = occupied & ~seen                         # walls drawn into the base map only
    cleared = np.zeros(shape, bool)
    for item in layout.get("clear", []):
        cleared |= box(item) & drawn
    walls = np.zeros(shape, bool)
    for item in layout["walls"]:
        walls |= box(item)
    beyond = ~box(layout["area"])
    for item in layout.get("exclude", []):
        beyond |= box(item)

    base_meta = yaml.safe_load(args.base.read_text())
    pixels = np.where(base.occupancy < 0, UNKNOWN, np.where(occupied, OCCUPIED, FREE))
    pixels[cleared] = FREE
    pixels[walls] = OCCUPIED
    pixels[beyond & (pixels == FREE)] = UNKNOWN
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_map(args.out, pixels, base_meta)

    structure = (occupied & (wall_cells(base, points, args.floor_z, WallBand()) | drawn)) & ~cleared
    room = np.where(base.occupancy < 0, UNKNOWN, FREE)
    room[structure | walls] = OCCUPIED
    room[beyond] = UNKNOWN
    rooms_out = args.out.with_name(args.out.name + "_rooms")
    write_map(rooms_out, room, base_meta)

    print(f"{args.out.with_suffix('.yaml')}: 칸막이 {int(walls.sum())}칸 추가, "
          f"지도 전용 벽 {int(cleared.sum())}칸 제거")
    print(f"{rooms_out.with_suffix('.yaml')}: 벽 {int((room == OCCUPIED).sum())}칸 "
          f"(가구 {int((occupied & ~structure & ~walls).sum())}칸 제외)")


if __name__ == "__main__":
    main()
