import importlib.util
from pathlib import Path

import numpy as np
import pytest

from athome.data.hm3d.coordinates import convert_bbox
from athome.data.hm3d.mesh_bbox import mesh_bbox


def square(x0, y0, size, z=0.):
    a, b = [x0, y0, z], [x0 + size, y0, z]
    c, d = [x0 + size, y0 + size, z], [x0, y0 + size, z]
    return [[a, b, c], [a, c, d]]


def test_speck_far_away_is_dropped():
    tri = np.array(square(0, 0, 1.) + square(5, 5, 0.1))    # 1% area, 5 m away
    box, audit = mesh_bbox(tri, "cabinet")
    assert box["max"][:2] == [1., 1.]
    assert audit["dropped_speck_area_share"] == pytest.approx(0.01 / 1.01)
    assert audit["dropped_detached_area_share"] == 0


def test_nearby_part_of_rigid_object_is_kept():
    seat = square(0, 0, 1.)
    back = square(0, 0, 0.8, z=0.2)          # 39% area, 0.2 m above the seat
    box, audit = mesh_bbox(np.array(seat + back), "chair")
    assert box["max"][2] == pytest.approx(0.2)
    assert audit["kept_components"] == 2


def test_detached_piece_of_rigid_object_is_dropped():
    window = square(0, 0, 1.)
    bleed = square(2.5, 0, 0.6)              # 26% area, 1.5 m away
    box, audit = mesh_bbox(np.array(window + bleed), "window")
    assert box["max"][0] == 1.
    assert audit["dropped_detached_area_share"] == pytest.approx(0.36 / 1.36)


def test_structural_surface_keeps_occluded_pieces():
    floor = square(0, 0, 1.) + square(2.5, 0, 0.6)
    box, audit = mesh_bbox(np.array(floor), "floor")
    assert box["max"][0] == pytest.approx(3.1)
    assert audit["structural"] and audit["kept_components"] == 2


def test_touching_parts_are_one_component():
    _, audit = mesh_bbox(np.array(square(0, 0, 1.) + square(1.05, 0, 1.)), "table")
    assert audit["components"] == 1


def test_habitat_roundtrip_is_exact():
    path = Path(__file__).resolve().parents[1] / "scripts/build_mesh_bbox_annotations.py"
    spec = importlib.util.spec_from_file_location("build_mesh_bbox_annotations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    box, _ = mesh_bbox(np.array(square(-1.3, 18.1, 0.7, z=3.2) + square(-1.3, 18.1, 0.7, z=3.25)),
                       "pillow")
    back = convert_bbox(module.athome_to_habitat_bbox(box))
    assert back["min"] == box["min"] and back["max"] == box["max"]
