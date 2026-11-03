import numpy as np

from athome.scene_graph.geometric_relations import pair_geometry, support_surface_z


def _desk_points(top=0.75, monitor=True):
    """Desk: legs and sides up to the top, many points on the top; a few
    points of a monitor fused into it up to 1.1 m."""
    rng = np.random.default_rng(0)
    z = [rng.uniform(0.0, top, 400), rng.uniform(top - 0.04, top, 1200)]
    if monitor:
        z.append(rng.uniform(top, 1.1, 80))
    return np.concatenate(z)


def _box(lo_z, hi_z, **extra):
    return {"min": [0.0, 0.0, lo_z], "max": [1.0, 0.6, hi_z], **extra}


def test_surface_is_the_dense_top_layer_not_the_highest_point():
    z = _desk_points()
    assert abs(support_surface_z(z) - 0.75) <= 0.05
    assert abs(support_surface_z(_desk_points(monitor=False)) - 0.75) <= 0.05
    assert support_surface_z(np.array([])) is None


def test_object_on_the_surface_is_level_with_the_desk_under_a_monitor():
    z = _desk_points()
    cup = {"min": [0.4, 0.2, 0.76], "max": [0.5, 0.3, 0.86]}
    assert pair_geometry(cup, _box(0.0, float(z.max())))["delta_z"] > 0.1      # box top only
    desk = _box(0.0, float(z.max()), surface_z=support_surface_z(z))
    assert pair_geometry(cup, desk)["delta_z"] < 0.1
    # Objects on the box top still count (HM3D boxes have no surface_z).
    book = {"min": [0.4, 0.2, 1.12], "max": [0.5, 0.3, 1.3]}
    assert pair_geometry(book, desk)["delta_z"] < 0.1
