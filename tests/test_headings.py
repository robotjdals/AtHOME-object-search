import math

from athome.execution import headings
from athome.schemas import Pose2D

HFOV = math.radians(70.6)


def deg(values):
    return [round(math.degrees(v)) for v in values]


def test_small_object_in_front_needs_one_heading_and_no_turn():
    goal = Pose2D(0.0, 0.0, 0.0)
    assert deg(headings.cover(goal, (0.5, -0.1, 0.0), (0.7, 0.1, 0.5), HFOV)) == [0]


def test_wide_desk_close_by_needs_three_headings_in_one_sweep():
    goal = Pose2D(0.0, 0.0, math.pi / 2)        # facing +y, desk 2.6 m wide 0.5 m ahead
    hs = deg(headings.cover(goal, (-1.3, 0.5, 0.0), (1.3, 1.2, 0.75), HFOV))
    assert len(hs) == 3 and hs[1] == 90
    assert abs(hs[0] - hs[2]) == 120                # one sweep, no turning back


def test_desk_seen_from_far_fits_one_view():
    goal = Pose2D(0.0, 0.0, 0.0)
    assert len(headings.cover(goal, (2.0, -0.6, 0.0), (2.6, 0.6, 0.75), HFOV)) == 1


def test_front_and_full():
    goal = Pose2D(0.0, 0.0, 0.0)
    assert deg(headings.front(goal)) == [0, -60, 60]
    assert deg(headings.full(goal)) == [0, 60, 120, 180, -120, -60]
    assert deg(headings.front_first(goal)) == [0, -60, 60, 120, 180, -120]
    assert deg(headings.front_first(goal, 8)) == [0, -45, -90, 45, 90, 135, 180, -135]
