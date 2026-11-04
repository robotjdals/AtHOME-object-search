import numpy as np

from athome.schemas import ObservedObject
from athome.search.detection_watch import DetectionWatch
from athome.search.matching import LabelMatcher, explain

K = [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]
MATCHER = LabelMatcher(0.15)
judge = lambda t, o: explain(MATCHER, t, o)          # noqa: E731


def camera_at(x):
    """Camera on the map's x axis looking along +y (camera z = map y)."""
    T = np.eye(4)
    T[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    T[:3, 3] = -T[:3, :3] @ np.array([x, 0.0, 0.0])
    return T


def cup(conf=0.5, label="cup", x=0.0, y=1.2):
    return ObservedObject(-1, label, conf, (x, y, 0.0))


def test_misses_count_once_the_object_has_a_place():
    w = DetectionWatch("cup", K, (1280, 720))
    w.observe([], judge, camera_at(0.0))                 # nothing known yet
    assert w.summary()["by_distance"][2]["frames"] == 0
    w.observe([cup()], judge, camera_at(0.0))            # first pass: the anchor
    w.observe([], judge, camera_at(0.0))                 # in view, missed
    s = w.summary()
    row = next(r for r in s["by_distance"] if r["bin"] == "1~1.5 m")
    assert (row["frames"], row["passed"]) == (2, 1) and s["passed"] == 1 and s["frames"] == 3
    assert s["by_angle"][0]["frames"] == 2


def test_object_outside_the_image_is_not_a_miss():
    w = DetectionWatch("cup", K, (1280, 720))
    w.observe([cup()], judge, camera_at(0.0))
    before = sum(r["frames"] for r in w.summary()["by_distance"])
    w.observe([], judge, camera_at(5.0))                 # far to the side: out of view
    assert sum(r["frames"] for r in w.summary()["by_distance"]) == before


def test_current_shows_why_and_labels_at_the_object_show_confusion():
    w = DetectionWatch("towel", K, (1280, 720))
    w.observe([cup(0.1, "towel")], judge, camera_at(0.0))
    assert w.current["reason"] == "confidence" and w.current["distance_m"] == 1.2
    w.observe([cup(0.4, "towel")], judge, camera_at(0.0))
    w.observe([cup(0.37, "shoe")], judge, camera_at(0.0))
    s = w.summary()
    assert s["labels_at_object"][0][0] == "shoe" and s["current"]["reason"] == "none"
    assert not s["confirmed"]                            # one passing frame
    w.observe([cup(0.4, "towel")], judge, camera_at(0.0))
    assert w.summary()["confirmed"]


def test_a_pass_far_from_the_object_counts_as_elsewhere():
    w = DetectionWatch("cup", K, (1280, 720))
    w.observe([cup()], judge, camera_at(0.0))
    w.observe([cup(x=3.0)], judge, camera_at(0.0))
    s = w.summary()
    assert s["passed"] == 1 and s["elsewhere"] == 1


def scene():
    from athome.search.detection_watch import SceneWatch
    return SceneWatch(["cup", "towel"], str.lower, K, (1280, 720))


def test_every_object_gets_its_own_row_without_naming_it():
    w = scene()
    judge_own = lambda t, o: explain(MATCHER, t, o)    # noqa: E731
    for _ in range(3):
        w.observe([cup(), cup(0.5, "towel", x=0.4), cup(0.9, "desk", x=-0.3)], judge_own, camera_at(0.0))
    w.observe([cup(0.5, "towel", x=0.4)], judge_own, camera_at(0.0))          # the cup missed
    s = w.summary()
    rows = {r["name"]: r for r in s["objects"]}
    assert set(rows) == {"cup", "towel"}                                      # not the furniture
    assert rows["towel"]["rate"] == 1.0 and rows["towel"]["passed"] == 4
    assert rows["cup"]["passed"] == 3 and rows["cup"]["in_view"] == 4 and rows["cup"]["rate"] == 0.75


def test_two_of_a_kind_are_two_rows_and_a_flicker_is_no_row():
    w = scene()
    judge_own = lambda t, o: explain(MATCHER, t, o)    # noqa: E731
    for _ in range(2):
        w.observe([cup(), cup(x=1.5, y=2.0)], judge_own, camera_at(0.0))
    w.observe([cup(0.5, "towel", x=-1.0)], judge_own, camera_at(0.0))
    s = w.summary()
    assert [r["name"] for r in s["objects"]] == ["cup", "cup"] and s["flickers"] == 1
