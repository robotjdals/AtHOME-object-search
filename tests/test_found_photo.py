import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from athome.execution.found_photo import BOX, draw_found  # noqa: E402


def _jpeg(width=640, height=360):
    return cv2.imencode(".jpg", np.full((height, width, 3), 128, np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes()


def _decoded(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _is_box(pixel):
    return np.abs(pixel.astype(int) - np.array(BOX)).max() < 40


def test_box_is_drawn_where_the_object_is():
    img = _decoded(draw_found(_jpeg(), "cup: cup 0.70", (200, 150, 400, 300), "cup 0.70"))
    assert img.shape == (360, 640, 3)
    assert _is_box(img[225, 200]) and _is_box(img[225, 400])     # left and right edges
    assert _is_box(img[300, 300])                                # bottom edge
    assert not _is_box(img[225, 300])                            # the object stays visible


def test_no_box_without_a_rectangle():
    img = _decoded(draw_found(_jpeg(), "cup: cup 0.70", None, "cup 0.70"))
    assert not _is_box(img[225, 300])
    assert img[10, 630].max() < 40                               # the header line is there


def test_box_at_the_top_keeps_its_label_below_the_header():
    img = _decoded(draw_found(_jpeg(), "cup", (100, 0, 300, 200), "cup 0.70"))
    assert _is_box(img[46, 103])                                 # label background under the 44 px header


def test_not_an_image():
    assert draw_found(b"not a jpeg", "cup", None, "cup") is None
