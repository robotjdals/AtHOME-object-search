import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cgd_fast import crop_box, whole_name_tokens

CLS, SEP, DOT = 101, 102, 1012
# [CLS] cup . tumble ##r . ke ##tch ##up . tooth ##brush . tooth ##paste . trash can . [SEP]
IDS = [CLS, 1, DOT, 2, 3, DOT, 4, 5, 6, DOT, 7, 8, DOT, 7, 9, DOT, 10, 11, DOT, SEP]
SEPARATORS = {CLS, SEP, DOT}


def tokens(*selected):
    return whole_name_tokens(selected, IDS, SEPARATORS)


def test_a_piece_of_a_name_becomes_the_name():
    assert tokens(3) == [2, 3]              # tumble -> tumbler
    assert tokens(4) == [2, 3]              # ##r -> tumbler
    assert tokens(8) == [4, 5, 6]           # ##up -> ketchup
    assert tokens(7, 8) == [4, 5, 6]        # ##tchup -> ketchup


def test_the_position_tells_names_with_the_same_token_apart():
    assert tokens(10) == [7, 8]             # the tooth of toothbrush
    assert tokens(13) == [7, 9]             # the tooth of toothpaste


def test_a_word_of_a_longer_name_becomes_the_name():
    assert tokens(17) == [10, 11]           # can of trash can


def test_a_phrase_with_a_whole_name_stays():
    assert tokens(1) is None
    assert tokens(3, 4) is None
    assert tokens(1, 8) is None             # cup + ##up: the phrase still says cup


def test_pieces_of_two_names_take_the_one_with_more_of_them():
    assert tokens(6, 7, 3) == [4, 5, 6]
    assert tokens(3, 6) == [2, 3]           # as many: the earlier


def test_only_separators_stay():
    assert tokens(0, 2) is None


def test_crop_box_stays_inside_the_image():
    assert crop_box((5, 5, 100, 100), 110, 200) == (0, 0, 110, 120)
    assert crop_box((50, 60, 70, 80), 640, 480) == (30, 40, 90, 100)
