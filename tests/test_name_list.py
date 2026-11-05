from athome.perception.name_list import MAX_NAME, NameList


def test_starts_from_the_default_and_keeps_changes(tmp_path):
    path = tmp_path / "sub" / "list.json"
    names = NameList(path, ["Cup", "cup", "book"])
    assert names.names == ["cup", "book"] and names.is_default and not path.exists()
    assert names.add("  Computer   Mouse ") == ""
    names.remove("BOOK")
    again = NameList(path, ["Cup", "book"])
    assert again.names == ["cup", "computer mouse"] and not again.is_default


def test_adding_twice_or_nothing(tmp_path):
    names = NameList(tmp_path / "list.json", ["cup"])
    assert names.add("cup") == "" and names.names == ["cup"]
    assert names.add("   ") and names.add("x" * (MAX_NAME + 1))
    assert names.names == ["cup"]


def test_reset_returns_to_the_default(tmp_path):
    path = tmp_path / "list.json"
    names = NameList(path, ["cup"])
    names.add("mouse")
    names.reset()
    assert names.is_default and not path.exists()
    assert NameList(path, ["cup"]).names == ["cup"]


def test_removing_an_unknown_name_changes_nothing(tmp_path):
    names = NameList(tmp_path / "list.json", ["cup"])
    names.remove("mouse")
    assert names.names == ["cup"]
