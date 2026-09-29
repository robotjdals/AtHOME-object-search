from athome.data.hm3d.semantic_mesh import read_color_table


def test_color_table_follows_habitat_loader(tmp_path):
    txt = tmp_path / "x.semantic.txt"
    txt.write_text('HM3D Semantic Annotations\n'
                   '1,8FA87E,"wall",4\n'
                   '2,c,"radiator",11\n'
                   '3,8fa87e,"ceiling",9\n', encoding="utf-8")
    shadowed = []
    table = read_color_table(txt, shadowed)
    assert table[0x8FA87E] == (1, "wall", "4")      # first row keeps a repeated color
    assert table[0x00000C] == (2, "radiator", "11")  # short hex parsed as std::stoul does
    assert shadowed == [3]
