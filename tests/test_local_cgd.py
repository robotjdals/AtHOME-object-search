import pytest

from athome.perception import local_cgd


def _home(tmp_path):
    for name in local_cgd.CLONES:
        (tmp_path / name).mkdir()
    (tmp_path / "athome/semantic_geometry_ws/tools/robot_live").mkdir(parents=True)
    return tmp_path


def test_private_home_links_the_clones_where_the_engine_looks(tmp_path):
    home = local_cgd.local_home(_home(tmp_path))
    for name in local_cgd.CLONES:
        assert (home / "conceptgraph_ws" / name).resolve() == tmp_path / name
    assert local_cgd.local_home(tmp_path) == home       # again: nothing to do


def test_server_command_keeps_the_normal_caches_and_the_tunnel_port(tmp_path):
    cmd, env = local_cgd.server_command(tmp_path, "python", _home(tmp_path))
    assert cmd[0] == "python" and str(local_cgd.PORT) == cmd[-1] == "18200"
    assert env["HF_HOME"] == str(tmp_path / ".cache/huggingface")
    assert env["HOME"] != str(tmp_path)


def test_missing_code_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="CG-D 코드"):
        local_cgd.local_home(tmp_path)
    for name in local_cgd.CLONES:
        (tmp_path / name).mkdir()
    with pytest.raises(FileNotFoundError, match="인지 코드"):
        local_cgd.server_command(tmp_path, "python", tmp_path)
