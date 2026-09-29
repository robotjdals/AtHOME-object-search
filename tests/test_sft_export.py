import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def exporter():
    spec = importlib.util.spec_from_file_location("export_sft_dataset", ROOT / "scripts/export_sft_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_building_split_follows_official_hm3d_split():
    config = json.loads((ROOT / "configs/data/scene_splits.json").read_text())
    split = exporter().scene_split
    assert split(Path("/d/hm3d/train/00123-abcDEF/abcDEF.basis.glb"), config) == "train"
    val = split(Path("/d/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"), config)
    assert val in ("val", "test")
    # minival is a subset of val: same building, same split
    assert split(Path("/d/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"), config) == val
    with pytest.raises(ValueError):
        split(Path("/d/other/scene.glb"), config)
