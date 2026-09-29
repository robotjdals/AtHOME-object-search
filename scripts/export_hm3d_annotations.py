"""Export HM3DSem annotations of one scene (habitat_native, OBB-derived AABBs).

First step of the scene pipeline (previously run by hand); the output matches
outputs/hm3d/<scene>.annotations.json. Object bboxes are later replaced by
semantic-mesh AABBs (scripts/build_mesh_bbox_annotations.py).
"""
import argparse
import json
from pathlib import Path

from athome.data.hm3d.loader import load_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="<scene>.basis.glb")
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    data = load_scene(args.scene, args.dataset_config)
    data.update({"schema_version": "0.2", "bbox_method": "aabb_from_obb", "up_axis": "y"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"객체 {len(data['objects'])}개, Region {len(data['regions'])}개 → {args.output}")


if __name__ == "__main__":
    main()
