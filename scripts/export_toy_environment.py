"""Write the toy environment as files: map_server map + workspace graph.

    python3 scripts/export_toy_environment.py --hide remote
    -> outputs/toy/toy_map.{pgm,yaml}, outputs/toy/toy_graph.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from athome.testing import toy_env  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/toy")
    parser.add_argument("--hide", nargs="*", default=[],
                        help="categories removed from the graph (unknown targets)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    occ = toy_env.toy_occupancy()
    # map_server image: 0 occupied, 254 free; top row is the highest y.
    image = np.where(occ >= 50, 0, 254).astype(np.uint8)[::-1]
    rows, cols = image.shape
    (args.out / "toy_map.pgm").write_bytes(
        b"P5\n%d %d\n255\n" % (cols, rows) + image.tobytes())
    (args.out / "toy_map.yaml").write_text(
        "image: toy_map.pgm\n"
        f"resolution: {toy_env.RESOLUTION}\n"
        "origin: [0.0, 0.0, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    (args.out / "toy_graph.json").write_text(
        json.dumps(toy_env.toy_graph(args.hide), indent=2), encoding="utf-8")
    # Perception module format, input of scripts/build_scene_graph.py.
    toy_env.write_toy_static_features(args.out / "static_features.json", args.hide)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
