"""Draw a scene graph on its map as one image, to check by eye.

    python3 scripts/render_graph_overview.py --robot-config configs/robot/demo.yaml

Map, scene graph (scene_graph.path) and goal geometry come from the robot
config, so the goal poses are the ones the search server would use
(athome.scene_graph.overview). build_scene_graph.py writes the same image;
this redraws it for graphs made elsewhere (toy) or after goal settings
change, without rebuilding the graph. Output: <graph>.overview.png unless
--out.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from athome.config import load_robot_config  # noqa: E402
from athome.scene_graph.overview import describe_problems, write_overview  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--graph", type=Path, help="scene_graph.path 대신 사용")
    parser.add_argument("--out", type=Path, help="기본: <graph>.overview.png")
    args = parser.parse_args()

    config = load_robot_config(args.robot_config)
    graph_path = args.graph or config.graph_path
    if graph_path is None:
        raise SystemExit("robot config에 scene_graph.path가 없으면 --graph 필요")
    out = args.out or graph_path.with_name(graph_path.stem + ".overview.png")
    for line in describe_problems(write_overview(config, graph_path, out)):
        print(line)
    print("저장:", out)


if __name__ == "__main__":
    main()
