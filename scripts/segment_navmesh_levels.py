"""Automatic floor-level / stairs split of NavMesh islands (no manual review).

Without ``--island`` every island of the NavMesh is segmented and the file
lists them under ``islands`` (schema 0.2); islands without a floor platform
(furniture tops, small fragments) keep no triangles. Which components are
room-level navigable areas is decided afterwards by inspect_component_rooms.py
(floor overlap of at least one robot footprint).

With ``--island`` one island is written in the single-island schema 0.1, the
same fields as the former manual ``stair_triangle_selection.review.json``.
Rule: src/athome/data/hm3d/navmesh_levels.py.
"""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import habitat_sim

from athome.data.hm3d.navmesh_levels import LevelRule, island_geometry, navmesh_components, segment_levels

ROOT = Path(__file__).resolve().parents[1]


def island_record(pf, island, rule, mode="levels"):
    vertices, faces, signature = island_geometry(pf, island)
    result = (navmesh_components(vertices, faces) if mode == "islands" else
              segment_levels(vertices, faces, float(pf.nav_mesh_settings.agent_max_climb), rule))
    record = {
        "geometry_sha256": signature, "island_id": island, "triangle_count": int(len(faces)),
        "excluded_triangle_ids": result["excluded_triangle_ids"],
        "retained_triangle_ids": result["retained_triangle_ids"],
    }
    if "degenerate_triangle_ids" in result:  # zero-area triangles, part of the excluded list
        record["degenerate_triangle_ids"] = result["degenerate_triangle_ids"]
    return record, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navmesh", type=Path, required=True, help="저장소 루트 기준 경로")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--island", type=int, help="생략하면 모든 섬")
    parser.add_argument("--mode", choices=["levels", "islands"], default="levels",
                        help="levels: 계단 분리 규칙(사람 기준 NavMesh), islands: 로봇 단차·경사로 만든 NavMesh의 연결 조각 그대로")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    nav_bytes = (ROOT / args.navmesh).read_bytes()
    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(ROOT / args.navmesh)):
        raise SystemExit("NavMesh 로딩 실패")
    settings = pf.nav_mesh_settings
    rule = LevelRule()
    head = {
        "schema_version": "0.1", "scene_id": args.scene_id, "status": "automatic",
        "coordinate_frame": "athome_z_up", "navmesh_path": str(args.navmesh),
        "navmesh_sha256": hashlib.sha256(nav_bytes).hexdigest(),
    }
    tail = {"selection_method": "robot_navmesh_islands_v1" if args.mode == "islands" else rule.version,
            "rule": None if args.mode == "islands" else asdict(rule),
            "agent_max_climb_m": float(settings.agent_max_climb)}
    if args.island is not None:
        record, result = island_record(pf, args.island, rule, args.mode)
        out = {**head, "geometry_sha256": record.pop("geometry_sha256"), **record, **tail,
               "platforms": result["platforms"], "components": result["components"],
               "floor_assignment_verified": False}
        results = [(args.island, result)]
    else:
        islands, results = [], []
        for island in range(pf.num_islands):
            record, result = island_record(pf, island, rule, args.mode)
            islands.append({**record, "platforms": result["platforms"],
                            "components": result["components"]})
            results.append((island, result))
        out = {**head, "schema_version": "0.2", **tail, "islands": islands,
               "floor_assignment_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for island, result in results:
        print(f"섬 {island}: 플랫폼", [(round(p["height_m"], 3), round(p["area_m2"], 2)) for p in result["platforms"]],
              "영역", [(len(c["triangle_ids"]), round(c["area_m2"], 2)) for c in result["components"]],
              "계단(제외)", len(result["excluded_triangle_ids"]))
    print("저장:", args.output)


if __name__ == "__main__":
    main()
