"""Target catalog from an unmasked workspace graph (evaluator-only GT).

Instances of each target category (aliases normalized, subtypes listed as
target members included, so an instance can belong to several targets:
a bedside lamp is GT for "bedside lamp" and "lamp") with their bbox, room
and unmasked graph anchor. Never passed to the planner. Instances whose
bottom is more than MAX_BASE_ABOVE_FLOOR_M above the room floor (ceiling or
wall-mounted lights) are outside the search scope and listed separately.
"""
import argparse
import json
from pathlib import Path

from athome.data.hm3d.floors import level_floor_height, multi_level_floor_heights
from athome.data.hm3d.target_masking import normalize_tag, target_terms
from athome.scene_graph.query import MAX_BASE_ABOVE_FLOOR_M, above_search_height, floor_references


def build_catalog(graph, config, scene_id, graph_path, height_scope=True):
    targets = [normalize_tag(t) for t in config["target_categories"]]
    terms = {t: target_terms(config, t) for t in targets}
    instances = {t: [] for t in targets}
    out_of_scope = {t: [] for t in targets}
    floor_z = floor_references(graph)
    # Rooms spanning several levels: the floor of the object's own level
    # (src/athome/data/hm3d/floors.py), not the median over levels.
    multi_level = multi_level_floor_heights(graph)
    for obj in graph["objects"]:
        tag = normalize_tag(obj["semantic_tag"])
        matched = [t for t in targets if tag in terms[t]]
        if not matched:
            continue
        reference = floor_z
        if obj["room_id"] in multi_level:
            reference = {obj["room_id"]: level_floor_height(
                obj["bbox"]["min"][2], multi_level[obj["room_id"]])}
        if height_scope and above_search_height(obj, reference):
            for target in matched:
                out_of_scope[target].append(obj["object_id"])
            continue
        if obj["role"] == "child":
            anchor = {"type": "workspace", "id": obj["parent_id"]}
        elif obj["role"] == "standalone":
            anchor = {"type": "standalone", "id": obj["object_id"]}
        else:  # a target selected as a workspace source
            anchor = {"type": "workspace_source", "id": obj["workspace_id"]}
        for target in matched:
            instances[target].append({
                "object_id": obj["object_id"], "original_category": obj["semantic_tag"],
                "room_id": obj["room_id"], "bbox": obj["bbox"],
                "graph_role": obj["role"], "unmasked_graph_anchor": anchor,
            })
    catalog = {
        "schema_version": "0.1", "scene_id": scene_id,
        "purpose": "target_reference_not_planner_input",
        "source_graph": str(graph_path),
        "coordinate_frame": graph["coordinate_frame"], "length_unit": graph["length_unit"],
        "category_policy": config, "anchor_provenance": "derived_from_unmasked_workspace_graph",
        "targets": [{"target_category": t, "instances": instances[t]} for t in targets],
    }
    if height_scope:
        catalog["height_scope"] = {"max_base_above_floor_m": MAX_BASE_ABOVE_FLOOR_M,
                                   "floor_reference": "median_semantic_floor_center_z",
                                   "excluded_object_ids": out_of_scope}
        if multi_level:  # absent for single-level buildings (earlier files unchanged)
            catalog["height_scope"]["multi_level_room_floor_reference"] = \
                "floor_of_object_level_highest_at_or_below_bottom"
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-height-scope", action="store_true",
                        help="최초 catalog 재현용: 높이 범위 제외를 적용하지 않음")
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    catalog = build_catalog(graph, config, args.scene_id, args.graph.resolve(),
                            height_scope=not args.no_height_scope)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    for t in catalog["targets"]:
        print(f"{t['target_category']}: {len(t['instances'])}")
    if "height_scope" in catalog:
        print("높이 범위 밖:", {k: v for k, v in catalog["height_scope"]["excluded_object_ids"].items() if v})
    print("저장:", args.output)


if __name__ == "__main__":
    main()
