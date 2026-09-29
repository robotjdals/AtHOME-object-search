"""Replace annotation bboxes with semantic-mesh AABBs (new versioned file).

Input: the existing habitat_native annotations (IDs, categories, regions).
Output: the same objects with ``bbox`` from src/athome/data/hm3d/mesh_bbox.py,
converted to habitat_native so the unchanged converter/coordinates steps
apply. Objects without mesh triangles get ``bbox: null`` (the converter
excludes them as before); so do objects whose mesh box has no volume (a flat,
axis-aligned fragment such as a single annotated triangle), which is not a
physical object. The audit lists per-object changes, objects kept as
multi-part, and objects whose detached pieces were dropped (see
scripts/review_mesh_bboxes.py for a 3D view of both lists).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from athome.data.hm3d.coordinates import convert_bbox
from athome.data.hm3d.mesh_bbox import MeshBboxRule, mesh_bbox, rule_dict
from athome.data.hm3d.semantic_mesh import instance_triangles


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def athome_to_habitat_bbox(box):
    """Exact inverse of coordinates.convert_bbox for AABBs."""
    lo, hi = box["min"], box["max"]
    new_lo = [lo[0], lo[2], -hi[1]]
    new_hi = [hi[0], hi[2], -lo[1]]
    return {"center": [(a + b) / 2 for a, b in zip(new_lo, new_hi)],
            "size": [b - a for a, b in zip(new_lo, new_hi)],
            "min": new_lo, "max": new_hi}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.audit):
        if path.exists():
            raise RuntimeError(f"이미 존재하는 출력: {path}")
    source = json.loads(args.annotations.read_text(encoding="utf-8"))
    if source.get("coordinate_frame") != "habitat_native" or source.get("up_axis") != "y":
        raise RuntimeError("habitat_native Y-up annotation만 입력할 수 있습니다.")
    scene = Path(source["scene_path"])
    glb = scene.with_name(scene.name.replace(".basis.glb", ".semantic.glb"))
    txt = glb.with_suffix(".txt")
    triangles, stats = instance_triangles(glb, txt)
    if stats["mixed_color"]:
        raise RuntimeError("여러 ID가 섞인 삼각형이 있습니다.")
    rule = MeshBboxRule()

    objects, rows = [], []
    for obj in source["objects"]:
        sid = int(obj["object_id"].rsplit("_", 1)[1])
        old = obj["bbox"]
        row = {"object_id": obj["object_id"], "category": obj["category"],
               "region_id": obj["region_id"], "had_obb_bbox": old is not None}
        new = None
        if sid in triangles:
            box, audit = mesh_bbox(triangles[sid], obj["category"] or "", rule)
            row.update(audit)
            if not all(hi > lo for lo, hi in zip(box["min"], box["max"])):
                row["degenerate_bbox"] = True
                rows.append(row)
                objects.append({**obj, "bbox": None})
                continue
            new = athome_to_habitat_bbox(box)
            # Round trip through the pipeline's own converter must be exact.
            back = convert_bbox(new)
            if not (np.allclose(back["min"], box["min"], atol=1e-12)
                    and np.allclose(back["max"], box["max"], atol=1e-12)):
                raise RuntimeError(f"{obj['object_id']}: 좌표 왕복 불일치")
            if old is not None:
                old_z = convert_bbox(old)
                row["obb_bbox_missed_mesh_m"] = max(
                    float(np.max(np.subtract(old_z["min"], box["min"]))),
                    float(np.max(np.subtract(box["max"], old_z["max"]))), 0.0)
                row["obb_bbox_excess_m"] = max(
                    float(np.max(np.subtract(box["min"], old_z["min"]))),
                    float(np.max(np.subtract(old_z["max"], box["max"]))), 0.0)
        else:
            row["triangles"] = 0
        rows.append(row)
        objects.append({**obj, "bbox": new})

    result = {**source, "bbox_method": rule.version, "bbox_rule": rule_dict(rule),
              "objects": objects,
              "bbox_source": {"semantic_glb": str(glb), "semantic_glb_sha256": sha(glb),
                              "semantic_txt": str(txt), "semantic_txt_sha256": sha(txt),
                              "source_annotations": str(args.annotations.resolve()),
                              "source_annotations_sha256": sha(args.annotations)}}
    multi_part = [r["object_id"] for r in rows
                  if r.get("kept_components", 0) > 1 and r["extent_beyond_main_component_m"] > 0.10]
    detached = [r["object_id"] for r in rows if r.get("dropped_detached_area_share", 0) > 0]
    audit = {
        "status": "review", "rule": rule_dict(rule), "mesh_stats": stats,
        "summary": {
            "objects": len(rows),
            "with_mesh": sum(r["triangles"] > 0 for r in rows),
            "gained_bbox": sum(r["triangles"] > 0 and not r["had_obb_bbox"] for r in rows),
            "lost_bbox": sum(r["triangles"] == 0 and r["had_obb_bbox"] for r in rows),
            "obb_missed_mesh_gt_0p05m": sum(r.get("obb_bbox_missed_mesh_m", 0) > 0.05 for r in rows),
            "speck_dropped_objects": sum(r.get("dropped_speck_area_share", 0) > 0 for r in rows),
            "multi_part_kept": len(multi_part),
            "detached_dropped": len(detached),
            "degenerate_dropped": sum(bool(r.get("degenerate_bbox")) for r in rows),
            "shadowed_color_ids": len(stats["shadowed_ids"]),
        },
        "multi_part_ids": multi_part,
        "detached_dropped_ids": detached,
        "objects": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with args.audit.open("x", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print("요약:", audit["summary"])
    print("다부품 유지:", multi_part)
    print("분리 조각 제거:", detached)


if __name__ == "__main__":
    main()
