"""Compute within-room geometry for Workspace source selection."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


# Height layer of support_surface_z [m] (= perception's voxel size).
SURFACE_LAYER_M = 0.05


def support_surface_z(z, layer=SURFACE_LAYER_M):
    """Top of the densest height layer in the upper half of an object's
    points: its main horizontal surface (a desk top returns many LiDAR points
    at one height, as in tabletop segmentation by the dominant horizontal
    plane). None without points."""
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        return None
    mid = (z.min() + z.max()) / 2
    upper = z[z >= mid]
    edges = np.arange(mid, z.max() + layer, layer)
    if edges.size < 2:
        return float(z.max())
    counts, _ = np.histogram(upper, bins=edges)
    return float(min(edges[int(np.argmax(counts)) + 1], z.max()))


# Footprint from an object's points: percentiles (stray points left out), only
# with enough points for them to mean something.
FOOTPRINT_PERCENTILES = (2.0, 98.0)
FOOTPRINT_MIN_POINTS = 50


def point_footprint(points):
    """{"min": [x, y], "max": [x, y]} of the object's points, or None with
    too few. A box fitted to real points can be rotated against the furniture
    (run6 desks: 16-25 degrees), and its axis-aligned box then is up to 2x the
    furniture (2.17 x 2.97 m for a 1.11 x 1.89 m desk). HM3D boxes are per
    object mesh, so the training graphs do not have this."""
    xy = np.asarray(points, dtype=float)[:, :2]
    if len(xy) < FOOTPRINT_MIN_POINTS:
        return None
    lo, hi = np.percentile(xy, FOOTPRINT_PERCENTILES, axis=0)
    return {"min": [float(v) for v in lo], "max": [float(v) for v in hi]}


def pair_geometry(child_bbox, source_bbox):
    lo, hi = child_bbox["min"], child_bbox["max"]
    slo, shi = source_bbox["min"], source_bbox["max"]
    # Support heights: the box top and, for objects built from points, the
    # main horizontal surface (surface_z). Points above the surface (a monitor
    # or partition fused into a desk) raise the box top only: HM3D meshes are
    # per object, so there the box top is the surface.
    tops = [shi[2]] + ([source_bbox["surface_z"]]
                       if source_bbox.get("surface_z") is not None else [])

    gaps = [
        max(slo[k] - hi[k], lo[k] - shi[k], 0.0)
        for k in (0, 1)
    ]
    intersection = math.prod(
        max(0.0, min(hi[k], shi[k]) - max(lo[k], slo[k]))
        for k in (0, 1)
    )
    child_area = (hi[0] - lo[0]) * (hi[1] - lo[1])

    return {
        "distance_xy": math.hypot(*gaps),
        "delta_z": min(abs(lo[2] - top) for top in tops),
        "overlap_xy": (
            intersection / child_area if child_area > 0 else None
        ),
    }


def _union_overlap(child_bbox, source_bboxes):
    """Part of the child's XY box over the union of the sources' boxes."""
    lo, hi = child_bbox["min"], child_bbox["max"]
    clipped = [
        [max(b["min"][k], lo[k]) for k in (0, 1)] + [min(b["max"][k], hi[k]) for k in (0, 1)]
        for b in source_bboxes
    ]
    xs = sorted({lo[0], hi[0], *(c[0] for c in clipped), *(c[2] for c in clipped)})
    ys = sorted({lo[1], hi[1], *(c[1] for c in clipped), *(c[3] for c in clipped)})
    covered = sum(
        (x1 - x0) * (y1 - y0)
        for x0, x1 in zip(xs, xs[1:]) for y0, y1 in zip(ys, ys[1:])
        if any(c[0] <= x0 and x1 <= c[2] and c[1] <= y0 and y1 <= c[3] for c in clipped)
    )
    area = (hi[0] - lo[0]) * (hi[1] - lo[1])
    return covered / area if area > 0 else 0.0


def _share_on(points, bbox):
    """Part of ``points`` over the source's footprint (its box without one)."""
    area = bbox.get("footprint") or {"min": bbox["min"][:2], "max": bbox["max"][:2]}
    xy = np.asarray(points, dtype=float)[:, :2]
    return float(np.all((xy >= area["min"]) & (xy <= area["max"]), axis=1).mean())


def straddling_children(objects, source_ids, height_threshold, overlap_threshold,
                        points=lambda object_id: None):
    """Objects on selected sources that meet at a corner (room_baram: a turned
    monitor whose axis-aligned box lies 0.49 + 0.47 on two desks): over no
    single source by the overlap rule of build_relations, over their union by
    it. Each goes to the source most of its points are over (the monitor:
    0.65 on the desk with its keyboard, 0.19 on the other, checked on site);
    without points to the one its box overlaps most. Real environments only:
    the training graphs keep that rule alone. ``objects``: one room's objects
    with bbox and child_candidate_ids; ``points``: object ID -> its points."""
    by_id = {o["object_id"]: o for o in objects}
    placed = {c for s in source_ids for c in by_id[s]["child_candidate_ids"]}
    found = []
    for obj in objects:
        oid = obj["object_id"]
        if oid in source_ids or oid in placed:
            continue
        under = []
        for sid in source_ids:
            g = pair_geometry(obj["bbox"], by_id[sid]["bbox"])
            if g["overlap_xy"] and g["delta_z"] < height_threshold:
                under.append((g["overlap_xy"], sid))
        if len(under) < 2:
            continue
        union = _union_overlap(obj["bbox"], [by_id[s]["bbox"] for _, s in under])
        if union > overlap_threshold:
            cloud = points(oid)
            has_points = cloud is not None and len(cloud) > 0
            shares = {s: _share_on(cloud, by_id[s]["bbox"]) if has_points else 0.0 for _, s in under}
            best = max(under, key=lambda u: (shares[u[1]], u[0]))[1]
            found.append({"object_id": oid, "source_object_id": best,
                          "over": sorted(shares), "overlap_xy": union,
                          **({"points_share": shares} if has_points else {})})
    return found


def build_relations(data, distance_threshold, height_threshold,
                    overlap_threshold):
    if (
        data.get("coordinate_frame") != "athome_z_up"
        or data.get("up_axis") != "z"
        or data.get("length_unit") != "meter"
    ):
        raise ValueError("AtHOME Z-up, meter 단위 데이터가 필요합니다.")

    values = [distance_threshold, height_threshold, overlap_threshold]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("임계값은 유한한 값이어야 합니다.")
    if distance_threshold < 0 or height_threshold <= 0:
        raise ValueError("거리 기준은 0 이상, 높이 기준은 양수여야 합니다.")
    if not 0 <= overlap_threshold < 1:
        raise ValueError("겹침 비율 기준은 0 이상 1 미만이어야 합니다.")

    groups = defaultdict(list)
    seen = set()
    valid_rooms = {room["room_id"] for room in data["rooms"]}

    for obj in data["objects"]:
        if obj["object_id"] in seen:
            raise ValueError(f"중복 객체: {obj['object_id']}")
        seen.add(obj["object_id"])
        if obj["room_id"] not in valid_rooms:
            raise ValueError(f"소속 Room 없음: {obj['object_id']}")

        bbox = obj["bbox"]
        if bbox is None:
            raise ValueError(f"Bounding Box 없음: {obj['object_id']}")
        lo, hi = bbox["min"], bbox["max"]
        if (
            len(lo) != 3 or len(hi) != 3
            or not all(math.isfinite(v) for v in lo + hi)
            or any(a > b for a, b in zip(lo, hi))
        ):
            raise ValueError(f"잘못된 Bounding Box: {obj['object_id']}")
        groups[obj["room_id"]].append(obj)

    records = []
    for room_id in sorted(groups):
        objects = sorted(groups[room_id], key=lambda obj: obj["object_id"])

        for source in objects:
            bbox = source["bbox"]
            lo, hi = bbox["min"], bbox["max"]
            neighbors = []
            children = []

            for child in objects:
                if child["object_id"] == source["object_id"]:
                    continue

                geometry = pair_geometry(child["bbox"], bbox)
                if geometry["distance_xy"] > distance_threshold:
                    continue

                neighbors.append({
                    "object_id": child["object_id"],
                    **geometry,
                })
                overlap = geometry["overlap_xy"]
                if (
                    overlap is not None
                    and geometry["delta_z"] < height_threshold
                    and overlap > overlap_threshold
                ):
                    children.append(child["object_id"])

            records.append({
                "room_id": room_id,
                "object_id": source["object_id"],
                "top_surface_center": [
                    (lo[0] + hi[0]) / 2,
                    (lo[1] + hi[1]) / 2,
                    hi[2],
                ],
                "neighbors": neighbors,
                "child_candidate_ids": children,
            })

    return {
        "schema_version": "0.1",
        "stage": "geometric_relations",
        "coordinate_frame": "athome_z_up",
        "length_unit": "meter",
        "thresholds": {
            "distance_xy_m": distance_threshold,
            "delta_z_m": height_threshold,
            "overlap_xy": overlap_threshold,
        },
        "objects": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--overlap", type=float, required=True)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("입력과 출력 경로는 달라야 합니다.")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    result = build_relations(data, args.distance, args.height, args.overlap)
    result["source_map"] = str(args.input.resolve())

    payload = json.dumps(
        result, ensure_ascii=False, indent=2, allow_nan=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(args.output)

    records = result["objects"]
    nonempty = [r for r in records if r["child_candidate_ids"]]

    print("=== 기하 관계 계산 결과 ===")
    print("처리한 Object 수:", len(records))
    print("Child 후보가 있는 Object 수:", len(nonempty))
    print("Child 후보 관계 수:",
          sum(len(r["child_candidate_ids"]) for r in records))
    print("저장 위치:", args.output.resolve())
    print("\n=== Child 후보 예시 ===")
    for record in nonempty[:10]:
        print(record["object_id"], "->", record["child_candidate_ids"][:10])


if __name__ == "__main__":
    main()
