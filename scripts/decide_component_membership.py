"""Automatic A/B membership of objects in rooms spanning components.

Replaces the manual decisions file (component_membership.decisions.review.json)
with a rule applied to the recorded evidence of prepare_component_membership.py:
an object is assigned to component c only if
1. its XY bbox overlaps the floor of c and of no other component, and
2. its bottom is not more than ``--below-floor-tolerance`` below that floor
   (objects starting lower belong to the stairwell or another level).
If several components stand on the same floor object (a room shared by
disconnected NavMesh parts), the floor cannot decide; then the component with
the unique nearest navigable surface is used (``navmesh_xy_distance_m``,
multi-island reports), as Habitat ObjectNav reaches an object from the island
of its nearest navigable point.
Otherwise it is deferred (kept out of both components), as in the manual
review. The rule reproduces all 25 manual decisions of wcojb4TFT35; the data
admit tolerances between 0.03 and 0.2 m, the default is half the NavMesh step
height (0.1 m). Layout: ATHOME_HM3D_LAYOUT.
"""
import argparse
import hashlib
import json

from athome.data.hm3d.layout import current_layout

RULE = "floor_overlap_unique_and_bottom_not_below_floor_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide(obj, tolerance):
    by_comp = {}
    for ev in obj["floor_bbox_evidence"]:
        best = by_comp.get(ev["component"])
        if best is None or ev["bbox_xy_overlap_m2"] > best["bbox_xy_overlap_m2"]:
            by_comp[ev["component"]] = ev
    overlapping = [c for c, ev in by_comp.items() if ev["bbox_xy_overlap_m2"] > 0]
    reason = "unique_floor_overlap_at_floor_level_or_above"
    if len(overlapping) > 1 and "navmesh_xy_distance_m" in obj:
        floors = [{e["floor_object_id"] for e in obj["floor_bbox_evidence"]
                   if e["component"] == c and e["bbox_xy_overlap_m2"] > 0} for c in overlapping]
        if all(f == floors[0] for f in floors):
            # The floor is shared, so it cannot tell the components apart:
            # nearest navigable surface (unique) decides.
            distance = obj["navmesh_xy_distance_m"]
            ranked = sorted(overlapping, key=lambda c: distance[c])
            if distance[ranked[0]] < distance[ranked[1]]:
                overlapping = ranked[:1]
                reason = "shared_floor_nearest_navigable_surface"
    if len(overlapping) != 1:
        reason = "no_component_floor_overlap" if not overlapping else "overlaps_several_component_floors"
        return None, None, reason
    comp = overlapping[0]
    ev = by_comp[comp]
    if ev["object_bottom_minus_floor_center_m"] < -tolerance:
        return None, None, "starts_below_component_floor"
    return comp, ev["floor_object_id"], reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--below-floor-tolerance", type=float, default=0.1)
    parser.add_argument("--compare", help="비교할 기존(수동) 결정 파일")
    args = parser.parse_args()
    layout = current_layout()
    if layout.decisions.exists():
        raise SystemExit(f"이미 존재하는 출력: {layout.decisions}")
    membership = json.loads(layout.membership.read_text(encoding="utf-8"))
    objects, counts = [], {}
    for obj in membership["unresolved_objects"]:
        comp, floor_id, reason = decide(obj, args.below_floor_tolerance)
        status = "proposed_assignment" if comp else "deferred"
        key = f"proposed_{comp}" if comp else "deferred"
        counts[key] = counts.get(key, 0) + 1
        objects.append({
            "object_id": obj["object_id"], "source_room_id": obj["room_id"],
            "status": status, "proposed_component": comp,
            "reference_floor_object_id": floor_id, "reason": reason,
            "floor_bbox_evidence": obj["floor_bbox_evidence"], "assignment_verified": False})
    result = {
        "schema_version": "1.0", "scene_id": layout.scene_id,
        "status": "membership_decisions_automatic", "coordinate_frame": "athome_z_up",
        "source_sha256": {"graph": sha(layout.graph), "membership": sha(layout.membership),
                          "report": sha(layout.room_report), "selection": sha(layout.stair_selection),
                          "floor_plan": sha(layout.floors)},
        "review_method": RULE, "below_floor_tolerance_m": args.below_floor_tolerance,
        "scope": "current_unresolved_objects_only", "room_assignment_verified": False,
        "navigation_accessibility_verified": False, "counts": counts, "objects": objects,
    }
    with layout.decisions.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("결과:", counts, "저장:", layout.decisions)
    if args.compare:
        manual = {o["object_id"]: o.get("proposed_component") for o in
                  json.load(open(args.compare, encoding="utf-8"))["objects"]}
        auto = {o["object_id"]: o["proposed_component"] for o in objects}
        diff = {k: (manual.get(k), auto.get(k)) for k in set(manual) | set(auto)
                if manual.get(k) != auto.get(k)}
        print("수동 결정과 차이:", diff or "없음")


if __name__ == "__main__":
    main()
