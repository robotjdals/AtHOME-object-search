"""Carry reviewed component-membership decisions to a new pipeline version.

A decision is a human judgement on recorded evidence (object bbox and
floor-overlap evidence). It is carried over only if the new version has the
same unresolved object set and every object's numeric evidence changed by at
most ``--tolerance`` (m / m^2) with the same structure; otherwise the script
stops and lists the objects that need a new review.
Source hashes are re-bound to the target layout (ATHOME_HM3D_LAYOUT).
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from athome.data.hm3d.layout import current_layout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from flatten(value[key], f"{prefix}{key}.")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from flatten(item, f"{prefix}{i}.")
    else:
        yield prefix, value


def max_change(old, new):
    a, b = dict(flatten(old)), dict(flatten(new))
    if a.keys() != b.keys():
        return None, "structure"
    worst, where = 0.0, ""
    for key, value in a.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            if value != b[key]:
                return None, key
            continue
        change = abs(value - b[key])
        if change > worst:
            worst, where = change, key
    return worst, where


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True, help="기존 검토 결정")
    parser.add_argument("--old-membership", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()
    layout = current_layout()
    output = layout.decisions
    if output.exists():
        raise SystemExit(f"이미 존재하는 출력: {output}")
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    old = {o["object_id"]: o for o in json.loads(args.old_membership.read_text(encoding="utf-8"))["unresolved_objects"]}
    new = {o["object_id"]: o for o in json.loads(layout.membership.read_text(encoding="utf-8"))["unresolved_objects"]}
    if decisions["source_sha256"]["membership"] != sha(args.old_membership):
        raise SystemExit("결정 파일이 지정한 기존 membership과 연결되어 있지 않습니다.")
    decided = {d["object_id"] for d in decisions["objects"]}
    if not (set(old) == set(new) == decided):
        raise SystemExit(f"미배정 객체 집합이 다릅니다: {sorted(set(old) ^ set(new))}")

    changes, blocked = {}, []
    for oid in sorted(new):
        change, where = max_change(old[oid], new[oid])
        changes[oid] = {"max_numeric_change": change, "at": where}
        if change is None or change > args.tolerance:
            blocked.append((oid, where, change))
    if blocked:
        for item in blocked:
            print("재검토 필요:", item)
        raise SystemExit("근거가 바뀐 객체가 있어 결정을 승계하지 않습니다.")

    result = copy.deepcopy(decisions)
    result["source_sha256"] = {
        "graph": sha(layout.graph), "membership": sha(layout.membership),
        "report": sha(layout.room_report), "selection": sha(layout.stair_selection),
        "floor_plan": sha(layout.floors),
    }
    for item in result["objects"]:
        item["floor_bbox_evidence"] = new[item["object_id"]]["floor_bbox_evidence"]
    result["carried_over"] = {
        "from_decisions": str(args.decisions.resolve()),
        "from_decisions_sha256": sha(args.decisions),
        "from_membership_sha256": sha(args.old_membership),
        "tolerance": args.tolerance, "evidence_changes": changes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    worst = max(changes.items(), key=lambda kv: kv[1]["max_numeric_change"])
    print(f"결정 {len(result['objects'])}개 승계. 최대 근거 변화: {worst[0]} {worst[1]}")
    print("저장:", output)


if __name__ == "__main__":
    main()
