"""Found check when the detector is sure: names at confidence >= trust pass
without CLIP, those between the floor and trust need CLIP rank <= k. Validated
on perception's 406 and 412 frames (CG-D, household targets, as
scripts/compare_clip_margin.py): every combination per room, the combination
chosen on one room applied to the other, an interval of the difference from
the robot's rule resampling stops, and each stop whose result changes.
Needs the CLIP text server.

    python scripts/compare_clip_trust.py --d406 d406.jsonl.gz --d412 d412.jsonl.gz --out summary.json
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

import compare_clip_margin as C
from compare_clip_route import LIDAR, OVERRIDES

E = C.E
FLOOR = 0.4
TRUSTS = (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01)       # 1.01: never without CLIP
RANKS = (3, 5, 10, 20)
BASE = (1.01, 5)                                            # the robot's rule


def stops(name, path, scores, vocab):
    """(target, stop) -> {rule: passing frames on the object, passing frames elsewhere, in view}."""
    E.__dict__.update(E.DATASETS[name])
    ds = E.Dataset(None, LIDAR[name] / "map_points.npz", LIDAR[name] / "traj.txt")
    stop_of = {int(f["frame"]): int(f["stop"]) for f in ds.frames}
    objects = collections.defaultdict(list)
    for r in ds.ref:
        if r["check"] in ("ok", "renamed") and r["category_used"]:
            objects[OVERRIDES.get(name, {}).get(int(r["object_id"]), r["category_used"])].append(int(r["object_id"]))
    dets, _, _ = C.load(path)
    rules = [(t, k) for t in TRUSTS for k in RANKS] + [("dino", 0)]
    out = {}
    for target, ids in objects.items():
        if target not in C.HOUSEHOLD[name]:
            continue
        keys = vocab.target_keys(target)
        rows = collections.defaultdict(lambda: {"in_view": False, "on": collections.Counter(), "off": collections.Counter(), "seen": []})
        for i, frame in dets.items():
            row = rows[(target, stop_of[i])]
            vis = ds.views[i]["vis"]
            here = [o for o in ids if o in vis]
            row["in_view"] |= bool(here)
            hit, miss = set(), set()
            for d in frame:
                if d["label"] in E.BG or d["confidence"] < FLOOR or vocab.canonical(d["label"]) not in keys or not d.get("clip_feature"):
                    continue
                values = scores.of(target, d["clip_feature"])
                if values is None:
                    continue
                rank, conf = values[2], d["confidence"]
                on = any(E.overlaps(dict(label=d["label"], conf=conf, box=d["box"]), vis[x], True) for x in here)
                row["seen"].append((d["label"], round(conf, 2), rank, on, i))
                for rule in rules:
                    if rule == ("dino", 0) or conf >= rule[0] or rank <= rule[1]:
                        (hit if on else miss).add(rule)
            for rule in hit:
                row["on"][rule] += 1
            for rule in miss:
                row["off"][rule] += 1
        out.update(rows)
    return out


def score(rows, rule):
    found = sum(r["in_view"] and r["on"][rule] >= 2 for r in rows.values())
    total = sum(r["in_view"] for r in rows.values())
    false = sum(r["off"][rule] >= 2 for r in rows.values())
    return found, total, false


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d406", type=Path, required=True)
    p.add_argument("--d412", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cfg = C.load_robot_config(C.ROOT / "configs/robot/room406.yaml")
    vocab = C.Vocabulary.load(cfg.target_categories, label_aliases=cfg.label_aliases)
    verify = C.make_matcher(cfg.matcher, vocab.canonical, vocab.target_keys, vocab.categories)
    assert verify.prepare(), verify.errors
    scores = C.Scores(verify)
    data = {"406": stops("406", args.d406, scores, vocab), "412": stops("412", args.d412, scores, vocab)}
    rules = [(t, k) for t in TRUSTS for k in RANKS] + [("dino", 0)]
    table = {}
    print("규칙 (trust, rank): 406 찾음,오탐 | 412 찾음,오탐 | 합계")
    for rule in rules:
        a, b = score(data["406"], rule), score(data["412"], rule)
        table[str(rule)] = dict(d406=a, d412=b)
        print(f"  {str(rule):14s} {a[0]:2d}/{a[1]},{a[2]} | {b[0]:2d}/{b[1]},{b[2]} | {a[0] + b[0]}/{a[1] + b[1]},{a[2] + b[2]}")

    # Chosen on one room: the most found with at most the robot rule's false stops + 0, then applied to the other.
    print("\n한 방에서 고른 규칙을 다른 방에 적용 (찾음 최대, 오탐은 현재 규칙 이하):")
    cross = {}
    for fit, test in (("406", "412"), ("412", "406")):
        limit = score(data[fit], BASE)[2]
        allowed = [r for r in rules if score(data[fit], r)[2] <= limit]
        best = max(allowed, key=lambda r: (score(data[fit], r)[0], -score(data[fit], r)[2], r != BASE))
        ties = [r for r in allowed if score(data[fit], r)[:1] == score(data[fit], best)[:1]]
        cross[fit] = dict(rule=str(best), ties=[str(t) for t in ties], fit=score(data[fit], best), test=score(data[test], best), base=score(data[test], BASE))
        print(f"  {fit}에서 고름 {best} (동률 {len(ties)}개): {fit} {score(data[fit], best)} → {test} {score(data[test], best)}, {test} 현재 규칙 {score(data[test], BASE)}")

    # Difference from the robot's rule, resampling stops.
    candidate = (0.5, 5)
    rows = [r for d in data.values() for r in d.values()]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(rows), (5000, len(rows)))
    for rule in (candidate, (1.01, 10), ("dino", 0)):
        df = np.array([(r["in_view"] and r["on"][rule] >= 2) - (r["in_view"] and r["on"][BASE] >= 2) for r in rows], float)
        dn = np.array([(r["off"][rule] >= 2) - (r["off"][BASE] >= 2) for r in rows], float)
        print(f"  {rule}: 찾음 차이 {int(df.sum())} [{np.percentile(df[idx].sum(1), 2.5):.0f}, {np.percentile(df[idx].sum(1), 97.5):.0f}], "
              f"오탐 차이 {int(dn.sum())} [{np.percentile(dn[idx].sum(1), 2.5):.0f}, {np.percentile(dn[idx].sum(1), 97.5):.0f}]")

    print("\n결과가 달라지는 정차:")
    changes = []
    for name, d in data.items():
        for (target, stop), r in sorted(d.items()):
            for rule, tag in ((candidate, "0.5"), (("dino", 0), "CLIP 없음")):
                gained = r["in_view"] and r["on"][rule] >= 2 and not r["on"][BASE] >= 2
                falsed = r["off"][rule] >= 2 and not r["off"][BASE] >= 2
                if gained or falsed:
                    seen = sorted(r["seen"], key=lambda s: -s[1])[:4]
                    changes.append(dict(room=name, target=target, stop=stop, rule=tag, kind="찾음" if gained else "오탐", seen=seen))
                    print(f"  {name} {target:13s} 정차 {stop:3d} {tag:7s} {'찾음' if gained else '오탐'}: {seen}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(table=table, cross=cross, changes=changes), indent=1, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
