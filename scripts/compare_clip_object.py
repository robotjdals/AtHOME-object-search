"""A target found by what CLIP sees in an object-sized box, whatever its name,
added to the robot's rule (named as the target, CLIP rank within 5, 2 frames):
any box with confidence >= 0.15 that is not named as furniture, covers at most
``cap`` of the image and has the target within CLIP rank ``k``, in ``n`` frames
of a stop. A detector may have no word for an object CLIP knows (ketchup, 2026-11-04:
nothing with its name alone, CLIP rank 3).

Object boxes only: scripts/compare_clip_margin.py also counts a box as on the
object when it merely contains it, e.g. the desk under it. Here the box must
fit the object (evaluate_detectors_412.overlaps, strict). Judged on perception's 406 and 412
frames as scripts/compare_clip_route.py. Needs the CLIP text server.

    python scripts/compare_clip_object.py --runs base=<dir> large=<dir> --out summary.json
"""
import argparse
import collections
import json
from pathlib import Path

import evaluate_detectors_412 as E
from athome.config import load_robot_config
from athome.inference.factory import make_matcher
from athome.scene_graph.vocabulary import Vocabulary
from compare_clip_margin import HOUSEHOLD, Scores
from compare_clip_route import GATE, LIDAR, OVERRIDES, ROOT, load

KS, FRAMES, CAPS = (1, 2, 3, 5), (2, 3, 4), (0.05, 0.1, 0.25)
AREA = 1280 * 720
FURNITURE = {"chair", "sofa", "bed", "cabinet", "refrigerator", "oven", "washing machine", "toilet", "bathtub", "sink",
             "rug", "mirror", "television", "wardrobe", "door", "window", "curtain", "bookshelf", "shelf"}


def evaluate(name, path, scores, vocab):
    E.__dict__.update(E.DATASETS[name])
    ds = E.Dataset(None, LIDAR[name] / "map_points.npz", LIDAR[name] / "traj.txt")
    stop = {int(f["frame"]): int(f["stop"]) for f in ds.frames}
    objects = collections.defaultdict(list)
    for r in ds.ref:
        if r["check"] in ("ok", "renamed") and r["category_used"]:
            objects[OVERRIDES.get(name, {}).get(int(r["object_id"]), r["category_used"])].append(int(r["object_id"]))
    # vocabulary_base.txt starts with the 24 Workspace source categories.
    names = (ROOT / "configs/perception/vocabulary_base.txt").read_text(encoding="utf-8").splitlines()
    sources = [n.strip() for n in names if n.strip() and not n.startswith("#")][:24]
    furniture = FURNITURE | {vocab.canonical(c) for c in sources}
    dets, _, _ = load(path)
    out = {}
    for t, ids in sorted(objects.items()):
        keys = set(vocab.target_keys(t))
        # (stop, on the object?) -> frames passing the label rule / each object rule
        label = {True: collections.Counter(), False: collections.Counter()}
        route = {True: collections.defaultdict(collections.Counter), False: collections.defaultdict(collections.Counter)}
        stops = set()
        for i, frame in dets.items():
            vis = ds.views[i]["vis"]
            here = [o for o in ids if o in vis]
            if here:
                stops.add(stop[i])
            seen_label, seen_route = {True: False, False: False}, {True: set(), False: set()}
            for d in frame:
                if d["confidence"] < GATE or d["label"] in E.BG or not d.get("clip_feature"):
                    continue
                values = scores.of(t, d["clip_feature"])
                if values is None:
                    continue
                rank = values[2]
                det = dict(label=d["label"], conf=d["confidence"], box=d["box"])
                named = vocab.canonical(d["label"]) in keys
                if named and rank <= 5:
                    seen_label[any(E.overlaps(det, vis[x], True) for x in here)] = True
                if vocab.canonical(d["label"]) in furniture and not named:
                    continue
                b = d["box"]
                area = (b[2] - b[0]) * (b[3] - b[1]) / AREA
                on = any(E.overlaps(det, vis[x]) for x in here)
                for k in KS:
                    for cap in CAPS:
                        if rank <= k and area <= cap:
                            seen_route[on].add((k, cap))
            for on in (True, False):
                if seen_label[on]:
                    label[on][stop[i]] += 1
                for key in seen_route[on]:
                    route[on][key][stop[i]] += 1
        row = {"now": dict(stops=len(stops), found=sum(label[True][s] >= 2 for s in stops),
                           false_stops=sum(v >= 2 for v in label[False].values()))}
        for k in KS:
            for n in FRAMES:
                for cap in CAPS:
                    pos, neg = route[True][(k, cap)], route[False][(k, cap)]
                    row[f"k{k}_n{n}_a{cap}"] = dict(
                        stops=len(stops),
                        found=sum(label[True][s] >= 2 or pos[s] >= n for s in stops),
                        false_stops=len({s for s, v in label[False].items() if v >= 2} | {s for s, v in neg.items() if v >= n}))
        out[t] = row
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="name=dir with d406.jsonl.gz and d412.jsonl.gz")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cfg = load_robot_config(ROOT / "configs/robot/room406.yaml")
    vocab = Vocabulary.load(cfg.target_categories)
    verify = make_matcher(cfg.matcher, vocab.canonical, vocab.target_keys, vocab.categories)
    if not verify.prepare():
        raise SystemExit(f"CLIP 문자 서버 필요: {verify.errors}")
    scores = Scores(verify)
    result = {}
    for run in args.runs:
        name, directory = run.split("=")
        result[name] = {ds: evaluate(ds, Path(directory) / f"d{ds}.jsonl.gz", scores, vocab) for ds in ("406", "412")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

    def total(r, ds, rule, only=None):
        rows = [v[rule] for t, v in r[ds].items() if only is None or t in only]
        return f"{sum(x['found'] for x in rows):2d}/{sum(x['stops'] for x in rows)},{sum(x['false_stops'] for x in rows):2d}"
    rules = ["now"] + [f"k{k}_n{n}_a{c}" for k in KS for n in FRAMES for c in CAPS]
    per_target = ("now", "k1_n2_a0.1", "k3_n2_a0.1", "k3_n4_a0.1")
    for name, r in result.items():
        print(f"\n== {name}: 찾은/정차,오탐  406 일상 | 406 전체 | 412 일상 | 412 전체")
        for rule in rules:
            print(f"  {rule:14s} {total(r, '406', rule, HOUSEHOLD['406'])} | {total(r, '406', rule)} | "
                  f"{total(r, '412', rule, HOUSEHOLD['412'])} | {total(r, '412', rule)}")
        for ds in ("406", "412"):
            for t in HOUSEHOLD[ds]:
                v = r[ds].get(t)
                if v:
                    print(f"    {ds} {t:14s} " + "  ".join(f"{k} {v[k]['found']}/{v[k]['stops']},{v[k]['false_stops']}" for k in per_target))


if __name__ == "__main__":
    main()
