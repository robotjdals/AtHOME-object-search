"""CLIP check of a target-labelled detection: rank among the training target
categories (the robot now: within 5) against the score gap to the best category
(the target within delta of the top). Close categories crowd the ranks, and a
rank does not see how far apart the scores are.

On perception's 406 and 412 frames (detections of query_detector_server.py
--clip-all), judged as scripts/compare_clip_route.py: stops where the reference
object passes in 2 frames, stops where the name passes in 2 frames elsewhere.
Names without a reference object (plate, ...) only show how many detections
each rule keeps. Needs the CLIP text server.

    python scripts/compare_clip_margin.py --d406 d406.jsonl.gz --d412 d412.jsonl.gz --out summary.json
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

import evaluate_detectors_412 as E
from athome.config import load_robot_config
from athome.inference.factory import make_matcher
from athome.scene_graph.vocabulary import Vocabulary
from compare_clip_route import GATE, LIDAR, OVERRIDES, ROOT, load

RANKS = (5, 10, 20)
DELTAS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
ANY_DELTAS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
RULES = (["none"] + [f"rank{r}" for r in RANKS] + [f"delta{d}" for d in DELTAS]
         + [f"any{d}" for d in ANY_DELTAS] + [f"label|any{d}" for d in ANY_DELTAS])
HOUSEHOLD = {"406": ["blanket", "cup", "cutting board", "pillow", "toilet paper", "towel"],
             "412": ["blanket", "bowl", "cup", "hanger", "pillow", "toilet paper"]}


class Scores:
    """Own best score, top score of the other categories and rank, as
    ClipRankVerifier ranks them."""

    def __init__(self, verify):
        self.texts = verify._vocabulary()
        self.names = verify._names
        self.keys = verify._target_keys

    def of(self, target, feature):
        keys = set(self.keys(target))
        scores = self.texts @ np.asarray(feature, float)
        own = [s for n, s in zip(self.names, scores) if n in keys]
        others = np.array([s for n, s in zip(self.names, scores) if n not in keys])
        if not own:
            return None
        best = max(own)
        return float(best), float(others.max()), 1 + int((others > best).sum())


def passes(rule, values, by_label=True):
    """by_label: the detector named the box as the target. "any": whatever its
    name, the target's score decides. "label|any": the robot's rank rule on
    the target's name, or the score gap on any name."""
    best, top, rank = values
    if rule.startswith("label|any"):
        return bool((by_label and rank <= RANKS[0]) or best >= top - float(rule[9:]))
    if rule.startswith("any"):
        return bool(best >= top - float(rule[3:]))
    if not by_label:
        return False
    if rule == "none":
        return True
    if rule.startswith("rank"):
        return rank <= int(rule[4:])
    return bool(best >= top - float(rule[5:]))


def evaluate(name, path, scores, vocab):
    E.__dict__.update(E.DATASETS[name])
    ds = E.Dataset(None, LIDAR[name] / "map_points.npz", LIDAR[name] / "traj.txt")
    stop = {int(f["frame"]): int(f["stop"]) for f in ds.frames}
    objects = collections.defaultdict(list)
    for r in ds.ref:
        if r["check"] in ("ok", "renamed") and r["category_used"]:
            oid = int(r["object_id"])
            objects[OVERRIDES.get(name, {}).get(oid, r["category_used"])].append(oid)
    dets, _, _ = load(path)
    targets = {}
    for t, ids in sorted(objects.items()):
        keys = vocab.target_keys(t)
        pos = {r: collections.defaultdict(list) for r in RULES}
        neg = {r: collections.defaultdict(list) for r in RULES}
        for i, frame in dets.items():
            vis = ds.views[i]["vis"]
            here = [o for o in ids if o in vis]
            hit, false = collections.Counter(), collections.Counter()
            for d in frame:
                if d["confidence"] < GATE or d["label"] in E.BG or not d.get("clip_feature"):
                    continue
                values = scores.of(t, d["clip_feature"])
                if values is None:
                    continue
                by_label = vocab.canonical(d["label"]) in keys
                on = any(E.overlaps(dict(label=d["label"], conf=d["confidence"], box=d["box"]), vis[x], True) for x in here)
                for rule in RULES:
                    if passes(rule, values, by_label):
                        (hit if on else false)[rule] += 1
            for rule in RULES:
                if here:
                    pos[rule][stop[i]].append(hit[rule] > 0)
                neg[rule][stop[i]].append(false[rule] > 0)
        targets[t] = {rule: dict(stops=len(pos[rule]), found=sum(sum(v) >= 2 for v in pos[rule].values()),
                                 false_stops=sum(sum(v) >= 2 for v in neg[rule].values())) for rule in RULES}
    # Every target-labelled detection: what the label rules keep, per name.
    kept = collections.defaultdict(collections.Counter)
    gaps = collections.defaultdict(list)
    for frame in dets.values():
        for d in frame:
            n = vocab.canonical(d["label"])
            if n in vocab.categories and d["confidence"] >= GATE and d.get("clip_feature"):
                values = scores.of(n, d["clip_feature"])
                if values is None:
                    continue
                gaps[n].append(values[1] - values[0])
                for rule in RULES:
                    if not rule.startswith(("any", "label|")):
                        kept[n][rule] += passes(rule, values)
    return dict(targets=targets, kept={n: {k: int(v) for k, v in c.items()} for n, c in kept.items()},
                gap_median={n: round(float(np.median(v)), 3) for n, v in gaps.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d406", type=Path, required=True)
    p.add_argument("--d412", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cfg = load_robot_config(ROOT / "configs/robot/room406.yaml")
    vocab = Vocabulary.load(cfg.target_categories)
    verify = make_matcher(cfg.matcher, vocab.canonical, vocab.target_keys, vocab.categories)
    if not verify.prepare():
        raise SystemExit(f"CLIP 문자 서버 필요: {verify.errors}")
    scores = Scores(verify)
    result = {n: evaluate(n, p_, scores, vocab) for n, p_ in (("406", args.d406), ("412", args.d412))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, r in result.items():
        print(f"\n== {name} 일상 물체 합계")
        for rule in RULES:
            rows = [r["targets"][t][rule] for t in HOUSEHOLD[name] if t in r["targets"]]
            allr = [v[rule] for v in r["targets"].values()]
            print(f"  {rule:12s} 찾은 위치 {sum(x['found'] for x in rows):2d}/{sum(x['stops'] for x in rows)} 오검출 {sum(x['false_stops'] for x in rows):2d}"
                  f"  | 전체 범주 {sum(x['found'] for x in allr):2d}/{sum(x['stops'] for x in allr)} 오검출 {sum(x['false_stops'] for x in allr):2d}")
        print("  목표별 found/오검출:")
        for t in HOUSEHOLD[name]:
            v = r["targets"].get(t)
            if v:
                print(f"    {t:14s} " + "  ".join(f"{k} {v[k]['found']}/{v[k]['stops']} 오{v[k]['false_stops']}" for k in ("none", "rank5", "delta0.03", "any0.02", "label|any0.0", "label|any0.02")))
        print("  이름별 검출 수와 유지 (none, rank5, delta0.03, delta0.05) | 1등과의 차이 중앙값:")
        for n, c in sorted(r["kept"].items(), key=lambda x: -x[1]["none"])[:14]:
            print(f"    {n:14s} {c['none']:4d} {c['rank5']:4d} {c['delta0.03']:4d} {c['delta0.05']:4d} | {r['gap_median'][n]}")


if __name__ == "__main__":
    main()
