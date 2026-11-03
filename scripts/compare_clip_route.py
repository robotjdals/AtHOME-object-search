"""Found check with and without the detector's label, on perception's 406 and
412 search frames (detections from scripts/query_detector_server.py --clip-all):

  label     the robot now: labelled as the target, confidence >= 0.15, CLIP
            ranks the target within 5 (matcher.verify)
  clip<k>   any box with confidence >= 0.15 whose CLIP zero-shot rank of the
            target is within k, whatever its label
  label+clip<k>  either

Per target: frames with the reference object in view that pass on it, stops
where it passes in 2 frames (the found check), and stops where it passes in 2
frames elsewhere. Needs the CLIP text server (scripts/serve_clip_text.py).

    python scripts/compare_clip_route.py --d406 d406.jsonl.gz --d412 d412.jsonl.gz --out summary.json
"""
import argparse
import collections
import gzip
import json
from pathlib import Path

import numpy as np

import evaluate_detectors_412 as E
from athome.config import load_robot_config
from athome.inference.factory import make_matcher
from athome.scene_graph.vocabulary import Vocabulary
from athome.schemas import ObservedObject

ROOT = Path(__file__).resolve().parents[1]
GATE = 0.15
KS = (1, 2, 3)
# 406 objects of the reference table named as the team checked them.
OVERRIDES = {"406": {23: "towel"}}
RUN5 = ROOT / "data/perception/run5_lidar_20261101/run5_lidar_for_search412_20261101"
LIDAR = {"406": ROOT / "data/real_run6/real_run6_406_20261103/lidar_for_search", "412": RUN5}


def load(path):
    out, clip_ms, boxes = {}, [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            r = json.loads(line)
            out[int("".join(c for c in r["frame"] if c.isdigit()))] = r["detections"]
            if r.get("stages_ms"):
                clip_ms.append(r["stages_ms"]["clip"])
            boxes.append(len(r["detections"]))
    return out, clip_ms, boxes


def evaluate(name, path, verify, vocab):
    E.__dict__.update(E.DATASETS[name])
    ds = E.Dataset(None, LIDAR[name] / "map_points.npz", LIDAR[name] / "traj.txt")
    stop = {int(f["frame"]): int(f["stop"]) for f in ds.frames}
    objects = collections.defaultdict(list)
    for r in ds.ref:
        if r["check"] in ("ok", "renamed") and r["category_used"]:
            oid = int(r["object_id"])
            objects[OVERRIDES.get(name, {}).get(oid, r["category_used"])].append(oid)
    dets, clip_ms, boxes = load(path)
    routes = ["label"] + [f"clip{k}" for k in KS] + [f"label+clip{k}" for k in KS]
    out = {}
    for t, ids in sorted(objects.items()):
        keys = vocab.target_keys(t)
        pos = {r: collections.defaultdict(list) for r in routes}
        neg = {r: collections.defaultdict(list) for r in routes}
        for i, frame in dets.items():
            vis = ds.views[i]["vis"]
            here = [o for o in ids if o in vis]
            hit, false = collections.Counter(), collections.Counter()
            for d in frame:
                if d["confidence"] < GATE or d["label"] in E.BG or not d.get("clip_feature"):
                    continue
                o = ObservedObject(object_id=-1, label=d["label"], confidence=d["confidence"],
                                   centroid=(0.0, 0.0, 0.0), clip_feature=d["clip_feature"])
                rank = verify.rank(t, o)
                by_label = vocab.canonical(d["label"]) in keys and (rank is None or rank <= verify.max_rank)
                passed = {"label": by_label}
                for k in KS:
                    by_clip = rank is not None and rank <= k
                    passed[f"clip{k}"] = by_clip
                    passed[f"label+clip{k}"] = by_label or by_clip
                dd = dict(label=d["label"], conf=d["confidence"], box=d["box"])
                on = any(E.overlaps(dd, vis[x], True) for x in here)
                for route, ok in passed.items():
                    if ok:
                        (hit if on else false)[route] += 1
            for route in routes:
                if here:
                    pos[route][stop[i]].append(hit[route] > 0)
                neg[route][stop[i]].append(false[route] > 0)
        out[t] = {}
        for route in routes:
            frames = [x for v in pos[route].values() for x in v]
            out[t][route] = dict(
                frames=len(frames), recall=round(float(np.mean(frames)), 3) if frames else None,
                stops=len(pos[route]), stops_found=sum(sum(v) >= 2 for v in pos[route].values()),
                false_stops=sum(sum(v) >= 2 for v in neg[route].values()),
                false_frames=sum(sum(v) for v in neg[route].values()))
    n_stops = len(set(stop.values()))
    totals = {}
    for route in routes:
        rows = [out[t][route] for t in out if out[t][route]["frames"]]
        totals[route] = dict(stops=sum(r["stops"] for r in rows), stops_found=sum(r["stops_found"] for r in rows),
                             false_stops=sum(r["false_stops"] for r in rows),
                             false_per_target=round(sum(r["false_stops"] for r in rows) / (len(rows) * n_stops), 3))
    return dict(frames=len(dets), stops=n_stops, targets=out, totals=totals,
                clip_ms_median=float(np.median(clip_ms)) if clip_ms else None,
                boxes_median=float(np.median(boxes)), boxes_p95=float(np.percentile(boxes, 95)))


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
    result = {name: evaluate(name, path, verify, vocab) for name, path in (("406", args.d406), ("412", args.d412))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, r in result.items():
        print(f"\n== {name}: 사진 {r['frames']}, 위치 {r['stops']}, 프레임당 상자 중앙 {r['boxes_median']} (p95 {r['boxes_p95']}), CLIP {r['clip_ms_median']} ms")
        print("경로별 합계:", json.dumps(r["totals"], ensure_ascii=False))
        for t, v in r["targets"].items():
            print(f"  {t:18s} " + "  ".join(f"{k} {v[k]['stops_found']}/{v[k]['stops']} 오{v[k]['false_stops']}" for k in ("label", "clip1", "clip2", "clip3")))


if __name__ == "__main__":
    main()
