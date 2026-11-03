"""Check the search matcher's gates for the real-time detector: the
confidence floor (matcher.min_confidence) and the CLIP rank
(matcher.verify.max_rank), with the deployed code
(athome.search.matching.ClipRankVerifier, the text server
scripts/serve_clip_text.py) on the detection server's own replies
(scripts/query_detector_server.py: CLIP features of target-labelled detections).

    python scripts/calibrate_clip_verify.py \
        --hm3d outputs/perception_bench/clip/cliphm3d.jsonl.gz \
        --r412 outputs/perception_bench/clip/clip412.jsonl.gz --out outputs/perception_bench/clip/calibration.json

Both gates are swept on the HM3D renders (scripts/evaluate_detectors_hm3d.py
rows: view x command target) and the 412 frames (perception's judging,
scripts/evaluate_detectors_412.py, rows of the command's target categories).
The confidence floor is the detector threshold chosen on HM3D (lowest with
false rate <= 5 %). The rank is not chosen on HM3D: the reconstructed
meshes' textures make CLIP misread true objects (true pillows ranked below
tissue, stone, artwork), so HM3D measures the renders, not the look-alike
check; the rank set earlier on independent data (perception's ProcTHOR set,
run4 replay) is confirmed on the real 412 images instead, against no CLIP
check at the same floor. The 2-frame confirmation of the search is not
modelled (single frames).
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import evaluate_detectors_412 as e412
import evaluate_detectors_hm3d as ehm
from athome.inference.text_encoder import HttpTextEncoder
from athome.scene_graph.query import normalize_category
from athome.scene_graph.vocabulary import Vocabulary
from athome.search.matching import ClipRankVerifier, LabelMatcher

ROOT = Path(__file__).resolve().parents[1]
CONFS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
RANKS = (1, 2, 3, 5, 10, 20, None)              # None: no CLIP check


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        config = json.loads(f.readline())["config"]
        return config, {r["frame"]: r["detections"] for r in map(json.loads, f)}


def passes(d, conf, rank):
    return d["conf"] >= conf and (rank is None or d["rank"] is None or d["rank"] <= rank)


def rates(rows, conf, rank):
    hit = lambda r: any(passes(d, conf, rank) for d in r["dets"])
    pr = [hit(r) for r in rows if r["present"]]
    ab = [hit(r) for r in rows if not r["present"]]
    return float(np.mean(pr)), float(np.mean(ab)), len(pr), len(ab)


def bootstrap(rows, unit, conf, rank, n=2000, seed=0):
    by = defaultdict(list)
    for r in rows:
        by[r[unit]].append(r)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    rec, fp = [], []
    for _ in range(n):
        pick = [r for k in rng.choice(keys, len(keys)) for r in by[k]]
        a, b, _, _ = rates(pick, conf, rank)
        rec.append(a)
        fp.append(b)
    q = lambda a: [round(float(np.percentile(a, p)), 3) for p in (2.5, 97.5)]
    return q(rec), q(fp)


def sweep(rows):
    out = {}
    for rank in RANKS:
        for conf in CONFS:
            rec, fp, npr, nab = rates(rows, conf, rank)
            out[f"{conf}|{rank}"] = dict(min_confidence=conf, max_rank=rank, recall=round(rec, 3),
                                         false_rate=round(fp, 4), present_rows=npr, absent_rows=nab)
    return out


def hm3d_rows(path, rank):
    """evaluate_detectors_hm3d.rows_for, keeping each detection's confidence
    and CLIP rank."""
    gt = json.loads((ROOT / "outputs/perception_bench/hm3d_set/ground_truth.json").read_text(encoding="utf-8"))
    commands = json.loads((ROOT / "outputs/perception_bench/hm3d_set/commands.json").read_text(encoding="utf-8"))
    _, dets = read(path)
    rows, missing = [], 0
    for v in gt["views"]:
        for c in commands[v["frame"]]:
            same = [d for d in dets.get(v["frame"], []) if normalize_category(d["label"]) == c]
            for d in same:
                d["conf"], d["rank"] = d["confidence"], rank(c, d)
                missing += d["rank"] is None
            pos = [i for i in v["instances"] if i["category"] == c and i["positive"]]
            if pos:
                rows.append(dict(scene=v["scene"], present=True, dets=[d for d in same if any(ehm.on(d, i) for i in pos)]))
            else:
                anyinst = [i for i in v["instances"] if i["category"] == c]
                stray = [d for d in same if not any(ehm.on(d, i) for i in anyinst)]
                if stray or not same:
                    rows.append(dict(scene=v["scene"], present=False, dets=stray))
    return rows, missing


def r412_rows(path, rank, map_points, traj):
    """evaluate_detectors_412.judge rows of the command's target categories."""
    ds = e412.Dataset(None, map_points, traj)
    _, raw = read(path)
    targets = json.loads((ROOT / "configs/perception/commands_412.json").read_text(encoding="utf-8"))
    dets, missing = {}, 0
    for frame, ds_ in raw.items():
        i = int("".join(ch for ch in frame if ch.isdigit()))
        keep = []
        for d in ds_:
            area = (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
            if d["label"] in e412.BG or area > e412.MAX_BBOX * e412.W * e412.H:
                continue
            r = rank(d["label"], d) if d["label"] in targets[frame] else None
            missing += d["label"] in targets[frame] and r is None
            keep.append(dict(label=d["label"], conf=d["confidence"], box=d["box"], rank=r))
        dets[i] = keep
    rows, _ = e412.judge(ds, dets)
    stop = {int(f["frame"]): int(f["stop"]) for f in ds.frames}
    cats = {ds.names.canon(t) for ts in targets.values() for t in ts}
    rows = [dict(stop=stop[r["frame"]], present=r["present"], dets=r["dets"])
            for r in rows if r["category"] in cats and not r["excluded"]]
    return rows, missing


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hm3d", type=Path, required=True)
    p.add_argument("--r412", type=Path, required=True)
    p.add_argument("--map-points", type=Path, default=ROOT / "data/perception/run5_lidar_20261101/"
                   "run5_lidar_for_search412_20261101/map_points.npz")
    p.add_argument("--traj", type=Path, default=ROOT / "data/perception/run5_lidar_20261101/"
                   "run5_lidar_for_search412_20261101/traj.txt")
    p.add_argument("--text-server", default="http://127.0.0.1:8100")
    p.add_argument("--min-confidence", type=float, default=0.15, help="detector threshold chosen on HM3D")
    p.add_argument("--max-rank", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    config, _ = read(args.hm3d)
    vocab = Vocabulary.load(ROOT / "configs/data/target_categories.v5.json")
    verifier = ClipRankVerifier(LabelMatcher(), HttpTextEncoder(args.text_server, config["clip_model"], timeout=30.0),
                                vocabulary=vocab.categories, max_rank=1,
                                category=vocab.canonical, target_keys=vocab.target_keys)
    assert verifier.prepare(), verifier.errors
    rank = lambda target, d: verifier.rank(target, SimpleNamespace(clip_feature=d.get("clip_feature") or []))

    hm3d, miss_h = hm3d_rows(args.hm3d, rank)
    r412, miss_r = r412_rows(args.r412, rank, args.map_points, args.traj)
    result = dict(detector=config["detector"], clip_model=config["clip_model"],
                  vocabulary=f"target_categories.v5 ({len(vocab.categories)} names)",
                  target_detections_without_rank=dict(hm3d=miss_h, r412=miss_r),
                  hm3d=dict(sweep=sweep(hm3d)), r412=dict(sweep=sweep(r412)))
    for name, (c, k) in (("deployed", (args.min_confidence, args.max_rank)),
                         ("confidence_only", (args.min_confidence, None))):
        point = dict(min_confidence=c, max_rank=k)
        for key, rows, unit in (("hm3d", hm3d, "scene"), ("r412", r412, "stop")):
            rec, fp, npr, nab = rates(rows, c, k)
            ci_r, ci_f = bootstrap(rows, unit, c, k)
            point[key] = dict(recall=round(rec, 3), recall_ci95=ci_r, false_rate=round(fp, 4),
                              false_rate_ci95=ci_f, present_rows=npr, absent_rows=nab)
        result[name] = point
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("target_detections_without_rank", "deployed", "confidence_only")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
