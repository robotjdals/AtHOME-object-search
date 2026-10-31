"""Score detectors on the HM3D detection set (scripts/render_hm3d_detection_set.py)
with the planner's target categories (target_categories.v5: a detection
counts for a target only under that category's name, as the search matcher
does).

    # per-command targets: commands per building, dealt over its views
    python scripts/evaluate_detectors_hm3d.py commands --gt outputs/perception_bench/hm3d_set/ground_truth.json \
        --out outputs/perception_bench/hm3d_set/commands.json
    python scripts/evaluate_detectors_hm3d.py evaluate --gt .../ground_truth.json --commands .../commands.json \
        --detections det.jsonl.gz --out outputs/perception_bench/hm3d_<run>.json

Rows are (view, command target): present when an instance of that category
is a positive in the view (>= 1024 px within 3 m). A present row is
detected by a detection of the category on one of those instances (box IoU
>= 0.25, or the instance's box centre inside the detection box: the 412
judging rule); an absent row by a detection of the category that lies on no
instance of it (detections on small or far instances, "ignore", count
neither way). Confidence intervals: 95 % bootstrap over buildings (views of
one building are not independent).
"""
import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from athome.scene_graph.query import normalize_category

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def on(det, inst):
    b, g = det["box"], inst["box"]
    cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
    return box_iou(b, g) >= 0.25 or (b[0] <= cx <= b[2] and b[1] <= cy <= b[3])


def target_names():
    cats = json.loads((ROOT / "configs/data/target_categories.v5.json").read_text(encoding="utf-8"))["categories"]
    return {normalize_category(c) for c in cats}


def cmd_commands(args):
    """Commands per building, as a command lasts one search in one building:
    --per-building commands of 3 targets each, two drawn from the categories
    in view somewhere in the building (so recall can be measured) and one
    from the rest of the building; the building's views are dealt to them in
    turn."""
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    views, every, seen = defaultdict(list), defaultdict(set), defaultdict(set)
    for v in gt["views"]:
        views[v["scene"]].append(v["frame"])
        every[v["scene"]] |= {i["category"] for i in v["instances"] if i["category"]}
        seen[v["scene"]] |= {i["category"] for i in v["instances"] if i["positive"]}
    out = {}
    for scene in sorted(views):
        pool = sorted(seen[scene])
        rng.shuffle(pool)
        commands = []
        for k in range(args.per_building):
            pick = [pool[(2 * k + j) % len(pool)] for j in range(2)]
            rest = sorted(every[scene] - set(pick))
            commands.append(sorted(set(pick)) + rng.sample(rest, min(1, len(rest))))
        for n, frame in enumerate(views[scene]):
            out[frame] = commands[n % len(commands)]
    args.out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} views, {len({tuple(c) for c in out.values()})} commands -> {args.out}")


def rows_for(gt, commands, dets, mode):
    """mode 'command': the command's targets; 'all': every target category in
    the vocabulary of the run (fixed large vocabulary)."""
    targets = target_names()
    rows = []
    for v in gt["views"]:
        ds = [d for d in dets.get(v["frame"], []) if normalize_category(d["label"]) in targets]
        cats = commands[v["frame"]] if mode == "command" else \
            sorted({normalize_category(d["label"]) for d in ds} | {i["category"] for i in v["instances"] if i["positive"]})
        for c in cats:
            pos = [i for i in v["instances"] if i["category"] == c and i["positive"]]
            same = [d for d in ds if normalize_category(d["label"]) == c]
            if pos:
                rows.append(dict(scene=v["scene"], present=True,
                                 confs=[d["confidence"] for d in same if any(on(d, i) for i in pos)]))
            else:
                anyinst = [i for i in v["instances"] if i["category"] == c]
                stray = [d["confidence"] for d in same if not any(on(d, i) for i in anyinst)]
                ignored = bool(same) and not stray
                if not ignored:
                    rows.append(dict(scene=v["scene"], present=False, confs=stray))
    return rows


def rates(rows, t):
    hit = lambda r: any(c >= t for c in r["confs"])
    pr = [hit(r) for r in rows if r["present"]]
    ab = [hit(r) for r in rows if not r["present"]]
    return (float(np.mean(pr)) if pr else None, float(np.mean(ab)) if ab else None, len(pr), len(ab))


def bootstrap(rows, t, n=2000, seed=0):
    by = defaultdict(list)
    for r in rows:
        by[r["scene"]].append(r)
    scenes = sorted(by)
    rng = np.random.default_rng(seed)
    rec, fp = [], []
    for _ in range(n):
        pick = [r for s in rng.choice(scenes, len(scenes)) for r in by[s]]
        a, b, _, _ = rates(pick, t)
        rec.append(a)
        fp.append(b)
    q = lambda a: [round(float(np.percentile([x for x in a if x is not None], p)), 3) for p in (2.5, 97.5)]
    return q(rec), q(fp)


def cmd_evaluate(args):
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    commands = json.loads(args.commands.read_text(encoding="utf-8"))
    dets, timing = {}, []
    with gzip.open(args.detections, "rt", encoding="utf-8") as f:
        config = json.loads(f.readline())["config"]
        for line in f:
            r = json.loads(line)
            dets[r["frame"]] = r["detections"]
            timing.append(r["total_ms"])
    mode = "command" if config.get("per_frame_extra") in (True, "True") else args.rows
    rows = rows_for(gt, commands, dets, mode)
    sweep = {}
    for t in THRESHOLDS:
        rec, fp, npr, nab = rates(rows, t)
        sweep[str(t)] = dict(recall=round(rec, 3), false_rate=round(fp, 4), present_rows=npr, absent_rows=nab)
    result = dict(config=config, rows=mode, sweep=sweep,
                  timing_ms=dict(median=round(float(np.median(timing)), 1),
                                 p95=round(float(np.percentile(timing, 95)), 1)))
    if args.false_cap is not None:
        ok = [t for t in THRESHOLDS if sweep[str(t)]["false_rate"] <= args.false_cap]
        if ok:
            t = min(ok)
            ci_r, ci_f = bootstrap(rows, t)
            result["operating_point"] = dict(threshold=t, false_cap=args.false_cap, **sweep[str(t)],
                                             recall_ci95=ci_r, false_rate_ci95=ci_f)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(config["detector"], config.get("segmenter"), config.get("imgsz"), mode,
          json.dumps(result.get("operating_point", {})), result["timing_ms"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands")
    c.add_argument("--gt", type=Path, required=True)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--per-building", type=int, default=3)
    c.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--gt", type=Path, required=True)
    e.add_argument("--commands", type=Path, required=True)
    e.add_argument("--detections", type=Path, required=True)
    e.add_argument("--rows", choices=["command", "all"], default="command",
                   help="for a fixed-vocabulary run: score the command targets (comparable) or every target")
    e.add_argument("--false-cap", type=float, default=0.05,
                   help="operating point: lowest threshold whose false rate is at most this")
    e.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    {"commands": cmd_commands, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
