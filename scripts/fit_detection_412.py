"""Detection model of the search simulator fitted to the 412 robot records.

Input: the perception team's measurement of 2026-10-30 (search412_dataset,
records/<condition>/object_views.csv and observations.csv): 13 robot stops of
run5, about 9 frames per stop, CG-D detections judged against 27 reference
objects. A stop is a viewpoint; its outcome for an object is the majority of
its frames (frames of a stop show almost the same scene).

Fitted (per condition, for every reference object and for the objects of the
275 target categories):
- viewpoint consistency: share of (object, stop) pairs detected in all or no
  frames, and detection in the next frame of the same stop after a miss;
- a Beta-Bernoulli model: each object has a detectability q ~ Beta(alpha,
  beta); each viewpoint of it is detected with probability q, the same
  outcome at every look from that viewpoint. Mean mu = P(detected) and
  intra-object correlation rho from P(B detected | A detected) = mu + rho (1 -
  mu) over pairs of distinct stops of an object; alpha = mu (1/rho - 1), beta
  = (1 - mu)(1/rho - 1);
- distance within an object: mean distance of its missed minus its detected
  stops (the pooled distance trend mixes in which objects are far);
- false detections per stop: a reference category reported at a stop where
  no object of it is visible (majority of frames, and any frame), overall
  and per category (most categories have none, a few many).
Confidence intervals: bootstrap over objects (stops of an object resampled
together) and, for false detections, over stops.

    python scripts/fit_detection_412.py data/perception/search412_20261030_zip/search412_dataset_20261030 \\
        --output outputs/perception_412/detection_model.json --config configs/perception/detection_412.json

``--config`` writes the simulator parameters (athome.symbolic.ViewpointDetectionModel) of the
search condition (target names queried) for the target-category objects.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {"no_targets": "map vocabulary only", "with_targets": "15 target names queried (target threshold 0.4)"}


def read(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def viewpoints(views):
    """(object, stop) -> outcome and distance, with each object's category."""
    frames = collections.defaultdict(list)
    for r in views:
        frames[(r["object_id"], r["stop"])].append(r)
    out = []
    for (obj, stop), rows in frames.items():
        rows.sort(key=lambda r: int(r["frame"]))
        hits = [int(r["detected"]) for r in rows]
        out.append({"object": obj, "stop": stop, "category": rows[0]["category"],
                    "distance_m": float(np.median([float(r["distance_m"]) for r in rows])),
                    "detected": sum(hits) / len(hits) >= 0.5, "hits": hits})
    return out


def beta_bernoulli(vps):
    by_object = collections.defaultdict(list)
    for v in vps:
        by_object[v["object"]].append(v["detected"])
    mu = float(np.mean([v["detected"] for v in vps]))
    after_hit, after_miss = [], []
    for outcomes in by_object.values():
        for i, a in enumerate(outcomes):
            for j, b in enumerate(outcomes):
                if i != j:
                    (after_hit if a else after_miss).append(b)
    p_hit = float(np.mean(after_hit)) if after_hit else float("nan")
    p_miss = float(np.mean(after_miss)) if after_miss else float("nan")
    rho = (p_hit - mu) / (1 - mu) if 0 < mu < 1 else float("nan")
    return {"mu": mu, "rho": rho, "p_detect_after_hit_elsewhere": p_hit, "p_detect_after_miss_elsewhere": p_miss}


def consistency(vps):
    all_or_none = np.mean([min(v["hits"]) == max(v["hits"]) for v in vps])
    again = [b for v in vps for a, b in zip(v["hits"], v["hits"][1:]) if not a]
    return {"viewpoints_all_or_none": float(all_or_none),
            "next_frame_same_stop_after_miss": float(np.mean(again)), "misses_followed": len(again)}


def distance_within_objects(vps):
    by_object = collections.defaultdict(list)
    for v in vps:
        by_object[v["object"]].append(v)
    diffs = []
    for vs in by_object.values():
        hit = [v["distance_m"] for v in vs if v["detected"]]
        miss = [v["distance_m"] for v in vs if not v["detected"]]
        if hit and miss:
            diffs.append(np.mean(miss) - np.mean(hit))
    return {"objects_with_both_outcomes": len(diffs), "mean_missed_minus_detected_m": float(np.mean(diffs)),
            "objects_missed_farther": int(sum(d > 0 for d in diffs))}


def bootstrap_objects(vps, n, rng):
    objects = sorted({v["object"] for v in vps})
    by_object = collections.defaultdict(list)
    for v in vps:
        by_object[v["object"]].append(v)
    draws = []
    for _ in range(n):
        sample = []
        for k, obj in enumerate(rng.choice(objects, len(objects))):
            sample += [{**v, "object": f"{obj}#{k}"} for v in by_object[obj]]
        draws.append(beta_bernoulli(sample))
    return {key: [float(np.nanpercentile([d[key] for d in draws], q)) for q in (2.5, 97.5)]
            for key in ("mu", "rho", "p_detect_after_hit_elsewhere", "p_detect_after_miss_elsewhere")}


def false_detections(observations, n, rng):
    """Reference categories at stops where none of them is visible."""
    cells = collections.defaultdict(list)
    for r in observations:
        if r["실제로 있었나"] == "0" and r["기준표에 있는 범주"] == "1" and r["판정 제외(불확실 물체 위 검출)"] != "1":
            cells[(r["탐색 위치"], r["물체 범주"])].append(int(r["검출됐나"]))
    stops = sorted({s for s, _ in cells})
    majority = {k: np.mean(v) >= 0.5 for k, v in cells.items()}
    anyframe = {k: max(v) == 1 for k, v in cells.items()}

    def rate(table, chosen):
        values = [table[k] for s in chosen for k in table if k[0] == s]
        return float(np.mean(values))
    by_category = collections.defaultdict(list)
    for (stop, category), hit in majority.items():
        by_category[category].append(hit)
    out = {"stop_category_cells": len(cells),
           "per_category_majority": {c: float(np.mean(v)) for c, v in sorted(by_category.items())}}
    for name, table in (("per_stop_majority", majority), ("per_stop_any_frame", anyframe)):
        boot = [rate(table, rng.choice(stops, len(stops))) for _ in range(n)]
        out[name] = {"rate": rate(table, stops), "ci95": [float(np.percentile(boot, q)) for q in (2.5, 97.5)]}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="search412_dataset_20261030 폴더")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--config", type=Path, help="시뮬레이터 검출 모델 설정(JSON)으로도 저장")
    args = parser.parse_args()
    for out in (args.output, args.config):
        if out is not None and out.exists():
            raise SystemExit(f"이미 존재하는 출력: {out}")
    targets = set(json.loads((ROOT / "configs/data/target_categories.v5.json").read_text(encoding="utf-8"))["categories"])
    rng = np.random.default_rng(0)
    result = {"source": str(args.dataset.name), "viewpoint": "majority of the frames of a robot stop", "conditions": {}}
    for cond, label in CONDITIONS.items():
        views = read(args.dataset / f"records/{cond}/object_views.csv")
        vps = viewpoints(views)
        entry = {"description": label}
        for subset, chosen in (("all_reference_objects", vps),
                               ("target_category_objects", [v for v in vps if v["category"] in targets])):
            fit = beta_bernoulli(chosen)
            scale = 1 / fit["rho"] - 1 if fit["rho"] > 0 else float("nan")
            entry[subset] = {
                "objects": len({v["object"] for v in chosen}), "viewpoints": len(chosen),
                "categories": sorted({v["category"] for v in chosen}),
                **fit, "ci95": bootstrap_objects(chosen, args.bootstrap, rng),
                "beta_alpha": fit["mu"] * scale, "beta_beta": (1 - fit["mu"]) * scale,
                "consistency": consistency(chosen), "distance": distance_within_objects(chosen)}
        entry["false_detections"] = false_detections(read(args.dataset / f"records/{cond}/observations.csv"),
                                                     args.bootstrap, rng)
        result["conditions"][cond] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    for cond, entry in result["conditions"].items():
        print(f"== {cond}: {entry['description']}")
        for subset in ("all_reference_objects", "target_category_objects"):
            e = entry[subset]
            ci = e["ci95"]
            print(f"  {subset}: {e['objects']} objects, {e['viewpoints']} viewpoints")
            print(f"    mu {e['mu']:.2f} [{ci['mu'][0]:.2f}, {ci['mu'][1]:.2f}]  rho {e['rho']:.2f} "
                  f"[{ci['rho'][0]:.2f}, {ci['rho'][1]:.2f}]  Beta({e['beta_alpha']:.2f}, {e['beta_beta']:.2f})")
            print(f"    P(detect | hit elsewhere) {e['p_detect_after_hit_elsewhere']:.2f}  "
                  f"P(detect | miss elsewhere) {e['p_detect_after_miss_elsewhere']:.2f}")
            c, d = e["consistency"], e["distance"]
            print(f"    all-or-none viewpoints {c['viewpoints_all_or_none']:.2f}, same stop next frame after miss "
                  f"{c['next_frame_same_stop_after_miss']:.2f}; within-object distance missed-detected "
                  f"{d['mean_missed_minus_detected_m']:+.2f} m ({d['objects_missed_farther']}/{d['objects_with_both_outcomes']})")
        f = entry["false_detections"]
        print(f"  false detections per stop: majority {f['per_stop_majority']['rate']:.3f} "
              f"{[round(x, 3) for x in f['per_stop_majority']['ci95']]}, any frame {f['per_stop_any_frame']['rate']:.3f}")
    print("wrote", args.output)
    if args.config:
        search = result["conditions"]["with_targets"]
        fit = search["target_category_objects"]
        config = {
            "model": "athome.symbolic.ViewpointDetectionModel",
            "source": f"{args.dataset.name} (412 robot run5, 13 stops), scripts/fit_detection_412.py",
            "condition": "with_targets: " + search["description"],
            "mean": round(fit["mu"], 3), "rho": round(fit["rho"], 3),
            "ci95": {k: [round(x, 3) for x in fit["ci95"][k]] for k in ("mu", "rho")},
            "objects": fit["objects"], "categories": fit["categories"],
            "false_positive_rates": [round(r, 3) for r in search["false_detections"]["per_category_majority"].values()],
            "false_positive_categories": list(search["false_detections"]["per_category_majority"])}
        args.config.parent.mkdir(parents=True, exist_ok=True)
        args.config.write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("wrote", args.config)


if __name__ == "__main__":
    main()
