"""CG-D (perception's RAM -> GroundingDINO -> SAM -> CLIP, tools/robot_live/cgd_engine.py,
unchanged) timed per stage on a folder of images, with RAM on the CPU (its
8 GB GPU setting) or on the GPU (GSA_RAM_GPU=1), recording detections in
scripts/detect_frames.py's format so the found check comparisons read them.
Runs in perception's environment on the GPU server:

    source activate.sh
    GSA_RAM_GPU=1 python time_cgd_stages.py --robot-live <tools/robot_live> --images <dir> \
        --targets-json cmd.json --out cgd.jsonl.gz --clip-model ViT-H-14
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot-live", type=Path, required=True)
parser.add_argument("--images", type=Path, required=True)
parser.add_argument("--targets-json", type=Path, help="frame stem -> target names")
parser.add_argument("--targets", nargs="*", default=[])
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--limit", type=int)
args = parser.parse_args()
images = args.images.resolve() if args.images.is_absolute() else Path.cwd() / args.images
out_path = args.out if args.out.is_absolute() else Path.cwd() / args.out
targets_by_frame = json.loads(args.targets_json.read_text()) if args.targets_json else {}
sys.path.insert(0, str(args.robot_live.resolve()))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from cgd_engine import CGDEngine, gsa, gsa_fast  # noqa: E402

times = {}


def timed(name, fn):
    def run(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            torch.cuda.synchronize()
            times[name] = times.get(name, 0.0) + time.perf_counter() - t
    return run


gsa.inference_ram = timed("ram", gsa.inference_ram)
gsa.Model.predict_with_classes = timed("dino", gsa.Model.predict_with_classes)
gsa.get_sam_segmentation_from_xyxy = timed("sam", gsa.get_sam_segmentation_from_xyxy)
gsa.compute_clip_features = timed("clip", gsa.compute_clip_features)
gsa.vis_result_fast = timed("draw", gsa.vis_result_fast)            # the annotated image of the live node
gsa.process_tag_classes = timed("classes", gsa.process_tag_classes)

t0 = time.time()
engine = CGDEngine("cuda")
print(f"models loaded in {time.time() - t0:.0f} s; RAM on {'GPU' if os.environ.get('GSA_RAM_GPU') == '1' else 'CPU'}; "
      f"allocated {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)

paths = sorted(images.glob("*.jpg"))[:args.limit]
rows = []
with gzip.open(out_path, "wt", encoding="utf-8") as out:
    out.write(json.dumps({"config": {"detector": "CG-D", "ram_gpu": os.environ.get("GSA_RAM_GPU") == "1"}}) + "\n")
    for k, f in enumerate(paths):
        image = cv2.imread(str(f))
        times.clear()
        t = time.perf_counter()
        r = engine.run(image, targets_by_frame.get(f.stem, args.targets))
        total = time.perf_counter() - t
        rows.append(dict(times, total=total, objects=len(r["objects"]), n_classes=r["n_classes"],
                         other=total - sum(times.values())))
        dets = [dict(label=o["label"], confidence=o["confidence"], box=o["bbox"],
                     clip_feature=[round(float(v), 5) for v in o["clip"]]) for o in r["objects"]]
        out.write(json.dumps({"frame": f.stem, "detect_ms": round(1e3 * total, 1), "total_ms": round(1e3 * total, 1),
                              "vocabulary_ms": 0.0, "stages_ms": {n: round(1e3 * times.get(n, 0.0), 1) for n in ("ram", "dino", "sam", "clip")},
                              "ram_tags": r["ram_tags"], "detections": dets}) + "\n")
        if k % 20 == 0:
            print(f"[{k}/{len(paths)}] {total:.2f} s, objects {len(r['objects'])}", flush=True)

skip = rows[1:] or rows                                 # the first image also warms up
summary = {name: dict(median=round(float(np.median([x.get(name, 0.0) for x in skip])), 3),
                      p90=round(float(np.percentile([x.get(name, 0.0) for x in skip], 90)), 3))
           for name in ("total", "ram", "dino", "sam", "clip", "draw", "other", "objects")}
summary["frames"] = len(rows)
summary["ram_gpu"] = os.environ.get("GSA_RAM_GPU") == "1"
summary["gpu_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
summary["gpu_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
summary["stage_peaks"] = {k: [round(x / 2**30, 2) for x in (v if isinstance(v, tuple) else (v,))] for k, v in gsa_fast._peak.items()}
print(json.dumps(summary, indent=1), flush=True)
out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
