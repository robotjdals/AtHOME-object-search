"""Perception's CG-D (tools/robot_live/cgd_engine.py, unchanged) behind the
detection server's HTTP interface (scripts/serve_detector.py), for the
perception check page. Runs in perception's environment on the GPU server:

    source activate.sh
    GSA_RAM_GPU=1 python serve_cgd_http.py --robot-live <tools/robot_live> --port 8200

POST /detect {"image": <base64 JPEG>, "query": [names added to the classes] (default "targets"), "classes": "list",
              "vocab": [fixed list], "masks": true}
  -> {"model", "detect_ms", "stages_ms", "ram_tags", "size", "detections":
      [{"label", "confidence", "box", "clip_feature", "mask": <packed, detect_frames.pack>}]}
GET /health -> {"model", "clip_model", "vocab_size"}

"classes", the DINO class names besides "query": "list" ("vocab", or perception's
household list without it; default), "target" (none) or "ram" (RAM tags +
household list, as the engine runs). Without RAM its tags do not compete with the search's names.

The engine's annotated image is not drawn: the page draws the boxes it gets
back, and drawing took about half of each request (4-7 s, scripts/time_cgd_stages.py).
"""
import argparse
import base64
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot-live", type=Path, required=True)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8200)
args = parser.parse_args()
sys.path.insert(0, str(args.robot_live.resolve()))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from cgd_engine import CGDEngine, gsa, gsa_fast  # noqa: E402
from cgd_fast import install  # noqa: E402
from detect_frames import pack  # noqa: E402

install(gsa, gsa_fast)
times = {}


def timed(name, fn):
    def run(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            times[name] = 1e3 * (time.perf_counter() - t)
    return run


household = [l.strip() for l in open(os.environ["GSA_EXTRA_CLASSES_FILE"]) if l.strip() and not l.startswith("#")]
mode = {"classes": "list", "vocab": household}
_ram, _tags = gsa.inference_ram, gsa.process_tag_classes
gsa.inference_ram = lambda image, model: _ram(image, model) if mode["classes"] == "ram" else ("",)
gsa.process_tag_classes = lambda *a, **k: (_tags(*a, **k) if mode["classes"] == "ram" else (
    ["other item", "ceiling", "floor", "wall"] + (mode["vocab"] if mode["classes"] == "list" else [])))
gsa.inference_ram = timed("ram", gsa.inference_ram)
gsa.Model.predict_with_classes = timed("dino", gsa.Model.predict_with_classes)
gsa.get_sam_segmentation_from_xyxy = timed("sam", gsa.get_sam_segmentation_from_xyxy)
gsa.compute_clip_features = timed("clip", gsa.compute_clip_features)
gsa.vis_result_fast = lambda image, detections, classes, **kwargs: (image, [])

engine = CGDEngine("cuda")
MODEL = "cg-d"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            return self._send(404, {"error": "unknown path"})
        self._send(200, {"model": MODEL, "clip_model": "ViT-H-14/laion2b_s32b_b79k", "vocab_size": 0})

    def do_POST(self):
        if self.path != "/detect":
            return self._send(404, {"error": "unknown path"})
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = np.frombuffer(base64.b64decode(body["image"]), np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            times.clear()
            mode["classes"] = body.get("classes", "list")
            mode["vocab"] = body.get("vocab") or household
            t = time.perf_counter()
            r = engine.run(image, body.get("query", body.get("targets", [])))
            t_masks = body.get("masks", True)
            torch.cuda.synchronize()
            self._send(200, {
                "model": MODEL, "detect_ms": round(1e3 * (time.perf_counter() - t), 1),
                "vocabulary_ms": 0.0, "stages_ms": {k: round(v, 1) for k, v in times.items()},
                "ram_tags": r["ram_tags"], "size": [image.shape[1], image.shape[0]],
                "detections": [dict(label=o["label"], confidence=o["confidence"], box=o["bbox"],
                                    clip_feature=[round(float(v), 5) for v in o["clip"]],
                                    **({"mask": pack(o["mask"], o["bbox"])} if t_masks else {}))
                               for o in r["objects"]]})
        except Exception as e:  # noqa: BLE001 - reported to the client
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


print(f"{MODEL} on {args.host}:{args.port}", flush=True)
HTTPServer((args.host, args.port), Handler).serve_forever()     # one request at a time, all GPU work on this thread
