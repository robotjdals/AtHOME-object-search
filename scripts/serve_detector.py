"""2D detection server for real-time perception (runs on the GPU server).

    python scripts/serve_detector.py --detector yoloe-l --weights weights --port 8200

POST /detect {"image": <base64 JPEG>, "vocab": [...], "targets": [...]}
  -> {"model": ..., "detect_ms": ..., "size": [w, h], "detections": [
        {"label", "confidence", "box": [x0, y0, x1, y1], "mask": <packed>, "clip_feature": [...]?}]}
GET /health -> {"model": ..., "vocab_size": ...}

The robot's node (ros2/athome_perception) sends each image with the
command's vocabulary (base list + its targets, as VLFM queries the episode's
target); the vocabulary is re-encoded only when it changes. Detections whose
label is one of the targets also get a CLIP image feature of their box (the
model of perception's features, for the search's CLIP verification); others
do not, so the heavy encoder runs only on target candidates. Masks: packed
bits (athome.perception.lifting.unpack_mask). Detector classes and choices:
scripts/detect_frames.py, compared on HM3D renders and the 412 frames
(scripts/evaluate_detectors_hm3d.py, scripts/evaluate_detectors_412.py).
"""
import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

from detect_frames import BoxSegmenter, GroundingDino, OwlV2, Ultralytics, pack, sync


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector", default="yoloe-l",
                   choices=["yoloe-l", "yolo-world-l", "yolo-world-x", "owlv2-b", "grounding-dino-b"])
    p.add_argument("--segmenter", default="none", choices=["none", "mobile_sam", "sam2.1_b"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.05, help="floor; the robot applies its own threshold")
    p.add_argument("--weights", default="weights")
    p.add_argument("--clip-model", default="ViT-H-14/laion2b_s32b_b79k", help="'' to send no CLIP features")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8200)
    args = p.parse_args()

    if args.detector == "grounding-dino-b":
        detector = GroundingDino(args.conf)
    elif args.detector == "owlv2-b":
        detector = OwlV2(args.conf)
    else:
        detector = Ultralytics(args.detector, args.imgsz, args.conf, args.weights)
    segmenter = None if args.segmenter == "none" else BoxSegmenter(args.segmenter, args.weights)
    model_name = f"{args.detector}+{args.segmenter}@{args.imgsz}"

    clip = None
    if args.clip_model:
        import open_clip
        arch, pretrained = args.clip_model.split("/")
        net, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device="cuda")
        clip = (net.eval(), preprocess)

    lock = threading.Lock()
    state = {"vocab": None}

    @torch.inference_mode()
    def clip_features(image, boxes):
        net, preprocess = clip
        crops = []
        for b in boxes:
            x0, y0, x1, y1 = b
            pad = 0.1 * max(x1 - x0, y1 - y0)
            crops.append(preprocess(Image.fromarray(image).crop(
                (max(0, x0 - pad), max(0, y0 - pad), min(image.shape[1], x1 + pad), min(image.shape[0], y1 + pad)))))
        f = net.encode_image(torch.stack(crops).cuda()).float()
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()

    def detect(body):
        image = np.asarray(Image.open(io.BytesIO(base64.b64decode(body["image"]))).convert("RGB"))
        vocab = [str(n) for n in body["vocab"]]
        targets = set(body.get("targets", []))
        with lock:
            if vocab != state["vocab"]:
                detector.set_vocab(vocab)
                state["vocab"] = vocab
            sync()
            t = time.perf_counter()
            dets, masks = detector(image)
            if segmenter is not None:
                masks = segmenter(image, [d["box"] for d in dets])
            if masks is not None:
                for d, m in zip(dets, masks):
                    d["mask"] = pack(m)
            hits = [d for d in dets if d["label"] in targets]
            if clip is not None and hits:
                for d, f in zip(hits, clip_features(image, [d["box"] for d in hits])):
                    d["clip_feature"] = [round(float(v), 5) for v in f]
            sync()
            ms = 1e3 * (time.perf_counter() - t)
        return {"model": model_name, "clip_model": args.clip_model, "detect_ms": round(ms, 1),
                "size": [image.shape[1], image.shape[0]], "detections": dets}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {"error": "unknown path"})
            self._send(200, {"model": model_name, "clip_model": args.clip_model,
                             "vocab_size": len(state["vocab"] or [])})

        def do_POST(self):
            if self.path != "/detect":
                return self._send(404, {"error": "unknown path"})
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if not isinstance(body.get("vocab"), list) or not body["vocab"]:
                    return self._send(400, {"error": "vocab must be a non-empty list"})
                self._send(200, detect(body))
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

    print(f"{model_name} on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
