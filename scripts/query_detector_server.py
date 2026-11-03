"""Send a folder of frames to the detection server (scripts/serve_detector.py)
the way the robot's node does (JPEG bytes, base vocabulary + the frame's
command targets, CLIP features for target-labelled detections) and record the
replies in scripts/detect_frames.py's format, so the evaluations read them
unchanged and the CLIP features are the deployed server's.

    python scripts/query_detector_server.py --url http://127.0.0.1:8200 --frames <dir of .jpg> \
        --vocab vocabulary_base.txt --vocab-per-frame commands.json --out det.jsonl.gz

Standard library only (runs in the server's environment or anywhere).
"""
import argparse
import base64
import gzip
import json
import time
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8200")
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True, help="base vocabulary, one name per line (# comments)")
    p.add_argument("--vocab-per-frame", type=Path, required=True,
                   help="JSON frame stem -> the command's targets (added to the vocabulary, CLIP features)")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--min-confidence", type=float, help="server floor for this run (as the node's request)")
    p.add_argument("--clip-all", action="store_true", help="CLIP features of every detection, not only the targets'")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    base = [l.strip() for l in args.vocab.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]
    targets = json.loads(args.vocab_per_frame.read_text(encoding="utf-8"))
    url = args.url.rstrip("/")
    with urllib.request.urlopen(url + "/health", timeout=args.timeout) as r:
        health = json.loads(r.read())
    # frames of one command together, so the server switches vocabulary once per command
    paths = sorted(args.frames.glob("*.jpg"), key=lambda f: (tuple(targets.get(f.stem, [])), f.name))
    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        out.write(json.dumps({"config": {**{k: str(v) for k, v in vars(args).items()}, "detector": health["model"],
                                         "clip_model": health["clip_model"], "vocab_size": len(base),
                                         "per_frame_extra": True}}) + "\n")
        for f in paths:
            names = targets.get(f.stem, [])
            request = {"image": base64.b64encode(f.read_bytes()).decode("ascii"),
                       "vocab": base + [n for n in names if n not in base], "targets": names}
            if args.min_confidence is not None:
                request["min_confidence"] = args.min_confidence
            if args.clip_all:
                request["clip_all"] = True
            body = json.dumps(request).encode()
            req = urllib.request.Request(url + "/detect", data=body, headers={"Content-Type": "application/json"})
            t = time.perf_counter()
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                res = json.loads(r.read())
            out.write(json.dumps({"frame": f.stem, "detect_ms": res["detect_ms"],
                                  "total_ms": round(1e3 * (time.perf_counter() - t), 2),
                                  "vocabulary_ms": res["vocabulary_ms"], "stages_ms": res.get("stages_ms"),
                                  "detections": res["detections"]}) + "\n")
    print(f"{health['model']} ({health['clip_model']}): {len(paths)} frames -> {args.out}")


if __name__ == "__main__":
    main()
