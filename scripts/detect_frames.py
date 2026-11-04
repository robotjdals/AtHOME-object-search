"""Run one open-vocabulary detector (+ box-prompted segmenter) on a folder of
frames and record every detection with its mask and per-frame timing, for
the real-time perception comparison (scripts/evaluate_detectors_412.py).

Runs on the GPU server in its own environment (no athome import):

    python scripts/detect_frames.py --frames <dir of .jpg> --vocab vocab.txt \
        --detector yolo-world-l --segmenter mobile_sam --imgsz 640 --out det.jsonl.gz

--vocab-per-frame adds a command's targets per frame (per-command
vocabulary, as VLFM queries the episode's target).

Detectors: yolo-world-l / yolo-world-x (YOLO-World v2, Cheng et al. 2024; the
vocabulary is encoded once, as in its offline-vocabulary mode), yoloe-l
(YOLOE-11L-seg, Wang et al. 2025; masks from the detector itself),
grounding-dino-b (Liu et al. 2023; the vocabulary does not fit its 256-token
text input and is queried in chunks, as CG-D's queries are), owlv2-b
(OWLv2 base, Minderer et al. 2023; OK-Robot, DynaMem). Segmenters:
mobile_sam, sam2.1_b (box prompts, one image encoding per frame), none.

The confidence floor is low on purpose so that thresholds can be chosen
afterwards. Times are GPU-synchronised wall times per frame after a warm-up;
the image is decoded before the clock starts (the robot sends it decoded).
"""
import argparse
import base64
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

WEIGHTS = {"yolo-world-l": "yolov8l-worldv2.pt", "yolo-world-x": "yolov8x-worldv2.pt",
           "yoloe-l": "yoloe-11l-seg.pt", "mobile_sam": "mobile_sam.pt", "sam2.1_b": "sam2.1_b.pt"}
GDINO = "IDEA-Research/grounding-dino-base"
OWLV2 = "google/owlv2-base-patch16-ensemble"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Ultralytics:
    def __init__(self, name, imgsz, conf, weights, multi_label=False):
        """``multi_label``: a box keeps every name above ``conf``, not only its
        best one (the per-class scores are independent sigmoids). Needs the
        library's multi_label and a per-class NMS: its predictor runs YOLOE
        with agnostic_nms, which removes the other names of a box."""
        from ultralytics import YOLOE, YOLOWorld
        self.max_det = 300
        if multi_label:
            from ultralytics.utils import nms
            original = nms.non_max_suppression

            def per_class(prediction, conf=0.25, iou=0.45, classes=None, agnostic=False, **kwargs):
                return original(prediction, conf, iou, classes, False, multi_label=True, **kwargs)
            nms.non_max_suppression = per_class
            self.max_det = 3000
        path = str(Path(weights) / WEIGHTS[name])
        self.yoloe = name.startswith("yoloe")
        self.model = YOLOE(path) if self.yoloe else YOLOWorld(path)
        self.imgsz, self.conf = imgsz, conf
        self._text = {}                        # YOLOE: name -> text embedding (1, 1, D)
        if self.yoloe and torch.cuda.is_available():
            # the text encoder runs on the model's device: on the CPU it took
            # ~0.16 s per name (11 s for 66), on the GPU 0.5 s for all
            self.model.model.to("cuda")

    def set_vocab(self, vocab):
        if self.yoloe:
            new = [n for n in vocab if n not in self._text]
            if new:
                for n, pe in zip(new, self.model.get_text_pe(new).split(1, dim=1)):
                    self._text[n] = pe
            self.model.set_classes(vocab, torch.cat([self._text[n] for n in vocab], dim=1))
            # The predictor keeps working with the new classes: same detections
            # as a rebuilt predictor and a fresh model (checked on a 412 frame).
        else:
            self.model.set_classes(vocab)
            self.model.predictor = None        # rebuilt with the new class names

    def __call__(self, image):
        # retina_masks: masks at the image size from the GPU (upsampled before
        # thresholding); resizing each network-size mask on the CPU took ~6 ms
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, max_det=self.max_det, verbose=False,
                               retina_masks=True)[0]
        names = r.names
        dets = [dict(label=names[int(c)], confidence=float(s), box=[float(v) for v in b])
                for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
        masks = None
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy() > 0.5
            if masks.shape[1:] != image.shape[:2]:      # masks come at the network size
                masks = np.stack([np.asarray(Image.fromarray(m).resize(image.shape[1::-1], Image.NEAREST))
                                  for m in masks]) if len(masks) else masks
        return dets, masks


class GroundingDino:
    """Vocabulary queried in chunks that fit the 256-token text input."""

    def __init__(self, conf, chunk_tokens=200):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(GDINO)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).cuda().eval()
        self.conf, self.chunk_tokens = conf, chunk_tokens

    def set_vocab(self, vocab):
        self.chunks, chunk, used = [], [], 0
        for name in vocab:
            n = len(self.processor.tokenizer(name + " .", add_special_tokens=False)["input_ids"])
            if chunk and used + n > self.chunk_tokens:
                self.chunks.append(chunk)
                chunk, used = [], 0
            chunk.append(name)
            used += n
        self.chunks.append(chunk)

    @torch.inference_mode()
    def __call__(self, image):
        pil = Image.fromarray(image)
        dets = []
        for chunk in self.chunks:
            inputs = self.processor(images=pil, text=" . ".join(chunk) + " .", return_tensors="pt").to("cuda")
            out = self.model(**inputs)
            r = self.processor.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=self.conf, text_threshold=self.conf,
                target_sizes=[image.shape[:2]])[0]
            labels = r.get("text_labels", r["labels"])
            for b, s, lab in zip(r["boxes"].tolist(), r["scores"].tolist(), labels):
                if lab:
                    dets.append(dict(label=lab, confidence=float(s), box=[float(v) for v in b]))
        return dets, None


class OwlV2:
    """OWLv2 (Minderer et al. 2023), the detector of OK-Robot / DynaMem. The
    text queries are encoded once per vocabulary; boxes refer to the padded
    square input, so the target size is the longer image side."""

    def __init__(self, conf):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.processor = Owlv2Processor.from_pretrained(OWLV2)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWLV2).cuda().eval()
        self.conf = conf

    @torch.inference_mode()
    def set_vocab(self, vocab):
        self.vocab = list(vocab)
        t = self.processor(text=[[f"a photo of a {n}" for n in self.vocab]], return_tensors="pt").to("cuda")
        self.text = {"input_ids": t.input_ids, "attention_mask": t.attention_mask}

    @torch.inference_mode()
    def __call__(self, image):
        side = max(image.shape[:2])
        px = self.processor(images=Image.fromarray(image), return_tensors="pt").to("cuda")
        out = self.model(pixel_values=px.pixel_values, **self.text)
        r = self.processor.post_process_object_detection(out, threshold=self.conf, target_sizes=[(side, side)])[0]
        dets = [dict(label=self.vocab[int(l)], confidence=float(s), box=[float(v) for v in b])
                for b, s, l in zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist())]
        return dets, None


class BoxSegmenter:
    def __init__(self, name, weights):
        from ultralytics import SAM
        self.model = SAM(str(Path(weights) / WEIGHTS[name]))

    def __call__(self, image, boxes):
        if not boxes:
            return np.zeros((0, *image.shape[:2]), dtype=bool)
        r = self.model.predict(image, bboxes=boxes, verbose=False)[0]
        return r.masks.data.cpu().numpy() > 0.5


def pack(m, box=None):
    """Mask cropped to a box (the detection's box: the detector's masks lie
    inside it; else the box of its pixels), bits packed row-major (decode:
    np.unpackbits(bits)[:h * w].reshape(h, w) placed at x0, y0)."""
    if box is not None:
        x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
        x1, y1 = min(m.shape[1], int(np.ceil(box[2]))), min(m.shape[0], int(np.ceil(box[3])))
    else:
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            return {"x0": 0, "y0": 0, "w": 0, "h": 0, "bits": ""}
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    if x1 <= x0 or y1 <= y0:
        return {"x0": 0, "y0": 0, "w": 0, "h": 0, "bits": ""}
    bits = np.packbits(m[y0:y1, x0:x1].ravel())
    return {"x0": int(x0), "y0": int(y0), "w": int(x1 - x0), "h": int(y1 - y0),
            "bits": base64.b64encode(bits.tobytes()).decode("ascii")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True, help="one class name per line (# comments)")
    p.add_argument("--vocab-per-frame", type=Path,
                   help="JSON frame stem -> extra names (a command's targets) added to --vocab for that frame")
    p.add_argument("--detector", required=True,
                   choices=["yolo-world-l", "yolo-world-x", "yoloe-l", "grounding-dino-b", "owlv2-b"])
    p.add_argument("--segmenter", default="mobile_sam", choices=["mobile_sam", "sam2.1_b", "none"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--weights", default="weights")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    vocab = [l.strip() for l in args.vocab.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    extra = json.loads(args.vocab_per_frame.read_text(encoding="utf-8")) if args.vocab_per_frame else {}
    t0 = time.perf_counter()
    if args.detector == "grounding-dino-b":
        detector = GroundingDino(args.conf)
    elif args.detector == "owlv2-b":
        detector = OwlV2(args.conf)
    else:
        detector = Ultralytics(args.detector, args.imgsz, args.conf, args.weights)
    segmenter = None if args.segmenter == "none" else BoxSegmenter(args.segmenter, args.weights)
    setup_s = time.perf_counter() - t0

    # frames of one command together, so its vocabulary is set once
    paths = sorted(args.frames.glob("*.jpg"), key=lambda f: (tuple(extra.get(f.stem, [])), f.name))
    images = [np.asarray(Image.open(f).convert("RGB")) for f in paths]
    current = None

    def use_vocab(stem):
        """A command's vocabulary is set once per command on the robot, so
        it is set (and warmed up) outside the per-frame clock."""
        nonlocal current
        names = vocab + [n for n in extra.get(stem, []) if n not in vocab]
        if names != current:
            detector.set_vocab(names)
            current = names
            for image in images[:args.warmup]:
                dets, _ = detector(image)
                if segmenter is not None:
                    segmenter(image, [d["box"] for d in dets])
            sync()

    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        out.write(json.dumps({"config": {**{k: str(v) for k, v in vars(args).items()}, "vocab_size": len(vocab), "per_frame_extra": bool(extra),
                                         "setup_s": round(setup_s, 2), "gpu": torch.cuda.get_device_name()}}) + "\n")
        for f, image in zip(paths, images):
            use_vocab(f.stem)
            sync()
            t = time.perf_counter()
            dets, masks = detector(image)
            sync()
            t_det = time.perf_counter() - t
            if segmenter is not None:
                masks = segmenter(image, [d["box"] for d in dets])
                sync()
            t_all = time.perf_counter() - t
            for d, m in zip(dets, masks if masks is not None else [None] * len(dets)):
                if m is not None:
                    d["mask"] = pack(m, d["box"])
            out.write(json.dumps({"frame": f.stem, "detect_ms": round(1e3 * t_det, 2),
                                  "total_ms": round(1e3 * t_all, 2), "detections": dets}) + "\n")
    print(f"{args.detector}+{args.segmenter}@{args.imgsz}: {len(paths)} frames -> {args.out}")


if __name__ == "__main__":
    main()
