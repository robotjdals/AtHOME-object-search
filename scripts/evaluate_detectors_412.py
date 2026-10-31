"""Compare detectors for real-time perception on the 412 search frames
(perception's run5 dataset of 2026-10-30: 135 frames at 13 stops, reference
objects checked on photos), with perception's own judging rules
(dataset scripts/measure412.py), so CG-D's published numbers are the baseline.

    # fixed vocabulary for the candidates
    python scripts/evaluate_detectors_412.py vocab --out configs/perception/realtime_vocabulary.txt
    # CG-D (records/<condition>/detections_all.csv) or scripts/detect_frames.py output
    python scripts/evaluate_detectors_412.py evaluate --cgd with_targets --out outputs/perception_bench/cgd.json
    python scripts/evaluate_detectors_412.py evaluate --detections det.jsonl.gz --out outputs/perception_bench/x.json

Judging (as measure412): a reference object is present in a frame when at
least min(5, half) of its LiDAR points project into the image unhidden
(z-buffer of the map); a detection is on it by 2D box IoU >= 0.25, or object
points / centre inside the box (size ratio <= 6 for other names). Rows are
(frame, category); a present row is detected by a same-name (synonym)
detection on the object, an absent row by a same-name detection anywhere
outside 'unclear' objects. Camera poses: frames.csv gives the body (IMU)
pose; with the calibration's T_cam_body it reproduces perception's
visibility table (1038/1038 object views).

3D (detections with masks): the map points visible inside the mask (or the
box) give the object's box, compared with the pre-map box of the reference
object (bottom height drives the Workspace child rule, |dz| < 0.1 m).
"""
import argparse
import base64
import collections
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np
import yaml

from athome.perception.lifting import Camera, FrameView
from athome.scene_graph.wall_map import read_pcd_xyz

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/perception/search412_20261030_zip/search412_dataset_20261030"
CALIB = ROOT / "data/real_run5/real_run5_20261028/robot_recording/calibration.yaml"
SCENE = DATA / "scene_graph_412/scene_graph_dataset/scenes/robot_run5"
W, H, CELL = 1280, 720, 16
BG, MAX_BBOX = {"wall", "floor", "ceiling"}, 0.5          # pipeline filters (measure412)
THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)


# ---------------------------------------------------------------- names
# Names of the candidates' vocabulary that perception's synonym groups lack
# but that fall in one of them (the group already has "mug", "coffee cup").
EXTRA_SYNONYMS = {"cup": ["coffee mug"]}


class Names:
    def __init__(self, groups):
        self.syn = [s + EXTRA_SYNONYMS.get(s[0], []) for s in groups["real_412_synonyms"]]
        self.rel = groups["real_412_related"]

    def canon(self, name):
        n = name.lower().strip()
        return next((s[0] for s in self.syn if n in s), n)

    def same(self, a, b):
        a, b = a.lower(), b.lower()
        return a == b or self.canon(a) == self.canon(b) or any(a in s and b in s for s in self.syn)

    def related(self, a, b):
        a, b = a.lower(), b.lower()
        return self.same(a, b) or any((a in s or self.canon(a) in s) and (b in s or self.canon(b) in s)
                                      for s in self.rel)


# ---------------------------------------------------------------- geometry
def calibration():
    c = yaml.safe_load(CALIB.read_text(encoding="utf-8"))
    k = c["camera"]["color_1280x720"]
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
    return K, np.array(c["extrinsics"]["T_cam_body"], dtype=float)


def camera_pose(row, T_cam_body):
    """frames.csv x, y, yaw: body (IMU) pose in the map, z = 0 (map origin)."""
    yaw = math.radians(float(row["yaw_deg"]))
    T = np.eye(4)
    T[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    T[:3, 3] = [float(row["x"]), float(row["y"]), 0.0]
    return T @ np.linalg.inv(T_cam_body)


def project(K, Tc, P):
    Ti = np.linalg.inv(Tc)
    Pc = P @ Ti[:3, :3].T + Ti[:3, 3]
    z = Pc[:, 2]
    ok = z > 0.3
    uv = np.full((len(P), 2), -1.0)
    uv[ok] = (Pc[ok] @ K.T)[:, :2] / z[ok, None]
    inn = ok & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv, z, inn


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def ref_box(obj):
    b = obj["bbox_3d"]
    c, s, yaw = np.array(b["center"]), np.array(b["size"]), b["yaw"]
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    corners = np.array([[a, b_, c_] for a in (-.5, .5) for b_ in (-.5, .5) for c_ in (-.5, .5)]) * s @ R.T + c
    return c, s, corners


# ---------------------------------------------------------------- dataset
class Dataset:
    """Visibility always from the map (perception's judging); 3D lifting from
    the map, or with ``scans`` (scripts/extract_frame_scans.py) from the
    LiDAR scans around each frame, projected with the bag's camera pose."""

    def __init__(self, scans=None, map_points=None, traj=None):
        self.scans = np.load(scans) if scans else None
        # perception's camera poses (traj.txt, one 4x4 per frame), else
        # rebuilt from frames.csv
        self.traj = np.loadtxt(traj).reshape(-1, 4, 4) if traj else None
        self.map_source = str(map_points) if map_points else "PCD map (downsampled; small objects have few points)"
        self.K, self.T_cam_body = calibration()
        self.frames = list(csv.DictReader(open(DATA / "frames/frames.csv", encoding="utf-8")))
        self.ref = list(csv.DictReader(open(DATA / "reference/reference_objects.csv", encoding="utf-8")))
        self.objects = {o["object_id"]: o for o in json.loads((SCENE / "objects.json").read_text())["objects"]}
        self.points = {int(r["object_id"]): np.load(SCENE / f"points/obj_{int(r['object_id']):04d}.npy").astype(float)
                       for r in self.ref}
        # perception's recording LiDAR map (measure412's z-buffer), else the
        # localization PCD
        self.map = (np.load(map_points)["points"].astype(float) if map_points else
                    read_pcd_xyz(DATA / "scene_graph_412/pcd_map/412_20261028_203939.pcd"))
        self.names = Names(json.loads((DATA / "vocabulary/name_groups.json").read_text(encoding="utf-8")))
        self.views = {int(f["frame"]): self._view(f) for f in self.frames}

    def _view(self, fr):
        Tc = self.traj[int(fr["frame"])] if self.traj is not None else camera_pose(fr, self.T_cam_body)
        uv, z, inn = project(self.K, Tc, self.map)
        zb = np.full((H // CELL, W // CELL), np.inf)
        np.minimum.at(zb, ((uv[inn, 1] // CELL).astype(int), (uv[inn, 0] // CELL).astype(int)), z[inn])
        vis = {}
        for r in self.ref:
            oid = int(r["object_id"])
            u, zz, ii = project(self.K, Tc, self.points[oid])
            if ii.sum() == 0:
                continue
            seen = zz[ii] <= zb[(u[ii, 1] // CELL).astype(int), (u[ii, 0] // CELL).astype(int)] + 0.2
            if seen.sum() < min(5, math.ceil(0.5 * len(self.points[oid]))):
                continue
            _, _, corners = ref_box(self.objects[oid])
            cu, cz, _ = project(self.K, Tc, corners)
            q = u[ii][seen]
            if (cz > 0.3).all():
                bx = [max(0, cu[:, 0].min()), max(0, cu[:, 1].min()), min(W, cu[:, 0].max()), min(H, cu[:, 1].max())]
            else:
                bx = [*q.min(0), *q.max(0)]
            vis[oid] = dict(r=r, box=bx, pts=q, dist=float(np.median(zz[ii][seen])), n=int(seen.sum()))
        view = dict(Tc=Tc, vis=vis, **self._lifting_points(Tc, self.map))
        i = int(fr["frame"])
        if self.scans is not None and f"points_{i}" in self.scans:
            T_bag = self.scans[f"T_cam_{i}"]
            view.update(self._lifting_points(T_bag, self.scans[f"points_{i}"].astype(float)))
            d = np.linalg.inv(Tc) @ T_bag          # reconstructed vs bag camera pose
            view["pose_diff"] = (float(np.linalg.norm(d[:3, 3])),
                                 math.degrees(math.acos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1))))
        return view

    def _lifting_points(self, Tc, P):
        """The node's lifting (athome.perception.lifting.FrameView)."""
        return dict(frame=FrameView(Camera(self.K, W, H, self.T_cam_body), Tc, P))


# ---------------------------------------------------------------- detections
def load_cgd(condition):
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(DATA / f"records/{condition}/detections_all.csv", encoding="utf-8-sig")):
        out[int(r["frame"])].append(dict(label=r["label"], conf=float(r["confidence"]),
                                         box=[float(v) for v in r["box"].split()]))
    return dict(out), {"detector": f"CG-D ({condition})", "source": "perception records"}, {}


def load_detections(path):
    out, timing = {}, {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        config = json.loads(f.readline())["config"]
        for line in f:
            r = json.loads(line)
            i = int("".join(ch for ch in r["frame"] if ch.isdigit()))
            out[i] = [dict(label=d["label"], conf=d["confidence"], box=d["box"], mask=d.get("mask"))
                      for d in r["detections"]
                      if d["label"] not in BG and (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) <= MAX_BBOX * W * H]
            timing[i] = (r["detect_ms"], r["total_ms"])
    return out, config, timing


def unpack(m):
    full = np.zeros((H, W), dtype=bool)
    if m["w"]:
        bits = np.unpackbits(np.frombuffer(base64.b64decode(m["bits"]), dtype=np.uint8))[:m["w"] * m["h"]]
        full[m["y0"]:m["y0"] + m["h"], m["x0"]:m["x0"] + m["w"]] = bits.reshape(m["h"], m["w"]).astype(bool)
    return full


# ---------------------------------------------------------------- judging
def overlaps(det, v, loose=False):
    b = det["box"]
    inside = ((v["pts"][:, 0] >= b[0]) & (v["pts"][:, 0] <= b[2]) &
              (v["pts"][:, 1] >= b[1]) & (v["pts"][:, 1] <= b[3])).mean()
    ab = max((v["box"][2] - v["box"][0]) * (v["box"][3] - v["box"][1]), 1.0)
    ad = (b[2] - b[0]) * (b[3] - b[1])
    cx, cy = (v["box"][0] + v["box"][2]) / 2, (v["box"][1] + v["box"][3]) / 2
    centre_in = b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
    if loose:
        return box_iou(b, v["box"]) >= 0.25 or inside >= 0.5 or centre_in
    return box_iou(b, v["box"]) >= 0.25 or ((inside >= 0.6 or centre_in) and ad <= 6 * ab and ab <= 6 * ad)


def judge(ds, dets_by_frame):
    """Rows as measure412, each keeping the best confidence of its matching
    detections (so any gate can be applied afterwards)."""
    nm = ds.names
    for i, dets in dets_by_frame.items():
        vis = ds.views[i]["vis"]
        for d in dets:
            d["objects"] = [o for o, v in vis.items() if overlaps(d, v)]
            d["loose"] = [o for o, v in vis.items() if overlaps(d, v, True)]
            d["unclear"] = any(vis[o]["r"]["check"] == "unclear" for o in d["objects"])
    ref_cats = {nm.canon(r["category_used"]) for r in ds.ref if r["check"] != "unclear"}
    cats = sorted(ref_cats | {nm.canon(d["label"]) for ds_ in dets_by_frame.values() for d in ds_})
    rows, views = [], []
    for fr in ds.frames:
        i = int(fr["frame"])
        vis, dets = ds.views[i]["vis"], dets_by_frame.get(i, [])
        for oid, v in vis.items():
            r = v["r"]
            if r["check"] == "unclear":
                continue
            same = [d for d in dets if oid in d["loose"] and nm.same(d["label"], r["category_used"])]
            views.append(dict(frame=i, stop=int(fr["stop"]), object_id=oid, category=nm.canon(r["category_used"]),
                              dist=v["dist"], dets=same))
        for c in cats:
            present = [v for v in vis.values() if v["r"]["check"] != "unclear" and nm.canon(v["r"]["category_used"]) == c]
            if present:
                ids = {int(v["r"]["object_id"]) for v in present}
                raw = [d for d in dets if nm.same(d["label"], c) and set(d["loose"]) & ids]
                excl = False
            else:
                raw = [d for d in dets if nm.same(d["label"], c)]
                excl = bool(raw) and all(d["unclear"] for d in raw)
            rows.append(dict(frame=i, category=c, present=bool(present), in_ref=c in ref_cats, excluded=excl, dets=raw))
    return rows, views


def gate_fn(gate, target_gate=None, targets=()):
    return lambda d: d["conf"] >= (target_gate if target_gate is not None and d["label"] in targets else gate)


def rates(rows, views, passes, n_frames):
    det = lambda r: any(passes(d) for d in r["dets"])
    pr = [r for r in rows if r["present"]]
    ab = [r for r in rows if not r["present"] and not r["excluded"]]
    mean = lambda a: round(float(np.mean(a)), 3) if len(a) else None
    return dict(
        recall=mean([det(r) for r in pr]), present_rows=len(pr),
        false_rate_ref_categories=mean([det(r) for r in ab if r["in_ref"]]),
        false_rate_other_categories=mean([det(r) for r in ab if not r["in_ref"]]),
        false_per_frame_ref_categories=round(sum(det(r) for r in ab if r["in_ref"]) / n_frames, 3),
        false_per_frame_other_categories=round(sum(det(r) for r in ab if not r["in_ref"]) / n_frames, 3),
        recall_object_views=mean([det(v) for v in views]))


def bootstrap_ci(ds, rows, views, passes, n=2000, seed=0):
    """95 % bootstrap over robot stops (frames of one stop are nearly the
    same view, so stops are the independent units)."""
    stop = {int(f["frame"]): f["stop"] for f in ds.frames}
    by_r, by_v = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        by_r[stop[r["frame"]]].append(r)
    for v in views:
        by_v[stop[v["frame"]]].append(v)
    stops = sorted(by_r)
    rng = np.random.default_rng(seed)
    rec, fp = [], []
    for _ in range(n):
        pick = rng.choice(stops, len(stops))
        rr = [r for s_ in pick for r in by_r[s_]]
        pr = [any(passes(d) for d in r["dets"]) for r in rr if r["present"]]
        ab = [any(passes(d) for d in r["dets"]) for r in rr if not r["present"] and not r["excluded"] and r["in_ref"]]
        rec.append(np.mean(pr) if pr else np.nan)
        fp.append(np.mean(ab) if ab else np.nan)
    q = lambda a: [round(float(np.nanpercentile(a, x)), 3) for x in (2.5, 97.5)]
    return {"recall": q(rec), "false_rate_ref_categories": q(fp)}


# ---------------------------------------------------------------- 3D
def lift(view, region):
    b = view["frame"].lift(region)
    if b is None:
        return None
    c, sz = np.array(b.center), np.array(b.size)
    return c - sz / 2, c + sz / 2, b.points


def lift_standard(view, region, erosion_px=5, eps=0.05, min_points=10):
    """The common rule of the open-source mappers: the mask eroded by 5 px
    (VLFM object_map_erosion_size), DBSCAN with ConceptGraphs' parameters
    (dbscan_eps 0.05, dbscan_min_points 10) keeping the largest cluster, and
    ConceptGraphs' fallback to all points when that cluster has < 5."""
    from scipy.ndimage import binary_erosion
    from sklearn.cluster import DBSCAN
    region = binary_erosion(region, iterations=erosion_px)
    f = view["frame"]
    keep = region[f.uv[:, 1].astype(int), f.uv[:, 0].astype(int)]
    if keep.sum() < 3:
        return None
    P = f.xyz[keep]
    labels = DBSCAN(eps=eps, min_samples=min_points).fit(P).labels_
    sizes = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.array([])
    if len(sizes) and sizes.max() >= 5:
        P = P[labels == int(sizes.argmax())]
    return P.min(0), P.max(0), len(P)


LIFT = {"nearest": lift, "standard": lift_standard}


def lift_errors(ds, views, passes, rule="nearest"):
    """Per size class: small objects (pre-map box < 0.5 m; the ones the
    Workspace child rule places) and large ones (often partly in view)."""
    out = collections.defaultdict(list)
    for v in views:
        hits = [d for d in v["dets"] if passes(d)]
        if not hits:
            continue
        d = max(hits, key=lambda x: x["conf"])
        c, s, _ = ref_box(ds.objects[v["object_id"]])
        size = "small" if s.max() < 0.5 else "large"
        view = ds.views[v["frame"]]
        b = d["box"]
        box_region = np.zeros((H, W), dtype=bool)
        box_region[int(b[1]):int(math.ceil(b[3])), int(b[0]):int(math.ceil(b[2]))] = True
        for kind, region in (("box", box_region), ("mask", unpack(d["mask"]) if d.get("mask") else None)):
            if region is None:
                continue
            key = (kind, size)
            out[key, "tried"].append(1)
            r = LIFT[rule](view, region)
            if r is None:
                continue
            lo, hi, n = r
            out[key, "bottom"].append(abs(lo[2] - (c[2] - s[2] / 2)))
            out[key, "top"].append(abs(hi[2] - (c[2] + s[2] / 2)))
            out[key, "xy"].append(float(np.linalg.norm((lo[:2] + hi[:2]) / 2 - c[:2])))
    res = {}
    for kind in ("box", "mask"):
        for size in ("small", "large"):
            key = (kind, size)
            tried = len(out[key, "tried"])
            if not tried:
                continue
            bot = np.array(out[key, "bottom"])
            med = lambda a: round(float(np.median(a)), 3) if len(a) else None
            res[f"{kind}_{size}"] = dict(
                detections=tried, lifted=round(len(bot) / tried, 3), bottom_abs_err_median_m=med(bot),
                bottom_within_5cm=round(float(np.mean(bot <= 0.05)), 3) if len(bot) else None,
                bottom_within_10cm=round(float(np.mean(bot <= 0.10)), 3) if len(bot) else None,
                top_abs_err_median_m=med(out[key, "top"]), xy_err_median_m=med(out[key, "xy"]))
    return res


# ---------------------------------------------------------------- commands
def cmd_evaluate(args):
    ds = Dataset(args.scans, args.map_points, args.traj)
    if args.cgd:
        dets, config, timing = load_cgd(args.cgd)
    else:
        dets, config, timing = load_detections(args.detections)
    rows, views = judge(ds, dets)
    n = len(ds.frames)
    cgd_summary = json.loads((DATA / "records/with_targets/summary_412.json").read_text(encoding="utf-8"))
    targets = cgd_summary["pipeline"]["targets"]
    diffs = [v["pose_diff"] for v in ds.views.values() if "pose_diff" in v]
    if diffs:
        config = {**config, "lifting_points": f"LiDAR scans ({args.scans})",
                  "pose_vs_bag_median": {"m": round(float(np.median([d[0] for d in diffs])), 3),
                                         "deg": round(float(np.median([d[1] for d in diffs])), 2)}}
    else:
        config = {**config, "lifting_points": ds.map_source}
    result = {"config": config,
              "sweep": {str(t): rates(rows, views, gate_fn(t), n) for t in THRESHOLDS}}
    if args.cgd:      # perception's gate: 0.25, target names 0.4 when they were queried
        queried = targets if args.cgd == "with_targets" else []
        result["pipeline_gate"] = rates(rows, views, gate_fn(0.25, 0.4, queried), n)
        result["pipeline_gate"]["ci95"] = bootstrap_ci(ds, rows, views, gate_fn(0.25, 0.4, queried))
        result["pipeline_gate"]["gate"] = "0.25 (target names 0.4)" if queried else "0.25"
    # operating point: the lowest threshold whose reference-category false
    # rate is no higher than CG-D's pipeline (0.079, with target queries)
    cap = cgd_summary["pipeline"]["false_detection_rate_reference_categories"]
    ok = [t for t in THRESHOLDS if (result["sweep"][str(t)]["false_rate_ref_categories"] or 0) <= cap]
    if ok:
        t = min(ok)
        result["at_cgd_false_rate"] = {"threshold": t, "false_rate_cap": cap, **result["sweep"][str(t)],
                                       "recall_ci95": bootstrap_ci(ds, rows, views, gate_fn(t)),
                                       "lifting": {rule: lift_errors(ds, views, gate_fn(t), rule) for rule in args.lift}}
    by_cat = collections.defaultdict(list)
    t = result.get("at_cgd_false_rate", {}).get("threshold", 0.25)
    for r in rows:
        if r["present"]:
            by_cat[r["category"]].append(any(d["conf"] >= t for d in r["dets"]))
    result["recall_by_category"] = {c: dict(rows=len(v), recall=round(float(np.mean(v)), 3)) for c, v in sorted(by_cat.items())}
    if timing:
        det_ms, tot_ms = np.array(list(timing.values())).T
        result["timing_ms"] = {"detect_median": round(float(np.median(det_ms)), 1),
                               "total_median": round(float(np.median(tot_ms)), 1),
                               "total_p95": round(float(np.percentile(tot_ms, 95)), 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    show = {k: result[k] for k in ("pipeline_gate", "at_cgd_false_rate", "timing_ms") if k in result}
    print(config.get("detector"), config.get("segmenter", ""), config.get("imgsz", ""))
    print(json.dumps(show, indent=1, ensure_ascii=False))


def cmd_vocab(args):
    """Vocabularies from the training data and a generic household list, never
    from the test room: ``base`` = the Workspace source categories (new
    furniture becomes a Workspace) + perception's household list; ``large``
    = base + all target categories (one fixed list). With a per-command
    vocabulary the detector gets base + the command's targets
    (--commands writes the 412 command: perception's 15 queried targets)."""
    targets = json.loads((ROOT / "configs/data/target_categories.v5.json").read_text(encoding="utf-8"))["categories"]
    priors = json.loads((ROOT / "configs/data/workspace_source_priors.json").read_text(encoding="utf-8"))["categories"]
    sources = [c for c, p in priors.items() if p["rate"] > 0.5 and p["present"] >= 20]
    household = [l.strip() for l in (DATA / "vocabulary/household.txt").read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
    groups = {"base": (sources, household), "large": (sources, household, list(targets))}[args.set]
    names = []
    for group in groups:
        for n in group:
            if n.lower() not in names:
                names.append(n.lower())
    header = (f"# Real-time perception vocabulary '{args.set}' (scripts/evaluate_detectors_412.py vocab):\n"
              f"# Workspace source categories ({len(sources)}), perception household.txt ({len(household)})"
              + (f", target_categories.v5 ({len(targets)})" if args.set == "large" else "") + f"; {len(names)} names.\n")
    args.out.write_text(header + "\n".join(names) + "\n", encoding="utf-8")
    print(f"{len(names)} names -> {args.out}")
    if args.commands:
        queried = json.loads((DATA / "records/with_targets/summary_412.json").read_text(encoding="utf-8"))["pipeline"]["targets"]
        frames = list(csv.DictReader(open(DATA / "frames/frames.csv", encoding="utf-8")))
        args.commands.write_text(json.dumps({f"frame{int(f['frame']):06d}": queried for f in frames}), encoding="utf-8")
        print(f"412 command ({len(queried)} targets) -> {args.commands}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vocab")
    v.add_argument("--set", choices=["base", "large"], required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--commands", type=Path, help="also write the 412 per-frame command targets (JSON)")
    e = sub.add_parser("evaluate")
    src = e.add_mutually_exclusive_group(required=True)
    src.add_argument("--cgd", choices=["with_targets", "no_targets"])
    src.add_argument("--detections", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--scans", type=Path, help="scripts/extract_frame_scans.py output: lift with LiDAR scans")
    e.add_argument("--map-points", type=Path, help="perception's map_points.npz (visibility and lifting)")
    e.add_argument("--traj", type=Path, help="perception's traj.txt (camera poses of the frames)")
    e.add_argument("--lift", nargs="+", choices=["nearest", "standard"], default=["nearest", "standard"])
    args = p.parse_args()
    {"vocab": cmd_vocab, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
