"""How typical each experiment placement is, from the HM3D training buildings:
the share of the object's instances that stand in the placement's room type,
counted over the room types of the experiment home only. A share of at least
1 / (number of those room types) is typical, below it atypical.

Training buildings: the train split without the SFT validation buildings (the
scenes the planner was trained on). Primary count: the exact HM3D tag; the
synonym count (e.g. "stack of books") is reported beside it as a check.

    python3 scripts/placement_typicality.py experiments/room_baram_final/scenarios.csv
"""
import argparse
import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from athome.data.hm3d.layout import load_layout, repo_path  # noqa: E402
from athome.scene_graph.query import normalize_category  # noqa: E402

SYNONYMS = {"book": {"book", "stack of books"},
            "plant": {"plant", "decorative plant", "flowerpot"},
            "laptop": {"laptop"}}


def count(graphs, tags):
    """(tag group, room label) -> instances, over the given scene graphs."""
    counts = Counter()
    for g in graphs:
        labels = {r["room_id"]: r["room_label"] for r in g["rooms"]}
        for o in g["objects"]:
            tag = normalize_category(o["semantic_tag"])
            for name, group in tags.items():
                if tag in group and o.get("room_id") in labels:
                    counts[(name, labels[o["room_id"]])] += 1
    return counts


def shares(counts, name, rooms):
    """Share of ``name``'s instances in each of ``rooms`` (renormalized over them)."""
    total = sum(counts[(name, r)] for r in rooms)
    return {r: counts[(name, r)] / total if total else 0.0 for r in rooms}


def training_graphs():
    sys.path.insert(0, str(ROOT / "scripts"))
    from export_sft_dataset import scene_split      # the data tools' environment (shapely)
    from train_grpo import held_out

    grpo = yaml.safe_load((ROOT / "configs/training/grpo.yaml").read_text(encoding="utf-8"))
    sft = yaml.safe_load((ROOT / "configs/training/sft_lora.yaml").read_text(encoding="utf-8"))
    fraction = sft["validation"]["holdout_scene_fraction"]
    split_config = json.loads((ROOT / "configs/data/scene_splits.json").read_text(encoding="utf-8"))
    for path in sorted(glob.glob(str(ROOT / grpo["layouts"].format(scene_id="*")))):
        layout = load_layout(path)
        scene_id = Path(path).name.split(".")[0]
        scene = json.loads(layout.annotations.read_text(encoding="utf-8"))["scene_path"]
        if scene_split(repo_path(scene), split_config) == "train" and not held_out(scene_id, fraction):
            yield json.loads(layout.graph.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, help="default: typicality.csv beside the plan")
    args = parser.parse_args()
    with open(args.plan, encoding="utf-8") as f:
        plan = list(csv.DictReader(f))
    rooms = sorted({r["room"] for r in plan})
    objects = sorted({r["object"] for r in plan})
    graphs = list(training_graphs())
    exact = count(graphs, {o: {o} for o in objects})
    synonym = count(graphs, {o: SYNONYMS.get(o, {o}) for o in objects})
    threshold = 1 / len(rooms)
    print(f"학습 건물 {len(graphs)}개, 방 종류 {', '.join(rooms)}, 기준 {threshold:.2f}")
    for o in objects:
        print(f"  {o:<7} " + "  ".join(f"{r} {exact[(o, r)]} ({shares(exact, o, rooms)[r]:.2f})" for r in rooms))
    out = args.output or args.plan.with_name("typicality.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "object", "room", "instances", "share", "share_synonyms", "typicality",
                    "typicality_synonyms", "label_before"])
        for r in plan:
            s = shares(exact, r["object"], rooms)[r["room"]]
            s2 = shares(synonym, r["object"], rooms)[r["room"]]
            w.writerow([r["scenario"], r["object"], r["room"], exact[(r["object"], r["room"])], f"{s:.3f}",
                        f"{s2:.3f}", "typical" if s >= threshold else "atypical",
                        "typical" if s2 >= threshold else "atypical", r.get("typical", "")])
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
