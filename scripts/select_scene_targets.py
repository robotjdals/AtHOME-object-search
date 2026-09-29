"""Per-scene target categories, sampled from the open-vocabulary list.

Each scene gets up to ``--per-scene`` categories, sampled uniformly (seeded
by the scene ID) among the list's categories present in the scene: Train
scenes use seen categories only; Val/Test scenes use seen, synonym and
unseen categories (the three HM3D-OVON evaluation splits). Sampling keeps the
labeling cost bounded while the whole dataset covers the full list.
Writes the targets file used by the catalog, masking and Teacher stages
(same format as configs/data/pilot_targets*.json, plus provenance).
``target_members`` lists, per chosen target, the raw names of its subtype
categories (``hyponyms`` of the list, transitively), which count as the
target for ground truth and masking.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from build_target_categories import load_mapping, normalize

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-map", type=Path, required=True, help="scoped room-object map")
    parser.add_argument("--categories", type=Path, required=True, help="build_target_categories.py 출력")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--per-scene", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    categories = json.loads(args.categories.read_text(encoding="utf-8"))
    held_out = set(categories.get("synonyms", [])) | set(categories["unseen"])
    allowed = set(categories["seen"]) | (held_out if args.split != "train" else set())
    mapping = load_mapping(ROOT / categories["sources"]["mapping"]) if not Path(
        categories["sources"]["mapping"]).is_absolute() else load_mapping(Path(categories["sources"]["mapping"]))
    data = json.loads(args.scene_map.read_text(encoding="utf-8"))
    objects = data["objects"] if "objects" in data else [o for r in data["rooms"] for o in r["objects"]]
    present = sorted({mapping[normalize(o["semantic_tag"])][0] for o in objects
                      if normalize(o["semantic_tag"]) in mapping
                      and mapping[normalize(o["semantic_tag"])][0] in allowed})
    scene_id = args.scene_map.name.split(".")[0]
    rng = random.Random(f"{args.seed}:{scene_id}")
    chosen = sorted(rng.sample(present, min(args.per_scene, len(present))))
    aliases = {raw: c for c in chosen for raw in categories["categories"][c]["raw_names"] if raw != c}
    hyponyms = categories.get("hyponyms", {})

    def subtypes(c):
        out = set()
        for s in hyponyms.get(c, []):
            out |= {s} | subtypes(s)
        return out

    members = {c: sorted({raw for s in subtypes(c) for raw in categories["categories"][s]["raw_names"]})
               for c in chosen}
    members = {c: m for c, m in members.items() if m}
    result = {
        "version": categories["version"], "purpose": "per_scene_targets",
        "target_categories": chosen, "category_aliases": aliases, "target_members": members,
        "synonym_categories": sorted(set(chosen) & set(categories.get("synonyms", []))),
        "unseen_categories": sorted(set(chosen) & set(categories["unseen"])),
        "split": args.split, "per_scene": args.per_scene, "seed": args.seed,
        "present_in_scene": len(present),
        "categories_file": str(args.categories),
        "categories_sha256": hashlib.sha256(args.categories.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"{scene_id} ({args.split}): 장면에 있는 목록 범주 {len(present)}개 중 {len(chosen)}개 → {chosen}")


if __name__ == "__main__":
    main()
