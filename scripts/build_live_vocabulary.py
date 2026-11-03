"""Base vocabulary of the real-time detector for one environment, in the
pre-map's own names:

    python scripts/build_live_vocabulary.py --premap <objects.json> --out configs/perception/vocabulary_<env>.txt

= the labels of the pre-mapped objects (the scene graph's objects), so that
furniture seen live gets the name the map has and the online object layer
recognises it, + the household objects of perception's list that are
training target categories (portable objects: what can appear in a room
after the pre-map, and what the online layer remembers for later commands).
The command's targets are added per command by the node.

A map built with one name set and updated with another breaks
category-gated association: the generic base list (Workspace source
categories) named the 412 desk "tv stand", "chest of drawers", "kitchen top",
and the online layer added them as new Workspaces. Systems that associate
within a category (Stretch AI instance memory, Hydra) map and update with
one detector and one vocabulary.
"""
import argparse
import json
from pathlib import Path

from athome.scene_graph.vocabulary import Vocabulary

ROOT = Path(__file__).resolve().parents[1]
HOUSEHOLD = ROOT / "data/perception/search412_20261030_zip/search412_dataset_20261030/vocabulary/household.txt"
EXCLUDED = {"other item"}            # perception's tag for objects without a usable label


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--premap", type=Path, required=True, help="perception's objects.json (the scene graph's objects)")
    p.add_argument("--household", type=Path, default=HOUSEHOLD, help="perception's household.txt")
    p.add_argument("--target-categories", type=Path, default=ROOT / "configs/data/target_categories.v5.json")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    vocab = Vocabulary.load(args.target_categories)
    targets = set(vocab.categories)
    premap = sorted({o["semantic_tag"].strip().lower() for o in json.loads(args.premap.read_text())["objects"]} - EXCLUDED)
    household = [l.strip().lower() for l in args.household.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
    portable = [n for n in household if vocab.canonical(n) in targets]
    names = list(dict.fromkeys(premap + portable))
    args.out.write_text(
        f"# Real-time perception base vocabulary (scripts/build_live_vocabulary.py): pre-map labels of\n"
        f"# {args.premap.parent.name} ({len(premap)}) + household objects that are training target\n"
        f"# categories ({len(portable)}); {len(names)} names.\n" + "\n".join(names) + "\n", encoding="utf-8")
    print(f"{len(names)} names ({len(premap)} pre-map, {len(portable)} portable) -> {args.out}")


if __name__ == "__main__":
    main()
