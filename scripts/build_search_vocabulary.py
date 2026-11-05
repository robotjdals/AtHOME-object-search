"""One set of names from the command to the found check. Perception's fixed
detection list (household.txt) was written apart from the planner's target
categories, so it calls some of them otherwise (television / tv, cell phone /
phone) and a detection under that name is not the target.

A name is a synonym of a target category when their SentenceBERT cosine is
above the threshold the categories were split with (0.68, HM3D-OVON Table II;
the model of scripts/build_target_categories.py) and the dataset's own category
table does not put them in different classes (mpcat40: sofa is not a cushion,
0.684). Writes the list with those names replaced and the aliases for labels
already in the graph or the records:

    python scripts/build_search_vocabulary.py --household <household.txt> \
        --out configs/perception/vocabulary_search.txt --aliases configs/perception/label_aliases.json
"""
import argparse
import csv
import json
from pathlib import Path

from athome.scene_graph.vocabulary import Vocabulary
from build_target_categories import SBERT

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--household", type=Path, required=True)
    p.add_argument("--categories", type=Path, default=ROOT / "configs/data/target_categories.v5.json")
    p.add_argument("--threshold", type=float, default=0.68)
    p.add_argument("--mapping", type=Path, default=ROOT / "configs/data/hm3dsem_category_mappings.tsv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--aliases", type=Path, required=True)
    args = p.parse_args()
    from sentence_transformers import SentenceTransformer
    vocab = Vocabulary.load(args.categories)
    names = [l.strip() for l in args.household.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    model = SentenceTransformer(SBERT)
    outside = [n for n in names if vocab.canonical(n) not in vocab.categories]
    sim = model.encode(outside, normalize_embeddings=True) @ model.encode(vocab.categories, normalize_embeddings=True).T
    with args.mapping.open(encoding="utf-8", newline="") as f:
        table = {r["category"].casefold(): r["mpcat40"] for r in csv.DictReader(f, delimiter="\t")}
    aliases, kept = {}, {}
    for name, row in zip(outside, sim):
        best = int(row.argmax())
        a, b = table.get(name), table.get(vocab.categories[best])
        if row[best] > args.threshold and (a is None or b is None or a == b):
            aliases[name] = {"category": vocab.categories[best], "similarity": round(float(row[best]), 3)}
        else:
            kept[name] = {"nearest": vocab.categories[best], "similarity": round(float(row[best]), 3)}
    out = list(dict.fromkeys(aliases[n]["category"] if n in aliases else vocab.canonical(n) for n in names))
    args.out.write_text(f"# Search detection list: {args.household.name} with synonyms of target categories replaced "
                        f"(scripts/build_search_vocabulary.py, SentenceBERT > {args.threshold}).\n" + "\n".join(out) + "\n",
                        encoding="utf-8")
    args.aliases.write_text(json.dumps({"model": SBERT, "threshold": args.threshold, "aliases": aliases,
                                        "not_synonyms": kept}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("바꿈:", aliases)
    print("유지:", kept)


if __name__ == "__main__":
    main()
