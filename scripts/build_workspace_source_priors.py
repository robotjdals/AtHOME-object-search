"""How often each object category was chosen as a Workspace source, and its
function label, in the training scene graphs (Train buildings only).

The online object layer (athome.scene_graph.online) does not rerun the LLM
labeling (proposal 4-2) when new furniture appears; it decides from these
rates instead: a category the labeling chose as a source in most of its
training occurrences becomes a Workspace with its most frequent function
label (the majority decision, as the labels themselves are 9-sample
majorities).

    python scripts/build_workspace_source_priors.py
"""
import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from athome.scene_graph.query import normalize_category

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", default=str(ROOT / "outputs/hm3d_v5/*.workspace_graph.json"))
    parser.add_argument("--splits", type=Path, default=ROOT / "outputs/sft_v5/manifest.json",
                        help="scene_splits of the SFT export (Train buildings only)")
    parser.add_argument("--out", type=Path, default=ROOT / "configs/data/workspace_source_priors.json")
    args = parser.parse_args()
    splits = json.loads(args.splits.read_text(encoding="utf-8"))["scene_splits"]
    present, chosen, functions = Counter(), Counter(), defaultdict(Counter)
    scenes = 0
    for path in sorted(glob.glob(args.graphs)):
        if splits.get(Path(path).name.split(".")[0]) != "train":
            continue
        scenes += 1
        graph = json.loads(Path(path).read_text(encoding="utf-8"))
        objects = {o["object_id"]: o for o in graph["objects"]}
        for o in graph["objects"]:
            present[normalize_category(o["semantic_tag"])] += 1
        for w in graph["workspaces"]:
            category = normalize_category(objects[w["source_object_id"]]["semantic_tag"])
            chosen[category] += 1
            functions[category][w.get("function_label")] += 1
    categories = {
        c: {"present": present[c], "source": chosen[c], "rate": round(chosen[c] / present[c], 4),
            **({"function_label": functions[c].most_common(1)[0][0],
                "function_labels": dict(functions[c].most_common())} if chosen[c] else {})}
        for c in sorted(present)}
    out = {"source": "training workspace graphs (unmasked, Train buildings)", "graphs": args.graphs,
           "scenes": scenes, "categories": categories}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{scenes} scenes, {len(present)} categories, {len(chosen)} chosen as sources -> {args.out}")


if __name__ == "__main__":
    main()
