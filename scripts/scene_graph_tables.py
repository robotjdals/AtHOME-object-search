"""Scene graph ablation table (TABLE I) from scripts/summarize_graph_compare.py
summaries, in the booktabs style of scripts/table_style.py.

    python scripts/summarize_graph_compare.py <inputs> --start-rule --step-budget 500 \
        --json outputs/graph_compare/valtest_summary_objectnav_500steps.json
    python scripts/scene_graph_tables.py outputs/graph_compare/valtest_summary_objectnav_500steps.json
"""
import argparse
import json
from pathlib import Path

from table_style import OUT, draw

ROWS = (("full/qwen", "Ours (full)"), ("wo_workspace/qwen", "w/o workspace layer"),
        ("object_list/qwen", "Object list (unstructured)"))
METRICS = (("success", "SR↑", max, 3), ("spl", "SPL↑", max, 3), ("visits", "Visits↓", min, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, default=OUT / "table1_ablation_main_llm_500steps.png")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    results = summary["results"]
    rows = [["Representation"] + [h for _, h, _, _ in METRICS]]
    rows += [[label] + [f"{results[key][m]:.{d}f}" for m, _, _, d in METRICS] for key, label in ROWS]
    bold = set()
    for c, (m, _, best, d) in enumerate(METRICS, 1):
        col = [round(results[key][m], d) for key, _ in ROWS]
        bold |= {(r, c) for r, v in enumerate(col, 1) if v == best(col)}
    notes = [f"HM3D val buildings (Val + Test), ObjectNav start rule (1–30 m): {summary['episodes']:,} episodes "
             f"in {summary['buildings']} scenes. Best in bold."]
    if summary.get("step_budget"):
        notes.append(f"Success: target seen within {summary['step_budget']} ObjectNav actions "
                     f"(D + 3N ≤ {summary['step_budget'] / 4:g} m). Visits: until found.")
    draw(rows, [2.9, 0.9, 0.9, 0.9], ["left"] + ["center"] * 3,
         "TABLE I. Ablation on search performance (Qwen3-4B, zero-shot).", notes, args.out, bold=bold)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
