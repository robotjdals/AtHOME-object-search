"""Paper tables from scripts/benchmark_report.py output: LaTeX (booktabs),
Markdown, and PNG renderings for slides.

Table 1  main results on Val (the official HM3D val buildings, used for no
         training or selection); best value per column in bold.
Table 2  Teacher (GPT-4.1) comparison on the holdout episodes the Teacher ran
         on (the Teacher ran on train-split buildings only), with the share
         of the Teacher's gain over random each policy recovers and the
         paired differences to the Teacher.
Table 3  target categories on Val (seen / synonym / unseen, HM3D-OVON).

    python scripts/paper_tables.py outputs/eval_report/benchmark.json --out outputs/paper
"""
import argparse
import json
from pathlib import Path

SPLIT = "val"
ROWS = (("random", "Random", "무작위"), ("nearest", "Nearest-first", "가장 가까운 곳부터"),
        ("base", "Qwen3-4B (zero-shot)", "Qwen3-4B (학습 전)"),
        ("SFT", "Ours: SFT", "Ours: SFT"), ("GRPO", "Ours: SFT + GRPO", "Ours: SFT + GRPO"))
COLUMNS = (("SR", "SR", True, 100, 1), ("SPL", "SPL", True, 100, 1), ("SR@5", "SR@5", True, 100, 1),
           ("AUC-E", "AUC-E", True, 100, 1), ("cost", "Cost", False, 1, 1))
CATEGORIES = ("seen", "synonym", "unseen")


def label(key, lang):
    for k, en, ko in ROWS + (("Teacher", "GPT-4.1 (Teacher)", "GPT-4.1 (Teacher)"),):
        if k == key:
            return en if lang == "en" else ko
    return key


def main_table(report):
    """[(policy, [values])], the best value per column, and the episode counts."""
    summary = report[SPLIT]["subsets"]["all"]["summary"]
    table = [(key, [round(summary["policies"][key][m]["mean"] * scale, digits) for m, _, _, scale, digits in COLUMNS])
             for key, _, _ in ROWS]
    best = [(max if higher else min)(values[c] for _, values in table)
            for c, (_, _, higher, _, _) in enumerate(COLUMNS)]
    return table, best, summary


def category_table(report):
    b = report[SPLIT]["subsets"]["all"]["breakdown"]
    rows = [(key, [round(b[c][key][m] * (1 if m == "cost" else 100), 1) for c in CATEGORIES for m in ("cost", "SPL")])
            for key in ("base", "SFT", "GRPO")]
    return {c: b[c]["episodes"] for c in CATEGORIES}, rows


def teacher_table(report):
    block = report["holdout"]["subsets"]["teacher_episodes"]
    pol = block["summary"]["policies"]
    random_cost, teacher_cost = pol["random"]["cost"]["mean"], pol["Teacher"]["cost"]["mean"]
    rows = []
    for key in ("random", "base", "Teacher", "SFT", "GRPO"):
        r = pol[key]
        recovered = 100 * (random_cost - r["cost"]["mean"]) / (random_cost - teacher_cost)
        rows.append((key, [round(r["cost"]["mean"], 1), round(100 * r["SPL"]["mean"], 1),
                           round(100 * r["SR@5"]["mean"], 1), round(recovered, 1)]))
    diffs = {c["pair"].split(" - ")[0]: c for c in block["comparisons"]}      # policy - Teacher
    return block["summary"], rows, diffs


def signed(x, digits):
    text = f"{x:+.{digits}f}"
    return text.replace("-", "+") if float(text) == 0 else text      # no "-0.0"


def ci(e, scale=1, digits=1):
    lo, hi = e["ci95_buildings"]
    return signed(e["mean_diff"] * scale, digits), f"[{signed(lo * scale, digits)}, {signed(hi * scale, digits)}]"


def pair_name(pair, lang):
    b, a = pair.split(" - ")
    return f"{label(b, lang)} vs {label(a, lang)}"


def latex(report):
    table, best, summary = main_table(report)
    up, down = r"$\uparrow$", r"$\downarrow$"
    head = " & ".join(f"{name}{up if hi else down}" for _, name, hi, _, _ in COLUMNS)
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{Object search on the HM3DSem val buildings (%d episodes in %d buildings unseen in training). "
           r"SR, SPL, SR@5 (success within 5 visits) and AUC-E in \%%; Cost $= D + 3N$ (m). Same episodes and "
           r"candidate orders for every method.}" % (summary["episodes"], summary["buildings"]),
           r"\label{tab:main}", r"\begin{tabular}{l" + "c" * len(COLUMNS) + "}", r"\toprule",
           f"Method & {head} \\\\", r"\midrule"]
    for key, values in table:
        if key == "SFT":
            out.append(r"\midrule")
        cells = [f"\\textbf{{{v:.1f}}}" if v == best[c] else f"{v:.1f}" for c, v in enumerate(values)]
        out.append(f"{label(key, 'en')} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    s, rows, diffs = teacher_table(report)
    out += [r"\begin{table}[t]", r"\centering",
            r"\caption{Comparison with the GPT-4.1 Teacher on the %d holdout episodes (%d buildings held out from "
            r"training) the Teacher ran on; not the val episodes of Table~\ref{tab:main}. Recovered: share of the "
            r"Teacher's cost reduction over random. SFT $-$ Teacher cost: %s %s m (95\%% CI over buildings).}"
            % (s["episodes"], s["buildings"], *ci(diffs["SFT"]["cost"])),
            r"\label{tab:teacher}", r"\begin{tabular}{lcccc}", r"\toprule",
            r"Method & Cost$\downarrow$ & SPL$\uparrow$ & SR@5$\uparrow$ & Recovered (\%) \\", r"\midrule"]
    for key, v in rows:
        out.append(f"{label(key, 'en')} & " + " & ".join(f"{x:.1f}" for x in v) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    counts, rows = category_table(report)
    out += [r"\begin{table}[t]", r"\centering",
            r"\caption{Target categories on val (HM3D-OVON split): used as targets in training, synonyms of those, "
            r"and categories never used as targets. Cost (m) and SPL (\%).}", r"\label{tab:categories}",
            r"\begin{tabular}{lcccccc}", r"\toprule",
            r" & \multicolumn{2}{c}{Seen (%d)} & \multicolumn{2}{c}{Synonym (%d)} & \multicolumn{2}{c}{Unseen (%d)} \\"
            % tuple(counts[c] for c in CATEGORIES),
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r"Method & Cost$\downarrow$ & SPL$\uparrow$ & Cost$\downarrow$ & SPL$\uparrow$ & Cost$\downarrow$ & SPL$\uparrow$ \\",
            r"\midrule"]
    for key, v in rows:
        out.append(f"{label(key, 'en')} & " + " & ".join(f"{x:.1f}" for x in v) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    return "\n".join(out)


def markdown(report, lang="ko"):
    table, best, summary = main_table(report)
    names = [f"{name}{'↑' if hi else '↓'}" for _, name, hi, _, _ in COLUMNS]
    out = [f"**표 1. 주요 결과** (Val: 학습에 쓰지 않은 {summary['buildings']}개 건물, {summary['episodes']}개 에피소드. "
           "SR·SPL·SR@5·AUC-E는 %, Cost = D + 3N m)", "",
           "| 방법 | " + " | ".join(names) + " |", "|---" * (1 + len(COLUMNS)) + "|"]
    for key, values in table:
        cells = [f"**{v:.1f}**" if v == best[c] else f"{v:.1f}" for c, v in enumerate(values)]
        out.append(f"| {label(key, lang)} | " + " | ".join(cells) + " |")
    s, rows, diffs = teacher_table(report)
    lo_hi = ci(diffs["SFT"]["cost"])
    out += ["", f"**표 2. Teacher(GPT-4.1) 비교** (Teacher가 실행된 홀드아웃 {s['episodes']}개 에피소드, {s['buildings']}개 건물. "
            "표 1의 Val과 다른 에피소드. 회복률 = 무작위 대비 Teacher 비용 개선분 중 재현한 비율. "
            f"SFT - Teacher 비용 {lo_hi[0]} {lo_hi[1]} m)", "",
            "| 방법 | Cost↓ | SPL↑ | SR@5↑ | 회복률(%) |", "|---|---|---|---|---|"]
    for key, v in rows:
        out.append(f"| {label(key, lang)} | " + " | ".join(f"{x:.1f}" for x in v) + " |")
    counts, rows = category_table(report)
    out += ["", "**표 3. 목표 범주별 일반화** (Val, HM3D-OVON 분할. Cost m, SPL %)", "",
            f"| 방법 | 학습 범주({counts['seen']}) Cost | SPL | 동의어({counts['synonym']}) Cost | SPL | "
            f"학습하지 않은 범주({counts['unseen']}) Cost | SPL |", "|---|---|---|---|---|---|---|"]
    for key, v in rows:
        out.append(f"| {label(key, lang)} | " + " | ".join(f"{x:.1f}" for x in v) + " |")
    return "\n".join(out) + "\n"


def render(path, title, header, rows, bold=(), group_rule=None, note=None, widths=None):
    """One table as a PNG (booktabs look: rules above, below the header and at the end)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_midterm_figures import INK, INK2, MUTED, SURFACE, setup
    setup()
    ncol = len(header[-1])
    widths = widths or [2.6] + [1.0] * (ncol - 1)
    total = sum(widths)
    notes = note.split("\n") if note else []
    height = 0.42 * (len(rows) + len(header)) + (0.5 if title else 0) + 0.3 * len(notes) + 0.35
    fig = plt.figure(figsize=(total, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, total)
    ax.set_ylim(height, 0)
    ax.axis("off")
    y = 0.2
    if title:
        ax.text(0.1, y + 0.2, title, fontsize=13, fontweight="bold", color=INK, va="center")
        y += 0.5
    xs = [sum(widths[:k]) for k in range(ncol)]

    def rule(at, lw):
        ax.plot([0.05, total - 0.05], [at, at], color=INK, linewidth=lw)
    rule(y, 1.4)
    for line in header:
        col = 0
        for cell in line:
            text, span = cell if isinstance(cell, tuple) else (cell, 1)
            x0, x1 = xs[col], xs[col] + sum(widths[col:col + span])
            ax.text((x0 + x1) / 2 if col else 0.15, y + 0.21, text, ha="center" if col else "left", va="center",
                    fontsize=11, color=INK2)
            if span > 1:
                ax.plot([x0 + 0.08, x1 - 0.08], [y + 0.4, y + 0.4], color=MUTED, linewidth=0.8)
            col += span
        y += 0.42
    rule(y, 0.8)
    for r, row in enumerate(rows):
        if group_rule is not None and r == group_rule:
            ax.plot([0.05, total - 0.05], [y, y], color=MUTED, linewidth=0.6)
        for c, cell in enumerate(row):
            ax.text(xs[c] + widths[c] / 2 if c else 0.15, y + 0.21, cell, ha="center" if c else "left", va="center",
                    fontsize=11, color=INK, fontweight="bold" if (r, c) in bold else "normal")
        y += 0.42
    rule(y, 1.4)
    for k, line in enumerate(notes):
        ax.text(0.1, y + 0.25 + 0.3 * k, line, fontsize=9, color=MUTED, va="center")
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def pngs(report, out):
    table, best, summary = main_table(report)
    names = [f"{name}{'↑' if hi else '↓'}" for _, name, hi, _, _ in COLUMNS]
    rows = [[label(key, "ko"), *(f"{v:.1f}" for v in values)] for key, values in table]
    bold = {(r, c + 1) for r, (_, values) in enumerate(table) for c, v in enumerate(values) if v == best[c]}
    render(out / "table1_main.png",
           f"표 1. HM3DSem 물체 탐색 결과 (Val: 학습에 쓰지 않은 {summary['buildings']}개 건물, {summary['episodes']}개 에피소드)",
           [["방법", *names]], rows, bold, group_rule=[k for k, _ in table].index("SFT"),
           note="SR·SPL·SR@5(5회 방문 안 성공)·AUC-E(효율 곡선 넓이)는 %, Cost = 이동거리 + 3 × 방문 수 (m). 모든 방법이 같은 에피소드·같은 후보 순서.",
           widths=[2.8] + [1.25] * len(COLUMNS))
    s, trows, diffs = teacher_table(report)
    rows = [[label(key, "ko"), *(f"{x:.1f}" for x in v)] for key, v in trows]
    dc, dci = ci(diffs["SFT"]["cost"])
    gc, gci = ci(diffs["GRPO"]["cost"])
    render(out / "table2_teacher.png",
           f"표 2. Teacher(GPT-4.1) 비교: 홀드아웃 {s['episodes']}개 에피소드 ({s['buildings']}개 건물)",
           [["방법", "Cost↓", "SPL↑", "SR@5↑", "회복률 (%)"]], rows, group_rule=3,
           note=f"표 1(Val)과 다른 에피소드: Teacher는 학습 분할 건물에서만 실행. 회복률 = 무작위 대비 Teacher 비용 개선분 중 재현한 비율.\n"
                f"Teacher 대비 비용 차: SFT {dc} {dci}, SFT+GRPO {gc} {gci} (95% CI, 차이 없음).",
           widths=[2.8, 1.2, 1.2, 1.2, 1.4])
    counts, crows = category_table(report)
    rows = [[label(k, "ko"), *(f"{x:.1f}" for x in v)] for k, v in crows]
    bold = set()
    for c in range(1, 7):
        vals = [float(r[c]) for r in rows]
        target = min(vals) if c % 2 == 1 else max(vals)
        bold |= {(r, c) for r in range(len(rows)) if float(rows[r][c]) == target}
    render(out / "table3_categories.png", "표 3. 목표 범주별 일반화 (Val, HM3D-OVON 분할)",
           [["", (f"학습 범주 ({counts['seen']})", 2), (f"동의어 ({counts['synonym']})", 2),
             (f"학습하지 않은 범주 ({counts['unseen']})", 2)],
            ["방법", "Cost↓", "SPL↑", "Cost↓", "SPL↑", "Cost↓", "SPL↑"]], rows, bold,
           note="Cost m, SPL %. 학습하지 않은 범주: 학습 때 한 번도 목표로 쓰지 않은 물체 종류(10종, 각 1~2개 건물).",
           widths=[2.6] + [1.05] * 6)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path, help="benchmark_report.py 출력 JSON (val과 holdout)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables.tex").write_text(latex(report))
    (args.out / "tables.md").write_text(markdown(report))
    pngs(report, args.out)
    print("wrote", args.out / "tables.tex", args.out / "tables.md", "and table1-3 PNGs")


if __name__ == "__main__":
    main()
