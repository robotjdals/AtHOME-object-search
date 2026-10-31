"""Booktabs-style table images for slides and papers (one font, rules above
and below the header and at the bottom, notes under the table)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path.home() / "Downloads"
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


def draw(rows, widths, aligns, title, notes, out, groups=None, bold=(), height_row=0.34, rule_after=()):
    """rows[0] = header, rows[1:] = body; widths in inches per column.
    groups: [(label, first_col, last_col)] spanning header cells above rows[0].
    bold: {(row, col)}; rule_after: body rows followed by a thin rule."""
    total = sum(widths)
    n = len(rows) + (1 if groups else 0)
    fig_h = height_row * (n + 1.4) + 0.32 * len(notes) + 0.5
    fig = plt.figure(figsize=(total, fig_h), dpi=250)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, total)
    ax.set_ylim(fig_h, 0)
    ax.axis("off")
    x = [sum(widths[:i]) for i in range(len(widths) + 1)]
    y = 0.15
    ax.text(0.05, y + 0.18, title, fontsize=11, va="center", ha="left")
    y += 0.42
    ax.plot([0.05, total - 0.05], [y, y], color="black", lw=1.4)
    if groups:
        for label, a, b in groups:
            ax.text((x[a] + x[b + 1]) / 2, y + height_row / 2, label, fontsize=10, ha="center", va="center")
            ax.plot([x[a] + 0.08, x[b + 1] - 0.08], [y + height_row - 0.04] * 2, color="black", lw=0.6)
        y += height_row
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            px = {"left": x[c] + 0.1, "center": (x[c] + x[c + 1]) / 2, "right": x[c + 1] - 0.1}[aligns[c]]
            ax.text(px, y + height_row / 2, cell, fontsize=10, ha=aligns[c], va="center",
                    fontweight="bold" if (r, c) in bold else "normal")
        y += height_row
        if r == 0:
            ax.plot([0.05, total - 0.05], [y, y], color="black", lw=0.8)
        elif r in rule_after:
            ax.plot([0.05, total - 0.05], [y, y], color="black", lw=0.5)
    ax.plot([0.05, total - 0.05], [y, y], color="black", lw=1.4)
    y += 0.15
    for note in notes:
        ax.text(0.05, y + 0.16, note, fontsize=8.3, ha="left", va="center", color="#333")
        y += 0.32
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
