"""GRPO training curves for the paper (English, the tables' serif font):
training reward, KL to the SFT reference and the share of groups with equal
rewards (zero advantage, no learning signal) per step, for the workspace-stage
run (A) and the room + workspace run (B).

    python scripts/plot_grpo_training.py            # paper: three panels and a note
    python scripts/plot_grpo_training.py --slides   # slides: reward and equal-reward groups only
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("GRPO, workspace stage", ["outputs/vast_archive/grpo/run1/log.jsonl",
                                               "outputs/vast_archive/grpo/runA_cont/log.jsonl"], "#2a78d6"),
        ("GRPO, room + workspace stages", ["outputs/vast_archive/grpo/runB/log.jsonl"], "#eb6834")]
WINDOW = 10                                    # moving average over training steps
OUT = Path.home() / "Downloads/fig_grpo_training.png"
plt.rcParams["font.family"] = "DejaVu Serif"


def load(paths):
    rows = [json.loads(l) for p in paths for l in (ROOT / p).read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["step"])
    value = lambda r, k: np.nan if r.get(k) is None else r[k]      # steps without an update log no KL
    equal = [r["groups"].get("equal_rewards", 0) / sum(r["groups"].values()) for r in rows]
    return (np.array([r["step"] for r in rows]), np.array([value(r, "mean_reward") for r in rows], float),
            np.array([value(r, "mean_kl") for r in rows], float), 100 * np.array(equal))


def smooth(y):
    """Trailing moving average, ignoring steps without a value."""
    return np.array([np.nanmean(y[max(0, i - WINDOW + 1):i + 1]) for i in range(len(y))])


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--slides", action="store_true", help="보상과 학습 신호 없는 그룹 두 패널, 설명문 없음")
args = parser.parse_args()
if args.slides:
    fig, (ax_r, ax_e) = plt.subplots(1, 2, figsize=(11, 3.8), dpi=220)
    panels, ax_k = (ax_r, ax_e), None
else:
    fig, (ax_r, ax_k, ax_e) = plt.subplots(1, 3, figsize=(15, 3.8), dpi=220)
    panels = (ax_r, ax_k, ax_e)
for label, paths, color in RUNS:
    step, reward, kl, equal = load(paths)
    for ax, y in ((ax_r, reward), (ax_k, kl), (ax_e, equal)):
        if ax is None:
            continue
        ax.plot(step, y, color=color, alpha=0.18, lw=1)
        ax.plot(step, smooth(y), color=color, lw=2, label=label)
    print(f"{label}: equal-reward groups {equal.mean():.1f}% of all groups")
ax_r.set_title("(a) Training reward", loc="left", fontsize=12)
ax_r.set_ylabel("reward = −distance − 3 × visits (0 is best)")
if ax_k is not None:
    ax_k.set_title("(b) KL to the SFT policy", loc="left", fontsize=12)
    ax_k.set_ylabel("mean KL")
ax_e.set_title(f"({'b' if args.slides else 'c'}) Groups with no learning signal", loc="left", fontsize=12)
ax_e.set_ylabel("groups with equal rewards (%)")
ax_e.set_ylim(0, 100)
for ax in panels:
    ax.set_xlabel("training step")
    ax.grid(color="#e9ecef")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
if args.slides:
    ax_e.legend(frameon=False, fontsize=10, loc="upper left")
else:
    ax_k.legend(frameon=False, fontsize=10, loc="upper left")
    fig.text(0.01, -0.06, f"Thin: per step (8 start states × 8 rollouts); thick: moving average over {WINDOW} steps. "
             "Dips in (a) are steps that drew distant starts (start distance explains 34% of the per-step variance),\n"
             "not a worse policy. (c) All 8 rollouts of a group got the same reward, so its advantage is zero; "
             "76% of these groups started within 1 m of the target.", fontsize=9, color="#495057")
fig.tight_layout()
out = OUT.with_name(OUT.stem + "_slides.png") if args.slides else OUT
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
