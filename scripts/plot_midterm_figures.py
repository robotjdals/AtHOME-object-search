"""Figures for the midterm presentation, from the evaluation result files.

Colors: the reference categorical palette's first three slots (validated
all-pairs in light mode): SFT blue, GRPO orange, base model aqua; the
nearest-first and random baselines are reference series in gray. One measure
per figure (no second axis). Writes PNGs into --out.

    python scripts/plot_midterm_figures.py --out outputs/midterm/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
SFT, GRPO, BASE = "#2a78d6", "#eb6834", "#1baf7a"
NEAREST, RANDOM = "#9a9994", "#c4c3be"
STANDARD = (1.0, 30.0)          # ObjectNav start rule: 1 m <= l* < 30 m


def setup():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for family in ("Noto Sans CJK KR", "NanumSquare", "NanumGothic"):
        if family in names:
            plt.rcParams["font.family"] = family
            break
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 13,
                         "axes.spines.top": False, "axes.spines.right": False})


def rows(path):
    return {r["state_id"]: r for r in json.loads((ROOT / path).read_text(encoding="utf-8"))["rows"]}


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def standard(r):
    return STANDARD[0] <= r["oracle_distance_m"] < STANDARD[1]


def within(s, k):
    return float(s["found"] and s["visits"] <= k)


def note(fig, text):
    fig.text(0.01, 0.01, text, fontsize=10, color=MUTED, ha="left", va="bottom")


def bars(ax, labels, values, colors, fmt):
    y = range(len(labels))
    ax.barh(list(y), values, height=0.55, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for i, v in enumerate(values):
        ax.text(v, i, " " + fmt(v), va="center", ha="left", color=INK, fontsize=12)


def fig_sft_cost(out):
    """Search cost of every policy on holdout and Val (same episodes)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, split, title in ((axes[0], "holdout", "홀드아웃 (학습에서 뺀 7개 건물, 440개)"),
                             (axes[1], "val", "Val (처음 보는 24개 건물, 1,495개)")):
        base = rows(f"outputs/eval_student/base_episodes_{split}.json")
        sft = rows(f"outputs/eval_student/sft_v1_episodes_{split}.json")
        grpo = rows(f"outputs/eval_student/grpoA200_episodes_{split}.json")
        ids = sorted(sft)
        values = [mean(sft[i]["random"]["cost"] for i in ids), mean(sft[i]["mincost"]["cost"] for i in ids),
                  mean(base[i]["student"]["cost"] for i in ids), mean(sft[i]["student"]["cost"] for i in ids),
                  mean(grpo[i]["student"]["cost"] for i in ids)]
        bars(ax, ["무작위", "가장 가까운 곳부터", "기본 Qwen3-4B", "SFT", "SFT + GRPO (v1)"], values,
             [RANDOM, NEAREST, BASE, SFT, GRPO], lambda v: f"{v:.1f}")
        ax.set_title(title, fontsize=13, loc="left")
        ax.set_xlabel("탐색 비용  D + 3N  (낮을수록 좋음)")
        ax.set_xlim(0, max(values) * 1.18)
    fig.suptitle("SFT로 탐색 비용이 홀드아웃 45%, 처음 보는 건물 31% 줄었다", x=0.01, ha="left", fontsize=17, fontweight="bold")
    note(fig, "D = 이동거리(m), N = 방문 수. 같은 시작점·같은 환경에서 짝지어 평가. 학습 서버(HF) 평가.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out / "1_sft_cost.png", dpi=200)
    plt.close(fig)


def fig_budget(out):
    """Success within k visits on Val (ObjectNav start rule)."""
    base = rows("outputs/eval_student/base_episodes_val.json")
    sft = rows("outputs/eval_student/sft_v1_episodes_val.json")
    grpo = rows("outputs/eval_student/grpoA200_episodes_val.json")
    ids = [i for i in sorted(sft) if standard(sft[i])]
    ks = list(range(1, 11))
    series = [("무작위", RANDOM, [mean(within(sft[i]["random"], k) for i in ids) for k in ks], ":"),
              ("가장 가까운 곳부터", NEAREST, [mean(within(sft[i]["mincost"], k) for i in ids) for k in ks], "--"),
              ("기본 Qwen3-4B", BASE, [mean(within(base[i]["student"], k) for i in ids) for k in ks], "-"),
              ("SFT + GRPO (v1)", GRPO, [mean(within(grpo[i]["student"], k) for i in ids) for k in ks], "-"),
              ("SFT", SFT, [mean(within(sft[i]["student"], k) for i in ids) for k in ks], "-")]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for name, color, ys, style in series:
        ax.plot(ks, [100 * y for y in ys], style, color=color, linewidth=2.5, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name)
    for name, color, ys, _ in series:
        if name in ("SFT", "기본 Qwen3-4B", "가장 가까운 곳부터"):
            ax.annotate(f"{100 * ys[4]:.0f}%", (5, 100 * ys[4]), textcoords="offset points",
                        xytext=(-10, 10) if name == "SFT" else (-10, -20), ha="right", color=INK, fontsize=12)
    ax.set_xticks(ks)
    ax.set_xlabel("방문 예산 (회)")
    ax.set_ylabel("예산 안에 찾은 비율 (%)")
    ax.set_ylim(0, 100)
    ax.grid(color=GRID, linewidth=1)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("적은 방문으로 찾는 비율: 5회 안에 SFT 82%, 기본 모델 56%", loc="left", fontsize=16, fontweight="bold")
    note(fig, f"Val {len(ids)}개 에피소드(목표까지 1~30 m인 시작점, ObjectNav 기준). 방문 60회 한도에서는 모든 LLM 정책이 거의 100% 찾음.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "2_budget_success.png", dpi=200)
    plt.close(fig)


def fig_grpo_curve(out):
    """Holdout cost along GRPO training (checkpoints evaluated)."""
    files = {("A", 0): "sft_v1_episodes_holdout.json", ("A", 100): "grpo_run1_episodes_holdout.json",
             ("A", 200): "grpoA200_episodes_holdout.json", ("B", 0): "sft_v1_episodes_holdout.json",
             ("B", 100): "grpoB100_episodes_holdout.json", ("B", 140): "grpoB140_episodes_holdout.json",
             ("B", 200): "grpoB200_episodes_holdout.json"}
    cost = {key: mean(r["student"]["cost"] for r in rows(f"outputs/eval_student/{f}").values()) for key, f in files.items()}
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for run, color, label in (("A", GRPO, "A: 워크스페이스만 GRPO (제안서, v1)"), ("B", "#6250d6", "B: 방 + 워크스페이스 GRPO")):
        steps = sorted(s for r, s in cost if r == run)
        ax.plot(steps, [cost[(run, s)] for s in steps], color=color, linewidth=2.5, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        ax.annotate(f"{cost[(run, steps[-1])]:.2f}", (steps[-1], cost[(run, steps[-1])]), textcoords="offset points",
                    xytext=(-4, 12 if run == "A" else -18), color=INK, fontsize=11, ha="right")
    ax.annotate(f"{cost[('B', 100)]:.2f}\n(사후 관찰, Val에서는 확인 안 됨)", (100, cost[("B", 100)]), textcoords="offset points",
                xytext=(10, -28), color=INK2, fontsize=11)
    ax.axhline(cost[("A", 0)], color=SFT, linewidth=1.5, linestyle="--")
    ax.text(2, cost[("A", 0)] + 0.08, f"SFT {cost[('A', 0)]:.2f}", color=SFT, fontsize=11)
    ax.set_xlabel("GRPO 학습 스텝")
    ax.set_ylabel("홀드아웃 탐색 비용 (낮을수록 좋음)")
    ax.set_xticks([0, 100, 140, 200])
    ax.set_ylim(23.5, 27.5)
    ax.grid(color=GRID, linewidth=1)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("GRPO는 SFT 대비 탐색 비용을 유의하게 바꾸지 못했다", loc="left", fontsize=16, fontweight="bold")
    note(fig, "홀드아웃 440개. 차이의 95% 신뢰구간이 모두 0을 포함(A200: +0.02, -1.5 ~ +1.9). 비교는 미리 정한 200스텝 기준.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "3_grpo_curve.png", dpi=200)
    plt.close(fig)


def fig_detection(out):
    """Val cost as detection recall drops (same draws for every policy)."""
    recalls = ["1.0", "0.9", "0.8", "0.7"]
    v1 = [rows(f"outputs/eval_server/v1_server_episodes_val_r{r}.json") for r in recalls]
    sft = [rows(f"outputs/eval_server/sft_server_episodes_val_r{r}.json") for r in recalls]
    xs = [100 * float(r) for r in recalls]
    series = [("무작위", RANDOM, [mean(x["random"]["cost"] for x in d.values()) for d in v1], ":"),
              ("가장 가까운 곳부터", NEAREST, [mean(x["mincost"]["cost"] for x in d.values()) for d in v1], "--"),
              ("SFT", SFT, [mean(x["student"]["cost"] for x in d.values()) for d in sft], "-"),
              ("SFT + GRPO (v1)", GRPO, [mean(x["student"]["cost"] for x in d.values()) for d in v1], "-")]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for name, color, ys, style in series:
        ax.plot(xs, ys, style, color=color, linewidth=2.5, marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=2, label=name)
    v1c = series[3][2]
    for x, y in zip(xs, v1c):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -18), ha="center", color=INK, fontsize=11)
    ax.set_xticks(xs, [f"{int(x)}%" for x in xs])
    ax.invert_xaxis()
    ax.set_xlabel("인식 검출률 (방문마다 보이는 물체를 검출할 확률)")
    ax.set_ylabel("Val 탐색 비용 (낮을수록 좋음)")
    ax.set_ylim(0, 75)
    ax.grid(color=GRID, linewidth=1)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title("인식을 놓치면 비용이 늘지만, 학습된 플래너는 계속 가장 효율적", loc="left", fontsize=16, fontweight="bold")
    note(fig, "Val 1,495개, 로봇과 같은 vLLM 서버로 평가. 모든 정책이 같은 누락 판정을 받음. 성공률은 70%에서도 99% 이상.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "4_detection_recall.png", dpi=200)
    plt.close(fig)


def fig_miss(out):
    """Extra cost after an effective miss in a target room: stayed vs left (Val, recall 0.8)."""
    ref = rows("outputs/eval_server/v1_server_episodes_val_r1.0.json")
    runs = rows("outputs/eval_server/v1_server_episodes_val_r0.8.json")
    stayed, left = [], []
    for sid, r in runs.items():
        traj = r["student"].get("trajectory", [])
        k = next((j for j, v in enumerate(traj) if v["missed"] and not v["detected"]), None)
        if k is None or k + 1 >= len(traj) or not r["student"]["found"]:
            continue
        extra = r["student"]["cost"] - ref[sid]["student"]["cost"]
        (left if traj[k + 1]["room_id"] != traj[k]["room_id"] else stayed).append(extra)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    bars(ax, [f"같은 방에 남음 ({len(stayed)}개)", f"바로 다른 방으로 떠남 ({len(left)}개)"],
         [mean(stayed), mean(left)], [GRPO, GRPO], lambda v: f"+{v:.1f}")
    ax.set_xlabel("놓치지 않았을 때 대비 늘어난 탐색 비용 (평균)")
    ax.set_xlim(0, mean(left) * 1.2)
    share = len(left) / (len(left) + len(stayed))
    ax.set_title(f"정답 방에서 물체를 놓치면 {share:.0%}가 바로 떠나고, 그때 비용이 크게 는다", loc="left", fontsize=16,
                 fontweight="bold")
    note(fig, "Val, 검출률 80%, v1. 정답 방에서 목표를 하나도 검출하지 못한 첫 방문 기준. 인식이 완벽하다고 가정하고 학습한 결과.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "5_miss_behavior.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/midterm/figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup()
    for fig in (fig_sft_cost, fig_budget, fig_grpo_curve, fig_detection, fig_miss):
        fig(args.out)
        print("wrote", fig.__name__)


if __name__ == "__main__":
    main()
