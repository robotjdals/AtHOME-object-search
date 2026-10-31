"""Training-result tables in the style of the scene graph tables (booktabs,
DejaVu Serif, English, SR / SPL / Visits on the ObjectNav start rule).
Episodes and runs are those of scripts/benchmark_report.py.

    python scripts/training_tables.py                      # success = found (60-visit cap)
    python scripts/training_tables.py --step-budget 500    # ObjectNav success: within 500 actions

TABLE III  object search on Val (Random, Greedy, Qwen3-4B zero-shot, Ours)
TABLE IV   distillation: Ours against the GPT-4.1 teacher (holdout episodes the teacher ran)
TABLE V    target categories on Val (HM3D-OVON seen / synonym / unseen)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_report as br                              # noqa: E402
from table_style import OUT, draw                          # noqa: E402

RULE = br.STANDARD
parser = argparse.ArgumentParser()
parser.add_argument("--step-budget", type=int,
                    help="ObjectNav 동작 상한 (Habitat: 500). 넘은 에피소드는 SR, SPL 0")
BUDGET = parser.parse_args().step_budget
# Habitat ObjectNav actions: forward 0.25 m, turn 30 deg. A visit's 360-degree
# look is 12 turns, so an episode takes D / 0.25 + 12 N = 4 (D + 3 N) actions
# (turns along the path not counted): the cost D + 3 N in units of 4 actions.
ACTIONS_PER_COST = 1 / 0.25
assert br.STEP_COST * ACTIONS_PER_COST == 12
SUFFIX = f"_{BUDGET}steps" if BUDGET else ""
BUDGET_NOTE = ([f"Success: target seen within {BUDGET} ObjectNav actions (forward 0.25 m, turn 30°; a visit's 360° look "
                f"= 12 turns), i.e. D + 3N ≤ {BUDGET / ACTIONS_PER_COST:g} m. Visits: until found."]
               if BUDGET else [])
NAMES = {"random": "Random", "nearest": "Greedy (nearest first)", "base": "Qwen3-4B (zero-shot)",
         "Teacher": "GPT-4.1 (teacher)", "SFT": "Ours (SFT)", "GRPO": "Ours (SFT + GRPO)"}
METRICS = (("SR", "SR↑", max, 3), ("SPL", "SPL↑", max, 3), ("visits", "Visits↓", min, 1))


def rule_ids(episodes, ids=None):
    ids = sorted(episodes) if ids is None else ids
    return [i for i in ids if RULE[0] <= episodes[i]["l_star"] < RULE[1]]


def values(run):
    """benchmark_report's per-episode values; beyond the action budget the
    episode is a failure (SR = SPL = 0)."""
    v = br.per_episode(run)
    if BUDGET and run["cost"] * ACTIONS_PER_COST > BUDGET:
        v["SR"] = v["SPL"] = 0.0
    return v


def means(episodes, ids, policy):
    vals = [values(episodes[i]["runs"][policy]) for i in ids]
    return {m: float(np.mean([v[m] for v in vals])) for m, *_ in METRICS}


def diff_ci(episodes, ids, a, b, metric):
    """Mean of b - a with the 95% building bootstrap interval."""
    boot = br.Bootstrap([episodes[i]["building"] for i in ids])
    d = np.array([values(episodes[i]["runs"][b])[metric] - values(episodes[i]["runs"][a])[metric]
                  for i in ids])
    return boot.mean(d)


def fmt_diff(mean_ci, digits):
    mean, (lo, hi) = mean_ci
    sig = "" if lo <= 0 <= hi else "*"
    return f"{mean:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]{sig}"


def table(rows_of, policies, groups=None):
    """rows: policy name + metrics per group; bold = best per column (ties included)."""
    groups = groups or [None]
    header = ["Method"] + [h for _ in groups for _, h, _, _ in METRICS]
    rows, bold = [header], set()
    values = {p: [rows_of(p, g) for g in range(len(groups))] for p in policies}
    for r, p in enumerate(policies, 1):
        row = [NAMES[p]]
        for g in range(len(groups)):
            row += [f"{values[p][g][m]:.{d}f}" for m, _, _, d in METRICS]
        rows.append(row)
    for g in range(len(groups)):
        for k, (m, _, best, d) in enumerate(METRICS):
            col = [round(values[p][g][m], d) for p in policies]
            bold |= {(r, 1 + 3 * g + k) for r, v in enumerate(col, 1) if v == best(col)}
    return rows, bold


# Decision latency (scripts/measure_planner_latency.py: recorded planner
# decisions, one request at a time, wall clock from this laptop).
lat = {name: json.loads((ROOT / f"outputs/paper/{name}.json").read_text())
       for name in ("planner_latency_n100", "planner_latency_n100_base")}
latency = {"base": lat["planner_latency_n100_base"]["student"], "Teacher": lat["planner_latency_n100"]["teacher"],
           "SFT": lat["planner_latency_n100"]["student"], "GRPO": lat["planner_latency_n100"]["student"]}


def with_latency(rows, bold, policies):
    """Latency column after the metrics ('–' for planners without an LLM); bold the fastest."""
    col = len(rows[0])
    out = [rows[0] + ["Latency (s)↓"]] + [row + [f"{latency[p]['median_s']:.2f}" if p in latency else "–"]
                                         for row, p in zip(rows[1:], policies)]
    fastest = min(round(latency[p]["median_s"], 2) for p in policies if p in latency)
    return out, bold | {(r, col) for r, p in enumerate(policies, 1)
                        if p in latency and round(latency[p]["median_s"], 2) == fastest}


# TABLE III: Val, ObjectNav start rule.
val = br.load_split("val", br.METHODS)
val_ids = rule_ids(val)
pol3 = ["random", "nearest", "base", "SFT", "GRPO"]
rows, bold = table(lambda p, g: means(val, val_ids, p), pol3)
sft_vs_base = diff_ci(val, val_ids, "base", "SFT", "visits")
grpo_vs_sft = {m: diff_ci(val, val_ids, "SFT", "GRPO", m) for m in ("visits", "SPL")}
n_build = len({val[i]["building"] for i in val_ids})
draw(rows, [2.9, 0.9, 0.9, 0.9], ["left"] + ["center"] * 3,
     "TABLE III. Object search on unseen buildings.",
     [f"HM3D val, ObjectNav start rule (1–30 m): {len(val_ids):,} episodes in {n_build} unseen scenes. Best in bold.",
      f"Δ visits, SFT vs zero-shot: {fmt_diff(sft_vs_base, 1)}; Δ SPL, +GRPO vs SFT: {fmt_diff(grpo_vs_sft['SPL'], 3)}.",
      "[95% scene-level bootstrap CI]; * excludes 0."] + BUDGET_NOTE,
     OUT / f"table3_search_val{SUFFIX}.png", bold=bold, rule_after=(2, 3))
rows_lat, bold_lat = with_latency(rows, bold, pol3)
draw(rows_lat, [2.9, 0.9, 0.9, 0.9, 1.15], ["left"] + ["center"] * 4,
     "TABLE III. Object search on unseen buildings.",
     [f"HM3D val, ObjectNav start rule (1–30 m): {len(val_ids):,} episodes in {n_build} unseen scenes. Best in bold.",
      f"Δ visits, SFT vs zero-shot: {fmt_diff(sft_vs_base, 1)}; Δ SPL, +GRPO vs SFT: {fmt_diff(grpo_vs_sft['SPL'], 3)}.",
      "[95% scene-level bootstrap CI]; * excludes 0.",
      f"Latency: median per LLM decision ({latency['SFT']['n']} decisions), Qwen3-4B on vLLM (A100 80GB, MIG 7g.80gb)."]
     + BUDGET_NOTE,
     OUT / f"table3_search_val_latency{SUFFIX}.png", bold=bold_lat, rule_after=(2, 3))

# TABLE IV: against the GPT-4.1 teacher on the holdout episodes it ran on.
hold = br.load_split("holdout", br.METHODS)
br.add_teacher(hold)
t_ids = rule_ids(hold, sorted(i for i, e in hold.items() if "Teacher" in e["runs"]))
pol4 = ["base", "Teacher", "SFT", "GRPO"]
rows, bold = table(lambda p, g: means(hold, t_ids, p), pol4)
vs_t = {p: {m: diff_ci(hold, t_ids, "Teacher", p, m) for m in ("visits", "SPL")} for p in ("SFT", "GRPO")}
n_build = len({hold[i]["building"] for i in t_ids})
draw(rows, [2.9, 0.9, 0.9, 0.9], ["left"] + ["center"] * 3,
     "TABLE IV. Distilled planner against the GPT-4.1 teacher.",
     [f"Held-out training scenes the teacher ran on, ObjectNav start rule: {len(t_ids)} episodes in {n_build} scenes.",
      f"Δ vs teacher, SFT: visits {fmt_diff(vs_t['SFT']['visits'], 2)}, SPL {fmt_diff(vs_t['SFT']['SPL'], 3)}.",
      f"Δ vs teacher, SFT + GRPO: visits {fmt_diff(vs_t['GRPO']['visits'], 2)}, SPL {fmt_diff(vs_t['GRPO']['SPL'], 3)}.",
      "[95% scene-level bootstrap CI]; * excludes 0. Best in bold."] + BUDGET_NOTE,
     OUT / f"table4_teacher{SUFFIX}.png", bold=bold, rule_after=(2,))

# TABLE IV with decision latency.
rows_lat, bold_lat = with_latency(rows, bold, pol4)
draw(rows_lat, [2.9, 0.9, 0.9, 0.9, 1.15], ["left"] + ["center"] * 4,
     "TABLE IV. Distilled planner against the GPT-4.1 teacher.",
     [f"Held-out training scenes the teacher ran on, ObjectNav start rule: {len(t_ids)} episodes in {n_build} scenes.",
      f"Δ vs teacher, SFT: visits {fmt_diff(vs_t['SFT']['visits'], 2)}, SPL {fmt_diff(vs_t['SFT']['SPL'], 3)}.",
      f"Δ vs teacher, SFT + GRPO: visits {fmt_diff(vs_t['GRPO']['visits'], 2)}, SPL {fmt_diff(vs_t['GRPO']['SPL'], 3)}.",
      "[95% scene-level bootstrap CI]; * excludes 0. Best in bold.",
      f"Latency: median per decision ({latency['SFT']['n']} decisions; teacher {latency['Teacher']['n']}),",
      "Qwen3-4B on vLLM (A100 80GB, MIG 7g.80gb); GPT-4.1 via the OpenAI API with its labeling output "
      f"(likelihood and reason per candidate, {lat['planner_latency_n100']['teacher']['mean_completion_tokens']:.0f} tokens)."]
     + BUDGET_NOTE,
     OUT / f"table4_teacher_latency{SUFFIX}.png", bold=bold_lat, rule_after=(2,))

# TABLE III + IV in one: Val next to the holdout episodes the teacher ran on
# (the teacher has no Val runs: it labelled training buildings only), plus latency.
pol34 = ["random", "nearest", "base", "Teacher", "SFT", "GRPO"]
parts = [(val, val_ids), (hold, t_ids)]
cells = {p: [None if p not in eps[ids[0]]["runs"] else means(eps, ids, p) for eps, ids in parts] for p in pol34}
rows = [["Method"] + [h for _ in parts for _, h, _, _ in METRICS]]
bold = set()
for r, p in enumerate(pol34, 1):
    rows.append([NAMES[p]] + [("–" if v is None else f"{v[m]:.{d}f}") for v in cells[p] for m, _, _, d in METRICS])
for g in range(len(parts)):
    for k, (m, _, best, d) in enumerate(METRICS):
        col = {r: round(cells[p][g][m], d) for r, p in enumerate(pol34, 1) if cells[p][g] is not None}
        bold |= {(r, 1 + 3 * g + k) for r, v in col.items() if v == best(col.values())}
rows, bold = with_latency(rows, bold, pol34)
draw(rows, [2.6] + [0.72] * 6 + [1.1], ["left"] + ["center"] * 7,
     "TABLE III. Object search: unseen buildings and the GPT-4.1 teacher.",
     [f"ObjectNav start rule (1–30 m). Unseen: HM3D val, {len(val_ids):,} episodes in "
      f"{len({val[i]['building'] for i in val_ids})} scenes. Held-out: training-split scenes left out of training, "
      f"{len(t_ids)} episodes in {len({hold[i]['building'] for i in t_ids})} scenes.",
      "The teacher ran only on training-split scenes (label generation), so it has no unseen-building result. Best in bold.",
      f"Latency: median per LLM decision ({latency['SFT']['n']} decisions; teacher {latency['Teacher']['n']}); "
      "Qwen3-4B on vLLM (A100 80GB), GPT-4.1 via the OpenAI API."] + BUDGET_NOTE,
     OUT / f"table3_search_teacher_latency{SUFFIX}.png",
     groups=[(f"Unseen buildings ({len(val_ids)})", 1, 3), (f"Held-out buildings ({len(t_ids)})", 4, 6)],
     bold=bold, rule_after=(2, 3))

# TABLE III with the teacher on unseen buildings: GPT-4.1 run as the planner
# (evaluate_student.py --teacher-openai) on 100 random Val start-rule episodes
# (seed 0, outputs/eval_teacher/val100_state_ids.txt), next to all 835.
parts_t = sorted((ROOT / "outputs/eval_teacher").glob("teacher_val100_part*.json"))
if len(parts_t) == 4:
    teacher_rows = {r["state_id"]: r for f in parts_t for r in json.loads(f.read_text())["rows"]}
    sub_ids = sorted(teacher_rows)
    for i in sub_ids:                 # same environment: the nearest-first run must agree
        assert abs(teacher_rows[i]["mincost"]["cost"] - val[i]["runs"]["nearest"]["cost"]) < 1e-6, i
        val[i]["runs"]["Teacher"] = teacher_rows[i]["student"]
    parts = [(val, val_ids), (val, sub_ids)]
    # Paper: every baseline; talk: the LLM planners only.
    for pols, name, rule in ((pol34, "", (2, 4)), (["base", "Teacher", "SFT", "GRPO"], "_talk", (2,))):
        cells = {p: [None if p not in eps[ids[0]]["runs"] else means(eps, ids, p) for eps, ids in parts] for p in pols}
        rows = [["Method"] + [h for _ in parts for _, h, _, _ in METRICS]]
        bold = set()
        for r, p in enumerate(pols, 1):
            rows.append([NAMES[p]] + [("–" if v is None else f"{v[m]:.{d}f}") for v in cells[p] for m, _, _, d in METRICS])
        for g in range(len(parts)):
            for k, (m, _, best, d) in enumerate(METRICS):
                col = {r: round(cells[p][g][m], d) for r, p in enumerate(pols, 1) if cells[p][g] is not None}
                bold |= {(r, 1 + 3 * g + k) for r, v in col.items() if v == best(col.values())}
        rows, bold = with_latency(rows, bold, pols)
        vs_teacher = {m: diff_ci(val, sub_ids, "Teacher", "SFT", m) for m in ("visits", "SPL")}
        fallbacks = sum(teacher_rows[i]["student"].get("policy_fallbacks", 0) for i in sub_ids)
        draw(rows, [2.6] + [0.72] * 6 + [1.1], ["left"] + ["center"] * 7,
             "TABLE III. Object search on unseen buildings.",
             [f"HM3D val, ObjectNav start rule (1–30 m): {len(val_ids):,} episodes in {len({val[i]['building'] for i in val_ids})} scenes not used for training; "
              f"the teacher ran on a random {len(sub_ids)} of them ({len({val[i]['building'] for i in sub_ids})} scenes).",
              f"SFT vs teacher on these: Δ visits {fmt_diff(vs_teacher['visits'], 1)}, Δ SPL {fmt_diff(vs_teacher['SPL'], 3)}. "
              "[95% scene-level bootstrap CI]; * excludes 0. Best in bold.",
              f"Latency: median per LLM decision ({latency['SFT']['n']} decisions; teacher {latency['Teacher']['n']}); "
              "Qwen3-4B on vLLM (A100 80GB), GPT-4.1 via the OpenAI API."] + BUDGET_NOTE,
             OUT / f"table3_search_teacher_val{name}{SUFFIX}.png",
             groups=[(f"All episodes ({len(val_ids)})", 1, 3), (f"With the teacher ({len(sub_ids)})", 4, 6)],
             bold=bold, rule_after=rule)
    print("teacher on Val:", len(sub_ids), "episodes, fallbacks", fallbacks)

# TABLE III with the teacher on every Val start-rule episode (the random 100
# plus the rest, outputs/eval_teacher/val_rest_state_ids.txt).
teacher_all = {r["state_id"]: r for f in sorted((ROOT / "outputs/eval_teacher").glob("teacher_val*_part*.json"))
               for r in json.loads(f.read_text())["rows"]}
if set(val_ids) <= set(teacher_all):
    for i in val_ids:
        assert abs(teacher_all[i]["mincost"]["cost"] - val[i]["runs"]["nearest"]["cost"]) < 1e-6, i
        val[i]["runs"]["Teacher"] = teacher_all[i]["student"]
    vs_t = {p: {m: diff_ci(val, val_ids, "Teacher", p, m) for m in ("visits", "SPL")} for p in ("SFT", "GRPO")}
    for pols, name, rule in ((pol34, "", (2, 4)), (["base", "Teacher", "SFT", "GRPO"], "_talk", (2,))):
        rows, bold = table(lambda p, g: means(val, val_ids, p), pols)
        rows, bold = with_latency(rows, bold, pols)
        draw(rows, [2.9, 0.9, 0.9, 0.9, 1.15], ["left"] + ["center"] * 4,
             "TABLE III. Object search on unseen buildings.",
             [f"HM3D val, ObjectNav start rule (1–30 m): {len(val_ids):,} episodes in "
              f"{len({val[i]['building'] for i in val_ids})} scenes not used for training. Best in bold.",
              f"Δ vs teacher, SFT: visits {fmt_diff(vs_t['SFT']['visits'], 2)}, SPL {fmt_diff(vs_t['SFT']['SPL'], 3)}; "
              f"SFT + GRPO: visits {fmt_diff(vs_t['GRPO']['visits'], 2)}, SPL {fmt_diff(vs_t['GRPO']['SPL'], 3)}.",
              "[95% scene-level bootstrap CI]; * excludes 0.",
              f"Latency: median per LLM decision ({latency['SFT']['n']} decisions; teacher {latency['Teacher']['n']}); "
              "Qwen3-4B on vLLM (A100 80GB), GPT-4.1 via the OpenAI API."] + BUDGET_NOTE,
             OUT / f"table3_search_teacher_all{name}{SUFFIX}.png", bold=bold, rule_after=rule)
    print("teacher on all Val:", len(val_ids), "episodes, fallbacks",
          sum(teacher_all[i]["student"].get("policy_fallbacks", 0) for i in val_ids))

# TABLE V: target categories (HM3D-OVON), Val, start rule.
cats = ("seen", "synonym", "unseen")
by_cat = {c: [i for i in val_ids if val[i]["target_split"] == c] for c in cats}
pol5 = ["base", "SFT", "GRPO"]
rows, bold = table(lambda p, g: means(val, by_cat[cats[g]], p), pol5, groups=cats)
draw(rows, [2.6] + [0.72] * 9, ["left"] + ["center"] * 9,
     "TABLE V. Generalization across target categories (HM3D-OVON splits).",
     [f"HM3D val, ObjectNav start rule. Seen: categories used for training; synonym: renamed seen categories; "
      "unseen: never a training target.",
      "Best in bold."] + BUDGET_NOTE,
     OUT / f"table5_categories{SUFFIX}.png",
     groups=[(f"{c.capitalize()} ({len(by_cat[c])})", 1 + 3 * g, 3 + 3 * g) for g, c in enumerate(cats)], bold=bold)
print("ok", len(val_ids), len(t_ids), {c: len(v) for c, v in by_cat.items()})

# TABLE V on Val + Test (both halves of the HM3D val buildings), every policy
# through the served planner (vLLM, as on the robot), for more unseen-category
# episodes than Val alone has.
SERVED = {"base": "outputs/eval_server/base_server_episodes_{split}_r1.0.json",
          "SFT": "outputs/eval_server/sft_server_episodes_{split}_r1.0.json",
          "GRPO": "outputs/eval_server/v1_server_episodes_{split}_r1.0.json"}
if all((ROOT / p.format(split=s)).exists() for p in SERVED.values() for s in ("val", "test")):
    both = {**br.load_split("val", SERVED), **br.load_split("test", SERVED)}
    both_ids = rule_ids(both)
    by_cat = {c: [i for i in both_ids if both[i]["target_split"] == c] for c in cats}
    rows, bold = table(lambda p, g: means(both, by_cat[cats[g]], p), pol5, groups=cats)
    unseen = by_cat["unseen"]
    gain = {m: diff_ci(both, unseen, "base", "SFT", m) for m in ("visits", "SPL")}
    draw(rows, [2.6] + [0.72] * 9, ["left"] + ["center"] * 9,
         "TABLE V. Generalization across target categories (HM3D-OVON splits).",
         [f"HM3D val buildings (Val + Test), ObjectNav start rule: {len(both_ids):,} episodes in "
          f"{len({both[i]['building'] for i in both_ids})} unseen scenes; planners served on vLLM as on the robot.",
          "Seen: categories used for training; synonym: renamed seen categories; unseen: never a training target "
          f"({len({i.split(':')[2] for i in unseen})} categories in {len({both[i]['building'] for i in unseen})} scenes).",
          f"Unseen, SFT vs zero-shot: Δ visits {fmt_diff(gain['visits'], 1)}, Δ SPL {fmt_diff(gain['SPL'], 3)}. "
          "[95% scene-level bootstrap CI]; * excludes 0. Best in bold."] + BUDGET_NOTE,
         OUT / f"table5_categories_valtest{SUFFIX}.png",
         groups=[(f"{c.capitalize()} ({len(by_cat[c])})", 1 + 3 * g, 3 + 3 * g) for g, c in enumerate(cats)], bold=bold)
    print("Val + Test", len(both_ids), {c: len(v) for c, v in by_cat.items()})
