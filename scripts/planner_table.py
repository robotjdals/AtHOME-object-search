"""Distillation table of the paper (Val buildings, ObjectNav start rule) from the stored episode files.

Rows: GPT-4.1 Teacher (outputs/eval_teacher), base Qwen3-4B, SFT and SFT + GRPO (outputs/eval_student).
Episodes: 1 m <= shortest path l* <= 30 m (835 of 1,495). Success = found with cost D + 3N <= 125 m, a budget modeled on
ObjectNav's 500 actions (0.25 m per step, 12 turns per look-around; turns along the path are not counted); SPL is 0 for
a failure. Intervals: 95 % percentile bootstrap over buildings (10,000 resamples).
The base / SFT / SFT + GRPO rows are also given on all 1,226 episodes (Val 835 + Test 391, vLLM server runs in
outputs/eval_server); the Teacher was not run on Test. No GPU and no API call.

    python scripts/planner_table.py [--output outputs/paper/planner_table.json]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
STEP_COST, COST_LIMIT, STANDARD, N_BOOT = 3.0, 125.0, (1.0, 30.0), 10_000
STUDENTS = {"base": "base", "SFT": "sft_v1", "SFT+GRPO": "grpoA200"}
SERVER = {"base": ("base_server_episodes_val835", "base_server_episodes_test391"),
          "SFT": ("sft_server_episodes_val_r1.0", "sft_server_episodes_test391"),
          "SFT+GRPO": ("v1_server_episodes_val_r1.0", "v1_server_episodes_test391")}


def success(run):
    """Found within the cost budget (D + 3N <= 125 m)."""
    return bool(run["found"]) and run["distance_m"] + STEP_COST * run["visits"] <= COST_LIMIT


def metrics(run):
    ok = success(run)
    return {"SR": float(ok), "SPL": run["spl"] if ok else 0.0, "visits": float(run["visits"])}


def load():
    """{state_id: {"l_star", "building", policy: run}} for the episodes every policy has."""
    students = {name: {r["state_id"]: r for r in json.loads((ROOT / f"outputs/eval_student/{f}_episodes_val.json").read_text())["rows"]}
                for name, f in STUDENTS.items()}
    teacher = {}
    for path in glob.glob(str(ROOT / "outputs/eval_teacher/teacher_val*_part*.json")):
        teacher.update({r["state_id"]: r for r in json.loads(Path(path).read_text())["rows"]})
    ids = sorted(set(teacher).intersection(*students.values()))
    episodes = {}
    for i in ids:
        r = students["SFT"][i]
        if STANDARD[0] <= r["oracle_distance_m"] <= STANDARD[1]:
            episodes[i] = {"building": i.split(":")[0], "GPT-4.1": teacher[i]["student"],
                           **{n: students[n][i]["student"] for n in STUDENTS}}
    return episodes


def load_server():
    """Same episode dict for the server runs on Val + Test (1,226 start-rule episodes)."""
    runs = {name: {r["state_id"]: r for f in files for r in json.loads((ROOT / f"outputs/eval_server/{f}.json").read_text())["rows"]}
            for name, files in SERVER.items()}
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    return {i: {"building": i.split(":")[0], **{n: runs[n][i]["student"] for n in SERVER}}
            for i in ids if STANDARD[0] <= runs["SFT"][i]["oracle_distance_m"] <= STANDARD[1]}


def report(episodes, policies, pairs):
    return {"episodes": len(episodes), "buildings": len({e["building"] for e in episodes.values()}),
            "rows": {p: summarize(episodes, p) for p in policies},
            "differences": {f"{a} - {b} {k}": difference(episodes, a, b, k) for a, b in pairs for k in ("visits", "SPL")},
            "base_vs_deployed": paired(episodes, "base", "SFT+GRPO")}


def summarize(episodes, policy):
    rows = [metrics(e[policy]) for e in episodes.values()]
    return {k: float(np.mean([r[k] for r in rows])) for k in ("SR", "SPL", "visits")}


def difference(episodes, a, b, key, seed=0):
    """Mean of (a - b) with a building bootstrap interval."""
    by_building = {}
    for e in episodes.values():
        by_building.setdefault(e["building"], []).append(metrics(e[a])[key] - metrics(e[b])[key])
    sums = np.array([sum(v) for v in by_building.values()])
    counts = np.array([len(v) for v in by_building.values()])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(sums), size=(N_BOOT, len(sums)))
    boot = sums[pick].sum(1) / counts[pick].sum(1)
    return {"mean": float(sums.sum() / counts.sum()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def paired(episodes, a, b):
    """Trained (b) against untrained (a): medians and the Wilcoxon signed-rank test (zeros dropped by Pratt)."""
    out = {}
    for key in ("visits", "SPL"):
        x = np.array([metrics(e[a])[key] for e in episodes.values()])
        y = np.array([metrics(e[b])[key] for e in episodes.values()])
        better = int((y < x).sum() if key == "visits" else (y > x).sum())
        worse = int((y > x).sum() if key == "visits" else (y < x).sum())
        out[key] = {"median": [float(np.median(x)), float(np.median(y))], "better": better, "worse": worse,
                    "ties": int((x == y).sum()), "wilcoxon_p": float(stats.wilcoxon(x, y, zero_method="pratt").pvalue)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "outputs/paper/planner_table.json"))
    args = parser.parse_args()
    out = {"val_835": report(load(), ("GPT-4.1", *STUDENTS), (("SFT", "GPT-4.1"), ("SFT+GRPO", "SFT"), ("base", "SFT"))),
           "server_val_test_1226": report(load_server(), tuple(SERVER), (("SFT+GRPO", "SFT"), ("base", "SFT")))}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for name, r in out.items():
        print(f"== {name}: {r['episodes']} episodes, {r['buildings']} buildings")
        for p, m in r["rows"].items():
            print(f"{p:10s} SR {m['SR']:.3f}  SPL {m['SPL']:.3f}  visits {m['visits']:.2f}")
        for k, d in r["differences"].items():
            print(f"{k:28s} {d['mean']:+.3f}  [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
        for k, d in r["base_vs_deployed"].items():
            print(f"base -> SFT+GRPO {k}: median {d['median']}, better/worse/tie {d['better']}/{d['worse']}/{d['ties']}, Wilcoxon p {d['wilcoxon_p']:.2g}")

if __name__ == "__main__":
    main()
