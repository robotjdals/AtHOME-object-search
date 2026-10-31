"""Summary of scripts/compare_graph_representations.py outputs.

Per representation x planner: SR, SPL (Anderson et al. 2018, shortest path
to the nearest view of the target), cost D + 3N, visits, distance, and the
paired cost difference to ``full`` with the same planner (95% CI from a
bootstrap over buildings, the unit the held-out split is drawn by). Per
target split (seen / synonym / unseen categories, HM3D-OVON style) and, for
the Teacher, prompt size per query.

    python scripts/summarize_graph_compare.py outputs/graph_compare/free_*.jsonl
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

import yaml

from athome.data.hm3d.layout import load_layout

ROOT = Path(__file__).resolve().parents[1]
# The robot's planner server: 4,096-token context, answers of at most 32
# tokens (athome.inference.llm_policy), so a refused prompt exceeded 4,064.
FAILED_PROMPT_MIN = 4096 - 32 + 1
ORDER = ("full","wo_observation", "wo_location_policy", "wo_workspace", "object_list")
LABELS = {"full": "Full", "wo_observation": "w/o observation state",
          "wo_location_policy": "w/o Search Location policy", "wo_workspace": "w/o workspace layer",
          "object_list": "Unstructured object list", "llm": "LLM (GPT-4.1)",
          "llm_dry": "LLM prompt dry run (equal likelihoods)",
          "qwen": "LLM (Qwen3-4B zero-shot)", "cooccurrence": "Co-occurrence statistics", "greedy": "Greedy", "random": "Random"}


def label(key):
    variant, planner = key.split("/")
    return f"{LABELS[variant]} + {LABELS[planner]}"


def spl(row, result):
    """Same as scripts/evaluate_student.py."""
    shortest = row["oracle_distance_m"]
    longest = max(shortest, result["distance_m"])
    if not result["found"]:
        return 0.0
    return shortest / longest if longest > 0 else 1.0


def target_split(scene_id, target, cache={}):
    if scene_id not in cache:
        pattern = yaml.safe_load((ROOT / "configs/training/grpo.yaml").read_text())["layouts"]
        targets = json.loads(load_layout(ROOT / pattern.format(scene_id=scene_id)).targets.read_text())
        cache[scene_id] = targets
    t = cache[scene_id]
    return ("unseen" if target in t.get("unseen_categories", []) else
            "synonym" if target in t.get("synonym_categories", []) else "seen")


def building_ci(rows, diff, n=2000, seed=0):
    """Mean of diff(row) and a 95% bootstrap CI resampling buildings."""
    by = defaultdict(list)
    for r in rows:
        by[r["scene_id"]].append(diff(r))
    scenes = sorted(by)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [v for s in (rng.choice(scenes) for _ in scenes) for v in by[s]]
        means.append(statistics.mean(sample))
    means.sort()
    return statistics.mean(diff(r) for r in rows), means[int(0.025 * n)], means[int(0.975 * n)]


def efficiency(result, budget):
    """This episode's share of the success-within-budget curve's area
    (MoMa-LLM's AUC-E with cost D + 3N as the budget axis): found within b
    for every b in [cost, budget], i.e. max(0, 1 - cost / budget)."""
    return max(0.0, 1.0 - result["cost"] / budget) if result["found"] else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--budget", type=float, default=100.0,
                        help="AUC-E 상한 (비용 D + 3N 단위, m 환산)")
    parser.add_argument("--start-rule", action="store_true",
                        help="ObjectNav 출발 규칙: 최단 거리 1 m 이상 30 m 미만 에피소드만 (HM3D-OVON, HSSD-200)")
    parser.add_argument("--step-budget", type=int,
                        help="ObjectNav 동작 상한 (Habitat: 500). 4 (D + 3N)이 넘은 에피소드는 실패")
    parser.add_argument("--json", type=Path, help="요약을 JSON으로도 저장")
    args = parser.parse_args()
    merged = {}                   # one row per episode (planners may come from separate runs)
    for p in args.inputs:
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                merged.setdefault(r["state_id"], {}).update(r)
    rows = list(merged.values())
    if args.start_rule:           # episodes whose target is not in view at the start
        rows = [r for r in rows if 1.0 <= r["oracle_distance_m"] < 30.0]
    if args.step_budget:
        # Habitat ObjectNav actions: forward 0.25 m, turn 30 deg; a visit's
        # 360-degree look is 12 turns, so an episode takes 4 (D + 3N) actions.
        # Beyond the budget the episode is a failure (SR, SPL, AUC-E 0).
        for r in rows:
            for k in [k for k in r if "/" in k]:
                if r[k]["found"] and 4 * r[k]["cost"] > args.step_budget:
                    r[k] = {**r[k], "found": False}
    for r in rows:
        r["split"] = target_split(r["scene_id"], r["target"])
    keys = sorted({k for r in rows for k in r if "/" in k},
                  key=lambda k: (k.split("/")[1], ORDER.index(k.split("/")[0])))
    out = {"episodes": len(rows), "buildings": len({r["scene_id"] for r in rows}), "start_rule": args.start_rule,
           "step_budget": args.step_budget,
           "budget": args.budget, "results": {}}
    print(f"에피소드 {len(rows)}개, 건물 {out['buildings']}개. "
          f"SPL(1-30)은 ObjectNav 시작 규칙(최단 1~30 m) 에피소드만, AUC-E는 비용 0~{args.budget:g}")
    print(f"{'표현 + 플래너':52} {'SR':>6} {'SPL':>6} {'SPL1-30':>8} {'SR@25':>6} {'SR@50':>6} "
          f"{'AUC-E':>6} {'방문':>6} {'비용':>7}  비용 차이 / AUC-E 차이 vs Full (95% CI, 건물)")
    for key in keys:
        sel = [r for r in rows if key in r]
        variant, planner = key.split("/")
        m = {name: statistics.mean(r[key][f] for r in sel)
             for name, f in (("success", "found"), ("cost", "cost"), ("visits", "visits"), ("distance", "distance_m"))}
        m["spl"] = statistics.mean(spl(r, r[key]) for r in sel)
        standard = [r for r in sel if 1.0 <= r["oracle_distance_m"] < 30.0]
        m["spl_1_30"] = statistics.mean(spl(r, r[key]) for r in standard) if standard else None
        for b in (25, 50, 100):
            m[f"sr_at_{b}"] = statistics.mean(r[key]["found"] and r[key]["cost"] <= b for r in sel)
        m["auc_e"] = statistics.mean(efficiency(r[key], args.budget) for r in sel)
        m["label"], m["episodes"] = label(key), len(sel)
        m["locations"] = statistics.mean(r[key]["locations"] for r in sel)
        line = (f"{m['label']:52} {m['success']:6.3f} {m['spl']:6.3f} {m['spl_1_30'] or 0:8.3f} "
                f"{m['sr_at_25']:6.3f} {m['sr_at_50']:6.3f} {m['auc_e']:6.3f} {m['visits']:6.2f} {m['cost']:7.2f}")
        ref = f"full/{planner}"
        if variant != "full" and all(ref in r for r in sel):
            d, lo, hi = building_ci(sel, lambda r: r[key]["cost"] - r[ref]["cost"])
            m["cost_diff_vs_full"] = [d, lo, hi]
            a, alo, ahi = building_ci(sel, lambda r: efficiency(r[key], args.budget) - efficiency(r[ref], args.budget))
            m["auc_e_diff_vs_full"] = [a, alo, ahi]
            m["visits_diff_vs_full"] = list(building_ci(sel, lambda r: r[key]["visits"] - r[ref]["visits"]))
            line += f"  {d:+7.2f} ({lo:+.2f}, {hi:+.2f}) / {a:+.3f} ({alo:+.3f}, {ahi:+.3f})"
        if planner in ("llm", "llm_dry", "qwen"):
            queries = sum(r[key].get("queries", 0) for r in sel)
            m["queries_per_episode"] = queries / len(sel)
            m["prompt_tokens_per_query"] = sum(r[key].get("prompt_tokens", 0) for r in sel) / queries if queries else None
            m["candidates_max"] = max(r[key].get("candidates_max", 0) for r in sel)
            if all("candidates_sum" in r[key] for r in sel):
                m["candidates_per_query"] = sum(r[key]["candidates_sum"] for r in sel) / queries if queries else None
            m["fallbacks"] = sum(r[key]["fallbacks"] for r in sel)
            m["invalid_rate"] = m["fallbacks"] / (queries + m["fallbacks"]) if queries + m["fallbacks"] else None
            if planner == "qwen":
                # A query without a valid answer was refused for its length
                # (re-run of every affected episode: all HTTP 400 context
                # errors), so its prompt had at least FAILED_PROMPT_MIN tokens.
                total = sum(r[key].get("prompt_tokens", 0) for r in sel) + m["fallbacks"] * FAILED_PROMPT_MIN
                m["prompt_tokens_per_episode"] = total / len(sel)
                m["prompt_tokens_per_episode_lower_bound"] = m["fallbacks"] > 0
        m["by_split"] = {s: {"n": len(v), "success": statistics.mean(r[key]["found"] for r in v),
                             "spl": statistics.mean(spl(r, r[key]) for r in v),
                             "auc_e": statistics.mean(efficiency(r[key], args.budget) for r in v),
                             "cost": statistics.mean(r[key]["cost"] for r in v)}
                         for s in ("seen", "synonym", "unseen")
                         for v in [[r for r in sel if r["split"] == s]] if v}
        out["results"][key] = m
        print(line)
    llm = {k: m for k, m in out["results"].items() if "queries_per_episode" in m}
    if llm:
        print("\n표현 크기 (탐색 위치 = 노드 수, 질의당 후보와 LLM 입력):")
        for key, m in llm.items():
            per_query = m.get("candidates_per_query")
            print(f"  {m['label']:52} 탐색 위치 {m['locations']:.0f}, 질의/에피소드 {m['queries_per_episode']:.1f}, "
                  f"후보/질의 {'-' if per_query is None else f'{per_query:.1f}'} (최대 {m['candidates_max']}), "
                  f"입력 토큰/질의 {m['prompt_tokens_per_query'] or 0:.0f}, 응답 실패 {m['fallbacks']}")
    print("\n목표 분류별 비용 / AUC-E (seen / synonym / unseen):")
    for key, m in out["results"].items():
        print(f"  {m['label']:52} " + "  ".join(f"{s} {v['cost']:.1f} / {v['auc_e']:.3f} (n={v['n']})"
                                                 for s, v in m["by_split"].items()))
    if args.json:
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
