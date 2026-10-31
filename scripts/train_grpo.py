"""Trajectory-level GRPO of the Workspace Selection Adapter (proposal 6-4).

    python scripts/train_grpo.py --config configs/training/grpo_v2.yaml --output outputs/grpo/run1

One update step:
1. sample start states (every_goal findable starts of the train scenes,
   without the buildings held out for SFT validation; with
   ``start_distance_m``, only starts whose l* is in [lo, hi), the ObjectNav
   rule of 1-30 m); each step and draw round has its own seed, so a resumed
   run (``--start-step``) draws what the full run would have drawn;
2. from each, G rollouts in the symbolic environment: room decisions by the
   Room adapter and standalone decisions by the Search Location adapter
   (their modes), workspace decisions sampled from the Workspace adapter
   (athome.training.hf_policy: candidate scoring); candidate order shuffled
   per query (athome.search.policy.ShuffledPolicy); room coverage as in the
   Teacher episodes;
3. R = -D - lambda_s N per rollout, group-relative advantage; groups with a
   failed rollout or equal rewards are left out (athome.training.grpo); with
   ``groups_per_step``, start states are drawn again until that many usable
   groups are collected (at most ``max_draw_rounds`` draws; DAPO dynamic
   sampling, Yu et al. 2025), so every update has the same batch;
4. maximize (1/G) sum_i (1/M_i) sum_k [min(rho A_ik, clip(rho) A_ik) - beta KL]
   over the workspace decisions k of rollout i (rho = pi_theta/pi_old of the
   chosen candidate, KL(pi_theta || pi_ref) over the candidates, pi_ref = the
   Search Location SFT adapter), A_ik = A_i + omega A^S_k with the GiGPO
   step-level advantage A^S (omega = ``step_advantage_weight``, 0: trajectory
   GRPO), updating the Workspace adapter only.
Metrics per step go to <output>/log.jsonl; the adapter is saved every
``save_every`` steps and at the end. With ``validation``, the current policy
(modes, no sampling) runs the starts of the held-out buildings before training
and every ``every`` steps (<output>/validation.jsonl); the policy of the lowest
mean cost D + lambda_s N is kept in <output>/best, and training stops after
``patience`` validations without a lower cost (early stopping on the
validation set; the official Val buildings stay untouched for the final
evaluation). ``--max-steps`` / ``--device`` override the config (smoke tests).
"""
import argparse
from collections import Counter, OrderedDict
import glob
import hashlib
import json
from pathlib import Path
import random
import time
import zlib

import yaml

from athome.data.hm3d.layout import load_layout
from athome.schemas import Pose2D
from athome.search.policy import Stage
from athome.symbolic.environment import SymbolicEnvironment
from athome.training.grpo import clipped_objective, group_advantages, is_trained, rollout, step_advantages
from athome.training.hf_policy import (SCORING_SECONDS, HFCandidatePolicy, LockstepScorer,
                                       packed_candidate_logprobs, packed_length, prompt_text)

from scene_episodes import SceneProblems

ROOT = Path(__file__).resolve().parents[1]


def held_out(scene_id: str, fraction: float) -> bool:
    """Same building split as scripts/train_sft_lora.py."""
    return int(hashlib.sha256(scene_id.encode()).hexdigest(), 16) % 10_000 < fraction * 10_000


class Scenes:
    """Scene problems loaded on demand (a few kept in memory)."""

    def __init__(self, layout_pattern: str, keep: int = 200):
        # Every train scene fits in memory (about 0.2 GB each); loading one
        # takes seconds, so scenes are kept once loaded.
        self.pattern, self.keep, self.cache = layout_pattern, keep, OrderedDict()

    def problems(self, scene_id):
        if scene_id not in self.cache:
            layout = load_layout(ROOT / self.pattern.format(scene_id=scene_id))
            self.cache[scene_id] = SceneProblems(layout=layout).by_key()
            while len(self.cache) > self.keep:
                self.cache.popitem(last=False)
        self.cache.move_to_end(scene_id)
        return self.cache[scene_id]


# Trained stage -> (trained adapter, reference adapter = its SFT policy).
TRAIN_ADAPTERS = {Stage.WORKSPACE: ("workspace", "search_location"), Stage.ROOM: ("room_train", "room")}


def trained_stages(cfg):
    return frozenset(Stage(s) for s in cfg["rollout"].get("trained_stages", ["workspace"]))


def load_model(cfg, device, workspace_init=None, room_init=None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base["name"], revision=base.get("revision"))
    model = AutoModelForCausalLM.from_pretrained(base["name"], revision=base.get("revision"),
                                                 torch_dtype=getattr(torch, base["dtype"])).to(device)
    ad = cfg["adapters"]
    model = PeftModel.from_pretrained(model, str(ROOT / ad["room"]), adapter_name="room", is_trainable=False)
    model.load_adapter(str(ROOT / ad["search_location"]), adapter_name="search_location", is_trainable=False)
    model.load_adapter(str(workspace_init or ROOT / ad["search_location"]), adapter_name="workspace",
                       is_trainable=True)
    if Stage.ROOM in trained_stages(cfg):
        # A trainable copy of the Room SFT adapter; "room" stays the reference.
        model.load_adapter(str(room_init or ROOT / ad["room"]), adapter_name="room_train", is_trainable=True)
    model.eval()                       # no dropout: pi_old and pi_theta are the same function
    if hasattr(model, "gradient_checkpointing_enable") and device != "cpu":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    names = [TRAIN_ADAPTERS[s][0] for s in trained_stages(cfg)]
    trainable = [p for n, p in model.named_parameters()
                 if "lora_" in n and any(f".{a}." in n for a in names)]
    return model, tokenizer, trainable


def _lora_of(model, name):
    c = model.peft_config[name]
    return {"r": c.r, "alpha": c.lora_alpha, "dropout": c.lora_dropout,
            "target_modules": sorted(c.target_modules)}


def _rollout_job(model, tokenizer, enable_thinking, scorer, problem, start, ro, key, stages, sample=True):
    """One rollout: decisions of the trained stages sampled (generator seeded
    by ``key``; ``sample=False``: their modes, for validation), the others by
    the fixed adapters (mode)."""
    import torch
    generator = torch.Generator(device="cpu").manual_seed(zlib.crc32(key.encode()))

    def job():
        policies = {
            Stage.ROOM: (HFCandidatePolicy(model, tokenizer, "room_train", sample=sample, generator=generator,
                                           enable_thinking=enable_thinking, scorer=scorer)
                         if Stage.ROOM in stages else
                         HFCandidatePolicy(model, tokenizer, "room", enable_thinking=enable_thinking, scorer=scorer)),
            Stage.WORKSPACE: HFCandidatePolicy(model, tokenizer, "workspace", sample=sample, generator=generator,
                                               enable_thinking=enable_thinking, scorer=scorer),
            Stage.STANDALONE: HFCandidatePolicy(model, tokenizer, "search_location",
                                                enable_thinking=enable_thinking, scorer=scorer)}
        env = SymbolicEnvironment(problem.grid, start, problem.world, problem.observer,
                                  problem.surface.height_at, problem.heading_count, verify_path=False)
        return rollout(problem.graph, problem.navigation(), env, problem.target, policies, ro["step_cost_m"],
                       max_steps=ro["max_steps"], coverage=problem.coverage(), shuffle_seed=key,
                       trained_stages=stages)
    return job


def _run(model, tokenizer, enable_thinking, scenes, ro, stages, starts, sample=True):
    """Rollouts of ``starts`` = [(state, key)], all at once (batched scoring)."""
    scorer = LockstepScorer(model, tokenizer, enable_thinking) if ro.get("batched", True) else None
    jobs = []
    for state, key in starts:
        p = scenes.problems(state["scene_id"])[(state["component"], state["target"])]
        p.navigation()                      # build shared goal candidates before the threads
        start = Pose2D(*p.grid.to_xy(tuple(state["start_row_col"])), 0.0)
        jobs.append(_rollout_job(model, tokenizer, enable_thinking, scorer, p, start, ro, key, stages, sample))
    return scorer.run(jobs) if scorer is not None else [job() for job in jobs]


def validate(model, tokenizer, enable_thinking, scenes, ro, stages, states, batch):
    """Mean cost D + lambda_s N and success rate of the current policy (modes)
    on ``states``, the same candidate orders at every validation."""
    results = []
    for k in range(0, len(states), batch):
        results += _run(model, tokenizer, enable_thinking, scenes, ro, stages,
                        [(s, f"validation:{s['state_id']}") for s in states[k:k + batch]], sample=False)
    return {"episodes": len(results),
            "cost": sum(r.distance_m + ro["step_cost_m"] * r.visits for r in results) / len(results),
            "success_rate": sum(r.success for r in results) / len(results),
            "mean_visits": sum(r.visits for r in results) / len(results)}


def _update_chunks(tokenizer, items, enable_thinking, opt):
    """Trained decisions in chunks of at most ``update_tokens`` packed tokens
    (prompt + every answer, athome.training.hf_policy.packed_candidate_logprobs):
    the activation memory of one backward pass grows with these tokens. A
    longer decision forms a chunk of its own."""
    max_tokens = opt.get("update_tokens", 2048)
    chunk, total = [], 0
    for item in items:
        d = item[0]
        n = packed_length(tokenizer, prompt_text(tokenizer, d.messages, enable_thinking), list(d.aliases))
        if chunk and total + n > max_tokens:
            yield chunk
            chunk, total = [], 0
        chunk.append(item)
        total += n
    if chunk:
        yield chunk


def save_adapters(model, step_dir, cfg, args, stages, step, results):
    saved = [TRAIN_ADAPTERS[s][0] for s in sorted(stages, key=lambda s: s.value)]
    model.save_pretrained(str(step_dir), selected_adapters=saved)
    from athome.training.adapter_meta import write_meta
    for name in saved:
        reference = "room" if name == "room_train" else "search_location"
        init = (args.resume_room if name == "room_train" else args.resume_workspace) or cfg["adapters"][reference]
        write_meta(step_dir / name, adapter="room" if name == "room_train" else name, method="GRPO",
                   base_model=cfg["base_model"], lora=_lora_of(model, name),
                   data={"start_states": cfg["start_states"], "start_distance_m": cfg.get("start_distance_m"),
                         "layouts": cfg["layouts"],
                         "initialized_from": str(init), "reference_policy": cfg["adapters"][reference],
                         "trained_stages": sorted(s.value for s in stages),
                         "fixed_adapters": cfg["adapters"], "start_step": args.start_step},
                   training={**cfg["rollout"], **cfg["optimization"], "step": step},
                   results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/grpo_v2.yaml")
    parser.add_argument("--output", type=Path, required=True, help="새 폴더")
    parser.add_argument("--resume-workspace", type=Path, help="이어서 학습할 workspace 어댑터 폴더")
    parser.add_argument("--resume-room", type=Path, help="이어서 학습할 room_train 어댑터 폴더(방 단계 학습 시)")
    parser.add_argument("--start-step", type=int, default=1,
                        help="이어서 학습할 때 첫 스텝 번호(그 스텝의 시작 상태와 시드부터 이어감)")
    parser.add_argument("--max-steps", type=int, help="설정의 steps 대신(시험 실행)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    import torch
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ro, opt, val = cfg["rollout"], cfg["optimization"], cfg.get("validation")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(opt["seed"])
    enable_thinking = cfg["base_model"].get("enable_thinking", False)
    target_groups, max_rounds = ro.get("groups_per_step"), ro.get("max_draw_rounds", 1)
    omega = opt.get("step_advantage_weight", 0.0)

    stages = trained_stages(cfg)
    model, tokenizer, trainable = load_model(cfg, device, args.resume_workspace, args.resume_room)
    optimizer = torch.optim.AdamW(trainable, lr=opt["learning_rate"], weight_decay=0.0)

    lo, hi = cfg.get("start_distance_m") or (0.0, float("inf"))
    states, validation_states = {}, []
    for path in sorted(glob.glob(str(ROOT / cfg["start_states"]))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            s = json.loads(line)
            if s["oracle_distance_m"] is None:
                continue
            if held_out(s["scene_id"], cfg["holdout_scene_fraction"]):
                validation_states.append(s)          # every start, as the holdout evaluation
            elif lo <= s["oracle_distance_m"] < hi:
                states.setdefault(s["scene_id"], []).append(s)
    if not states:
        raise SystemExit("시작 상태가 없습니다.")
    print(f"학습 시작 상태 {sum(map(len, states.values()))}개 ({len(states)}개 건물), "
          f"검증 {len(validation_states) if val else 0}개", flush=True)
    scenes = Scenes(cfg["layouts"])
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    log = (args.output / "log.jsonl").open("a", encoding="utf-8")
    steps = args.max_steps or opt["steps"]
    per_draw = ro["scenes_per_step"] * ro["states_per_scene"]

    def draw(step, k):
        rng = random.Random(f"{opt['seed']}:{step}:{k}")
        return [s for sid in rng.sample(sorted(states), min(ro["scenes_per_step"], len(states)))
                for s in rng.sample(states[sid], min(ro["states_per_scene"], len(states[sid])))]

    best, stale = None, 0

    def check(step):
        """Validation at ``step``; returns True when training should stop."""
        nonlocal best, stale
        t0 = time.time()
        result = validate(model, tokenizer, enable_thinking, scenes, ro, stages, validation_states,
                          per_draw * ro["group_size"])
        improved = best is None or result["cost"] < best["cost"]
        if improved:
            best, stale = {"step": step, **result}, 0
            save_adapters(model, args.output / "best", cfg, args, stages, step, {"validation": result})
        else:
            stale += 1
        row = {"step": step, **result, "best_step": best["step"], "seconds": round(time.time() - t0, 1)}
        with (args.output / "validation.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({"validation": row}), flush=True)
        return stale >= val["patience"]

    if val:
        check(args.start_step - 1)
    last = args.start_step + steps - 1
    for step in range(args.start_step, last + 1):
        t0 = time.time()
        groups, reasons, rollouts, drawn, used = [], Counter(), [], [], []
        SCORING_SECONDS[0] = 0.0
        for k in range(max_rounds):
            batch = draw(step, k)
            drawn += [s["state_id"] for s in batch]
            results = _run(model, tokenizer, enable_thinking, scenes, ro, stages,
                           [(s, f"{opt['seed']}:{step}:{k}:{s['state_id']}:{g}")
                            for s in batch for g in range(ro["group_size"])])
            for state, j in zip(batch, range(0, len(results), ro["group_size"])):
                group = results[j:j + ro["group_size"]]
                why = group_advantages(group)
                rollouts.extend(group)
                if why is None and target_groups is not None and len(groups) >= target_groups:
                    why = "surplus"                # usable, beyond the batch (DAPO keeps the batch size)
                reasons[why or "usable"] += 1
                if why is None:
                    groups.append(group)
                    used.append(state["state_id"])
            if target_groups is None or len(groups) >= target_groups:
                break
        anchored = sum(step_advantages(g, ro["step_cost_m"]) for g in groups) if omega else 0

        t_rollout = time.time() - t0
        kls, ratios, objective = [], [], 0.0
        # Trained decisions with their weight in (1/#groups)(1/G)(1/M_i) sum_k term.
        items = []
        for group in groups:
            for r in group:
                trained = [d for d in r.decisions if is_trained(d)]
                for d in trained:
                    items.append((d, r.advantage + omega * d.step_advantage,
                                  1.0 / (len(groups) * len(group) * len(trained))))
        by_stage = {}
        for item in items:
            by_stage.setdefault(Stage(item[0].stage), []).append(item)
        for _ in range(opt["updates_per_batch"]):
            optimizer.zero_grad()
            for stage_k, chunk in ((s, c) for s in sorted(by_stage, key=lambda s: s.value)
                                   for c in _update_chunks(tokenizer, by_stage[s], enable_thinking, opt)):
                train_adapter, ref_adapter = TRAIN_ADAPTERS[stage_k]
                prompts = [prompt_text(tokenizer, d.messages, enable_thinking) for d, _, _ in chunk]
                alias_lists = [list(d.aliases) for d, _, _ in chunk]
                model.set_adapter(ref_adapter)
                with torch.no_grad():
                    refs = packed_candidate_logprobs(model, tokenizer, prompts, alias_lists)
                model.set_adapter(train_adapter)
                news = packed_candidate_logprobs(model, tokenizer, prompts, alias_lists)
                loss = 0.0
                for (d, advantage, weight), new_s, ref_s, aliases in zip(chunk, news, refs, alias_lists):
                    new, ref = torch.log_softmax(new_s, 0), torch.log_softmax(ref_s, 0)
                    chosen = aliases.index(json.loads(d.output)["selected_id"])
                    kl = torch.sum(new.exp() * (new - ref))
                    # One update per batch: pi_old is the current policy (as TRL's
                    # GRPO with num_iterations=1), so rollout-time rounding does not
                    # enter the ratio; with more updates the rollout log-prob is used.
                    old = new[chosen].detach() if opt["updates_per_batch"] == 1 else d.logprob
                    term = clipped_objective(new[chosen], old, advantage, opt["clip_eps"], kl, opt["kl_beta"])
                    loss = loss - weight * term
                    kls.append(float(kl))
                    ratios.append(float(torch.exp(new[chosen].detach() - old)))
                loss.backward()
                objective += -float(loss)
            if groups:
                torch.nn.utils.clip_grad_norm_(trainable, opt["max_grad_norm"])
                optimizer.step()

        trained = [d for g in groups for r in g for d in r.decisions if is_trained(d)]
        row = {"step": step, "seconds": round(time.time() - t0, 1),
               "rollout_s": round(t_rollout, 1), "scoring_s": round(SCORING_SECONDS[0], 1),
               "update_s": round(time.time() - t0 - t_rollout, 1), "groups": dict(reasons),
               "draw_rounds": k + 1, "rollouts": len(rollouts), "states": drawn, "used_states": used,
               "success_rate": sum(r.success for r in rollouts) / len(rollouts),
               "mean_reward": sum(r.reward for r in rollouts) / len(rollouts),
               "mean_distance_m": sum(r.distance_m for r in rollouts) / len(rollouts),
               "mean_visits": sum(r.visits for r in rollouts) / len(rollouts),
               "trained_decisions": len(kls), "objective": objective,
               "anchor_groups": anchored,
               "mean_abs_step_advantage": (sum(abs(d.step_advantage) for d in trained) / len(trained)
                                           if omega and trained else None),
               "mean_kl": sum(kls) / len(kls) if kls else None,
               "mean_ratio": sum(ratios) / len(ratios) if ratios else None}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        stop = val is not None and step % val["every"] == 0 and check(step)
        if step % opt["save_every"] == 0 or step == last or stop:
            save_adapters(model, args.output / f"step_{step:05d}", cfg, args, stages, step,
                          {k: row[k] for k in ("success_rate", "mean_reward", "mean_distance_m",
                                               "mean_visits", "mean_kl")})
        if stop:
            print(f"검증 비용이 {val['patience']}회 연속 낮아지지 않아 {step} 스텝에서 멈춥니다 "
                  f"(최적: {best['step']} 스텝, 비용 {best['cost']:.2f}).", flush=True)
            break


if __name__ == "__main__":
    main()
