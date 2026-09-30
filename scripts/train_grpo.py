"""Trajectory-level GRPO of the Workspace Selection Adapter (proposal 6-4).

    python scripts/train_grpo.py --config configs/training/grpo.yaml --output outputs/grpo/run1

One update step:
1. sample start states (every_goal findable starts of the train scenes,
   without the buildings held out for SFT validation);
2. from each, G rollouts in the symbolic environment: room decisions by the
   Room adapter and standalone decisions by the Search Location adapter
   (their modes), workspace decisions sampled from the Workspace adapter
   (athome.training.hf_policy: candidate scoring); candidate order shuffled
   per query (athome.search.policy.ShuffledPolicy); room coverage as in the
   Teacher episodes;
3. R = -D - lambda_s N per rollout, group-relative advantage; groups with a
   failed rollout or equal rewards are left out (athome.training.grpo);
4. maximize (1/G) sum_i (1/M_i) sum_k [min(rho A_i, clip(rho) A_i) - beta KL]
   over the workspace decisions k of rollout i (rho = pi_theta/pi_old of the
   chosen candidate, KL(pi_theta || pi_ref) over the candidates, pi_ref = the
   Search Location SFT adapter), updating the Workspace adapter only.
Metrics per step go to <output>/log.jsonl; the adapter is saved every
``save_every`` steps and at the end. ``--max-steps`` / ``--device`` override
the config (smoke tests).
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
from athome.training.grpo import clipped_objective, group_advantages, rollout
from athome.training.hf_policy import (SCORING_SECONDS, HFCandidatePolicy, LockstepScorer,
                                       batch_candidate_logprobs, prompt_text)

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


def _rollout_job(model, tokenizer, enable_thinking, scorer, problem, start, ro, key, stages):
    """One rollout: decisions of the trained stages sampled (generator seeded
    by ``key``), the others by the fixed adapters (mode)."""
    import torch
    generator = torch.Generator(device="cpu").manual_seed(zlib.crc32(key.encode()))

    def job():
        policies = {
            Stage.ROOM: (HFCandidatePolicy(model, tokenizer, "room_train", sample=True, generator=generator,
                                           enable_thinking=enable_thinking, scorer=scorer)
                         if Stage.ROOM in stages else
                         HFCandidatePolicy(model, tokenizer, "room", enable_thinking=enable_thinking, scorer=scorer)),
            Stage.WORKSPACE: HFCandidatePolicy(model, tokenizer, "workspace", sample=True, generator=generator,
                                               enable_thinking=enable_thinking, scorer=scorer),
            Stage.STANDALONE: HFCandidatePolicy(model, tokenizer, "search_location",
                                                enable_thinking=enable_thinking, scorer=scorer)}
        env = SymbolicEnvironment(problem.grid, start, problem.world, problem.observer,
                                  problem.surface.height_at, problem.heading_count, verify_path=False)
        return rollout(problem.graph, problem.navigation(), env, problem.target, policies, ro["step_cost_m"],
                       max_steps=ro["max_steps"], coverage=problem.coverage(), shuffle_seed=key,
                       trained_stages=stages)
    return job


def _update_chunks(tokenizer, items, enable_thinking, opt):
    """Trained decisions in chunks bounded by padded prompt tokens and answer
    rows (GPU memory of one backward pass)."""
    max_tokens = opt.get("update_prompt_tokens", 4096)
    max_rows = opt.get("update_answer_rows", 64)
    lengths = [len(tokenizer(prompt_text(tokenizer, d.messages, enable_thinking),
                             add_special_tokens=False)["input_ids"]) for d, _, _ in items]
    order = sorted(range(len(items)), key=lambda k: lengths[k])
    chunk, width, rows = [], 0, 0
    for k in order:
        n = len(items[k][0].aliases)
        if chunk and (max(width, lengths[k]) * (len(chunk) + 1) > max_tokens or rows + n > max_rows):
            yield chunk
            chunk, width, rows = [], 0, 0
        chunk.append(items[k])
        width, rows = max(width, lengths[k]), rows + n
    if chunk:
        yield chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/grpo.yaml")
    parser.add_argument("--output", type=Path, required=True, help="새 폴더")
    parser.add_argument("--resume-workspace", type=Path, help="이어서 학습할 workspace 어댑터 폴더")
    parser.add_argument("--resume-room", type=Path, help="이어서 학습할 room_train 어댑터 폴더(방 단계 학습 시)")
    parser.add_argument("--start-step", type=int, default=1,
                        help="이어서 학습할 때 첫 스텝 번호(시작 상태 뽑기를 그 스텝부터 이어감)")
    parser.add_argument("--max-steps", type=int, help="설정의 steps 대신(시험 실행)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    import torch
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ro, opt = cfg["rollout"], cfg["optimization"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(opt["seed"])
    rng = random.Random(opt["seed"])
    enable_thinking = cfg["base_model"].get("enable_thinking", False)

    stages = trained_stages(cfg)
    model, tokenizer, trainable = load_model(cfg, device, args.resume_workspace, args.resume_room)
    optimizer = torch.optim.AdamW(trainable, lr=opt["learning_rate"], weight_decay=0.0)

    states = {}
    for path in sorted(glob.glob(str(ROOT / cfg["start_states"]))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            s = json.loads(line)
            if s["oracle_distance_m"] is not None and not held_out(s["scene_id"], cfg["holdout_scene_fraction"]):
                states.setdefault(s["scene_id"], []).append(s)
    if not states:
        raise SystemExit("시작 상태가 없습니다.")
    scenes = Scenes(cfg["layouts"])
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    log = (args.output / "log.jsonl").open("a", encoding="utf-8")
    steps = args.max_steps or opt["steps"]

    def draw():
        return [(sid, s) for sid in rng.sample(sorted(states), min(ro["scenes_per_step"], len(states)))
                for s in rng.sample(states[sid], min(ro["states_per_scene"], len(states[sid])))]

    # A resumed run continues the sequence of start states (and seeds) of the
    # run it extends instead of repeating its first steps. (Optimizer moments
    # start again from zero.)
    for _ in range(1, args.start_step):
        draw()
    last = args.start_step + steps - 1
    for step in range(args.start_step, last + 1):
        t0 = time.time()
        batch = draw()
        groups, reasons, rollouts = [], Counter(), []
        SCORING_SECONDS[0] = 0.0
        jobs, owners = [], []
        scorer = LockstepScorer(model, tokenizer, enable_thinking) if ro.get("batched", True) else None
        for scene_id, state in batch:
            p = scenes.problems(scene_id)[(state["component"], state["target"])]
            p.navigation()                      # build shared goal candidates before the threads
            start = Pose2D(*p.grid.to_xy(tuple(state["start_row_col"])), 0.0)
            for g in range(ro["group_size"]):
                key = f"{opt['seed']}:{step}:{state['state_id']}:{g}"
                jobs.append(_rollout_job(model, tokenizer, enable_thinking, scorer, p, start, ro, key, stages))
                owners.append(len(owners) // ro["group_size"])
        results = scorer.run(jobs) if scorer is not None else [job() for job in jobs]
        for k in range(0, len(results), ro["group_size"]):
            group = results[k:k + ro["group_size"]]
            why = group_advantages(group)
            reasons[why or "usable"] += 1
            rollouts.extend(group)
            if why is None:
                groups.append(group)

        t_rollout = time.time() - t0
        kls, ratios, objective = [], [], 0.0
        # Trained decisions with their weight in (1/#groups)(1/G)(1/M_i) sum_k term.
        items = []
        for group in groups:
            for r in group:
                trained = [d for d in r.decisions if d.trainable and d.aliases and len(d.aliases) > 1]
                for d in trained:
                    items.append((d, r.advantage, 1.0 / (len(groups) * len(group) * len(trained))))
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
                    refs = batch_candidate_logprobs(model, tokenizer, prompts, alias_lists, 10**9, 10**9)
                model.set_adapter(train_adapter)
                news = batch_candidate_logprobs(model, tokenizer, prompts, alias_lists, 10**9, 10**9)
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

        row = {"step": step, "seconds": round(time.time() - t0, 1),
               "rollout_s": round(t_rollout, 1), "scoring_s": round(SCORING_SECONDS[0], 1),
               "update_s": round(time.time() - t0 - t_rollout, 1), "groups": dict(reasons),
               "rollouts": len(rollouts), "states": [s["state_id"] for _, s in batch],
               "success_rate": sum(r.success for r in rollouts) / len(rollouts),
               "mean_reward": sum(r.reward for r in rollouts) / len(rollouts),
               "mean_distance_m": sum(r.distance_m for r in rollouts) / len(rollouts),
               "mean_visits": sum(r.visits for r in rollouts) / len(rollouts),
               "trained_decisions": len(kls), "objective": objective,
               "mean_kl": sum(kls) / len(kls) if kls else None,
               "mean_ratio": sum(ratios) / len(ratios) if ratios else None}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        if step % opt["save_every"] == 0 or step == last:
            step_dir = args.output / f"step_{step:05d}"
            saved = [TRAIN_ADAPTERS[s][0] for s in sorted(stages, key=lambda s: s.value)]
            model.save_pretrained(str(step_dir), selected_adapters=saved)
            from athome.training.adapter_meta import write_meta
            for name in saved:
                reference = "room" if name == "room_train" else "search_location"
                init = (args.resume_room if name == "room_train" else args.resume_workspace) or cfg["adapters"][reference]
                write_meta(step_dir / name, adapter="room" if name == "room_train" else name, method="GRPO",
                           base_model=cfg["base_model"], lora=_lora_of(model, name),
                           data={"start_states": cfg["start_states"], "layouts": cfg["layouts"],
                                 "initialized_from": str(init), "reference_policy": cfg["adapters"][reference],
                                 "trained_stages": sorted(s.value for s in stages),
                                 "fixed_adapters": cfg["adapters"], "start_step": args.start_step},
                           training={**cfg["rollout"], **cfg["optimization"], "step": step},
                           results={k: row[k] for k in ("success_rate", "mean_reward", "mean_distance_m",
                                                       "mean_visits", "mean_kl")})


if __name__ == "__main__":
    main()
