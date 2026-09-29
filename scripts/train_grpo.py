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

import yaml

from athome.data.hm3d.layout import load_layout
from athome.schemas import Pose2D
from athome.search.policy import Stage
from athome.symbolic.environment import SymbolicEnvironment
from athome.training.grpo import clipped_objective, group_advantages, rollout
from athome.training.hf_policy import HFCandidatePolicy, candidate_logprobs, prompt_text

from scene_episodes import SceneProblems

ROOT = Path(__file__).resolve().parents[1]


def held_out(scene_id: str, fraction: float) -> bool:
    """Same building split as scripts/train_sft_lora.py."""
    return int(hashlib.sha256(scene_id.encode()).hexdigest(), 16) % 10_000 < fraction * 10_000


class Scenes:
    """Scene problems loaded on demand (a few kept in memory)."""

    def __init__(self, layout_pattern: str, keep: int = 4):
        self.pattern, self.keep, self.cache = layout_pattern, keep, OrderedDict()

    def problems(self, scene_id):
        if scene_id not in self.cache:
            layout = load_layout(ROOT / self.pattern.format(scene_id=scene_id))
            self.cache[scene_id] = SceneProblems(layout=layout).by_key()
            while len(self.cache) > self.keep:
                self.cache.popitem(last=False)
        self.cache.move_to_end(scene_id)
        return self.cache[scene_id]


def load_model(cfg, device, workspace_init=None):
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
    model.eval()                       # no dropout: pi_old and pi_theta are the same function
    if hasattr(model, "gradient_checkpointing_enable") and device != "cpu":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    trainable = [p for n, p in model.named_parameters() if ".workspace." in n and "lora_" in n]
    return model, tokenizer, trainable


def _lora_of(model, name):
    c = model.peft_config[name]
    return {"r": c.r, "alpha": c.lora_alpha, "dropout": c.lora_dropout,
            "target_modules": sorted(c.target_modules)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/grpo.yaml")
    parser.add_argument("--output", type=Path, required=True, help="새 폴더")
    parser.add_argument("--resume-workspace", type=Path, help="이어서 학습할 workspace 어댑터 폴더")
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
    generator = torch.Generator(device="cpu").manual_seed(opt["seed"])
    enable_thinking = cfg["base_model"].get("enable_thinking", False)

    model, tokenizer, trainable = load_model(cfg, device, args.resume_workspace)
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

    for step in range(1, steps + 1):
        t0 = time.time()
        batch = [(sid, s) for sid in rng.sample(sorted(states), min(ro["scenes_per_step"], len(states)))
                 for s in rng.sample(states[sid], min(ro["states_per_scene"], len(states[sid])))]
        groups, reasons, rollouts = [], Counter(), []
        for scene_id, state in batch:
            p = scenes.problems(scene_id)[(state["component"], state["target"])]
            start = Pose2D(*p.grid.to_xy(tuple(state["start_row_col"])), 0.0)
            group = []
            for g in range(ro["group_size"]):
                policies = {
                    Stage.ROOM: HFCandidatePolicy(model, tokenizer, "room", enable_thinking=enable_thinking),
                    Stage.WORKSPACE: HFCandidatePolicy(model, tokenizer, "workspace", sample=True,
                                                       generator=generator, enable_thinking=enable_thinking),
                    Stage.STANDALONE: HFCandidatePolicy(model, tokenizer, "search_location",
                                                        enable_thinking=enable_thinking)}
                env = SymbolicEnvironment(p.grid, start, p.world, p.observer, p.surface.height_at, p.heading_count, verify_path=False)
                group.append(rollout(p.graph, p.navigation(), env, p.target, policies, ro["step_cost_m"],
                                     max_steps=ro["max_steps"], coverage=p.coverage(),
                                     shuffle_seed=f"{opt['seed']}:{step}:{state['state_id']}:{g}"))
            why = group_advantages(group)
            reasons[why or "usable"] += 1
            rollouts.extend(group)
            if why is None:
                groups.append(group)

        kls, ratios, objective = [], [], 0.0
        for _ in range(opt["updates_per_batch"]):
            optimizer.zero_grad()
            for group in groups:
                for r in group:
                    trained = [d for d in r.decisions if d.trainable and d.aliases and len(d.aliases) > 1]
                    for d in trained:
                        aliases = list(d.aliases)
                        prompt = prompt_text(tokenizer, d.messages, enable_thinking)
                        chosen = aliases.index(json.loads(d.output)["selected_id"])
                        model.set_adapter("search_location")
                        with torch.no_grad():
                            ref = torch.log_softmax(candidate_logprobs(model, tokenizer, prompt, aliases), 0)
                        model.set_adapter("workspace")
                        new = torch.log_softmax(candidate_logprobs(model, tokenizer, prompt, aliases), 0)
                        kl = torch.sum(new.exp() * (new - ref))
                        term = clipped_objective(new[chosen], d.logprob, r.advantage, opt["clip_eps"],
                                                 kl, opt["kl_beta"])
                        # (1/#groups)(1/G)(1/M_i) sum_k term, maximized.
                        loss = -term / (len(groups) * len(group) * len(trained))
                        loss.backward()
                        objective += -loss.item()
                        kls.append(float(kl))
                        ratios.append(float(torch.exp(new[chosen].detach() - d.logprob)))
            if groups:
                torch.nn.utils.clip_grad_norm_(trainable, opt["max_grad_norm"])
                optimizer.step()

        row = {"step": step, "seconds": round(time.time() - t0, 1), "groups": dict(reasons),
               "rollouts": len(rollouts),
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
        if step % opt["save_every"] == 0 or step == steps:
            step_dir = args.output / f"step_{step:05d}"
            model.save_pretrained(str(step_dir), selected_adapters=["workspace"])
            from athome.training.adapter_meta import write_meta
            write_meta(step_dir / "workspace", adapter="workspace", method="GRPO",
                       base_model=cfg["base_model"], lora=_lora_of(model, "workspace"),
                       data={"start_states": cfg["start_states"], "layouts": cfg["layouts"],
                             "initialized_from": str(args.resume_workspace or cfg["adapters"]["search_location"]),
                             "reference_policy": cfg["adapters"]["search_location"],
                             "fixed_adapters": cfg["adapters"]},
                       training={**cfg["rollout"], **cfg["optimization"], "step": step},
                       results={k: row[k] for k in ("success_rate", "mean_reward", "mean_distance_m",
                                                   "mean_visits", "mean_kl")})


if __name__ == "__main__":
    main()
