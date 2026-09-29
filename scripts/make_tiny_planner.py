"""Tiny random Qwen3 planner for local smoke tests of the training code.

Same model class and tokenizer as the real base model (Qwen3ForCausalLM,
Qwen/Qwen3-4B tokenizer at the pinned revision), 2 layers of width 64 and
random weights, as library test suites use tiny-random models. It checks the
code paths (templates, masking, LoRA on every linear layer, GRPO objective,
saving), not quality, memory or speed. Writes <output>/base and the configs
<output>/sft_lora.yaml and <output>/grpo.yaml pointing at it.
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    sft = yaml.safe_load((ROOT / "configs/training/sft_lora.yaml").read_text(encoding="utf-8"))
    grpo = yaml.safe_load((ROOT / "configs/training/grpo.yaml").read_text(encoding="utf-8"))
    base = sft["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base["name"], revision=base["revision"])
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         max_position_embeddings=8192, tie_word_embeddings=True)
    model = Qwen3ForCausalLM(config)
    out = args.output / "base"
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    tiny = {"name": str(out.resolve()), "revision": None, "dtype": "float32", "enable_thinking": False}
    sft["base_model"] = {**sft["base_model"], **tiny}
    sft["training"] = {**sft["training"], "per_device_batch_size": 4, "gradient_accumulation_steps": 1}
    grpo["base_model"] = {**grpo["base_model"], **tiny}
    grpo["adapters"] = {"room": str((args.output / "sft_room/adapter").resolve()),
                        "search_location": str((args.output / "sft_search_location/adapter").resolve())}
    grpo["rollout"] = {**grpo["rollout"], "group_size": 4, "scenes_per_step": 1, "states_per_scene": 2}
    grpo["optimization"] = {**grpo["optimization"], "save_every": 1}
    for name, cfg in (("sft_lora.yaml", sft), ("grpo.yaml", grpo)):
        (args.output / name).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"tiny planner: {sum(p.numel() for p in model.parameters())} parameters → {args.output}")


if __name__ == "__main__":
    main()
