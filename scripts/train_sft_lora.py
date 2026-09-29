"""LoRA-SFT of one planner adapter (proposal 6-3) on the GPU server.

    python scripts/train_sft_lora.py --data outputs/sft_v5g/train/room.jsonl \
        --config configs/training/sft_lora.yaml --output outputs/adapters/room
    (--dry-run: split and tokenize only, no model weights or GPU)

Input: an export of scripts/export_sft_dataset.py (<adapter>.jsonl with the
matching .meta.jsonl). Each example is rendered exactly as the planner sees it
at inference: the chat template of the base model with the generation prompt
(thinking disabled), followed by the answer and the end-of-turn token; the
loss covers the answer only. A share of buildings (sha256 of the scene ID) is
held out for the validation loss, so no building is in both. Settings and
their sources are in the config (configs/training/sft_lora.yaml).
Requires transformers, peft and torch (GPU); the dry run needs transformers
only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import yaml


def load(data_path: Path):
    meta_path = data_path.with_name(data_path.name.replace(".jsonl", ".meta.jsonl"))
    examples = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    metas = [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(examples) != len(metas) or any(e["id"] != m["id"] for e, m in zip(examples, metas)):
        raise SystemExit("데이터와 meta 파일의 줄이 맞지 않습니다.")
    return examples, metas


def split(examples, metas, fraction):
    def held(scene):
        return int(hashlib.sha256(scene.encode()).hexdigest(), 16) % 10_000 < fraction * 10_000
    train = [e for e, m in zip(examples, metas) if not held(m["scene_id"])]
    val = [e for e, m in zip(examples, metas) if held(m["scene_id"])]
    return train, val, sorted({m["scene_id"] for m in metas if held(m["scene_id"])})


def tokenize(tokenizer, example, cfg):
    """input_ids and labels (-100 on the prompt), rendered as at inference."""
    *prompt_messages, answer = example["messages"]
    if answer["role"] != "assistant":
        raise ValueError(f"{example['id']}: 마지막 메시지가 답이 아닙니다.")
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=cfg["base_model"]["enable_thinking"])
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer["content"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + answer_ids
    if len(ids) > cfg["max_length"]:
        return None
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + answer_ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="<adapter>.jsonl (export_sft_dataset.py)")
    parser.add_argument("--config", type=Path, default=Path("configs/training/sft_lora.yaml"))
    parser.add_argument("--output", type=Path, required=True, help="어댑터 저장 폴더(새 폴더)")
    parser.add_argument("--dry-run", action="store_true", help="분할·토큰화만 확인(가중치·GPU 불필요)")
    parser.add_argument("--limit", type=int, help="앞에서부터 이 개수만 사용(시험 실행)")
    parser.add_argument("--epochs", type=float, help="설정의 epochs 대신(시험 실행)")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output.exists() and not args.dry_run:
        raise SystemExit(f"이미 존재하는 출력: {args.output}")

    from transformers import AutoTokenizer
    base = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base["name"], revision=base.get("revision"))
    examples, metas = load(args.data)
    if args.limit:
        examples, metas = examples[:args.limit], metas[:args.limit]
    train, val, held_scenes = split(examples, metas, cfg["validation"]["holdout_scene_fraction"])
    train_tok = [t for t in (tokenize(tokenizer, e, cfg) for e in train) if t]
    val_tok = [t for t in (tokenize(tokenizer, e, cfg) for e in val) if t]
    lengths = sorted(len(t["input_ids"]) for t in train_tok + val_tok)
    report = {"data": str(args.data), "train": len(train_tok), "val": len(val_tok),
              "too_long_dropped": len(train) + len(val) - len(train_tok) - len(val_tok),
              "val_scenes": held_scenes,
              "tokens_median": lengths[len(lengths) // 2], "tokens_max": lengths[-1]}
    print(json.dumps(report, ensure_ascii=False))
    if args.dry_run:
        return

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer,
                              TrainingArguments)
    model = AutoModelForCausalLM.from_pretrained(base["name"], revision=base.get("revision"),
                                                 torch_dtype=getattr(torch, base["dtype"]))
    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"],
                                             lora_dropout=lora["dropout"],
                                             target_modules=lora["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    tr = cfg["training"]
    training_args = TrainingArguments(
        output_dir=str(args.output), learning_rate=tr["learning_rate"], lr_scheduler_type=tr["lr_scheduler"],
        warmup_ratio=tr["warmup_ratio"], num_train_epochs=args.epochs or tr["epochs"],
        per_device_train_batch_size=tr["per_device_batch_size"],
        per_device_eval_batch_size=tr["per_device_batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"], weight_decay=tr["weight_decay"],
        seed=tr["seed"], bf16=base["dtype"] == "bfloat16" and torch.cuda.is_available(),
        eval_strategy="epoch" if val_tok else "no", save_strategy="epoch",
        # Train examples include shuffled copies of the same decisions: keep
        # the epoch with the lowest validation loss, not the last one.
        load_best_model_at_end=bool(val_tok), metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2,
        logging_steps=10, report_to=[])
    trainer = Trainer(model=model, args=training_args, train_dataset=train_tok, eval_dataset=val_tok or None,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100))
    trainer.train()
    model.save_pretrained(args.output / "adapter")
    from athome.training.adapter_meta import data_version, write_meta
    write_meta(args.output / "adapter", adapter=args.data.name.replace(".jsonl", ""), method="SFT",
               base_model=base, lora=lora, data={**data_version(args.data.parent.parent / "manifest.json"),
                                                  "file": str(args.data), "train": report["train"],
                                                  "val": report["val"], "val_scenes": report["val_scenes"]},
               training=tr, results={"best_eval_loss": trainer.state.best_metric,
                                     "best_checkpoint": trainer.state.best_model_checkpoint})
    (args.output / "train_report.json").write_text(json.dumps(
        {**report, "config": cfg, "best_checkpoint": trainer.state.best_model_checkpoint,
         "best_eval_loss": trainer.state.best_metric, "log_history": trainer.state.log_history},
        ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
