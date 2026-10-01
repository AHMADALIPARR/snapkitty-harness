# SPDX-License-Identifier: LicenseRef-SnapKitty-Commercial-1.0
# Copyright (C) 2026 SnapKitty Collective
"""LoRA supervised fine-tuning for the snapkitty-harness reproduction.

Trains LoRA adapters on data/train.jsonl (general pairs) +
data/redteam.jsonl (adversarial pairs) on top of
nvidia/Nemotron-Mini-4B-Instruct.

Requires a GPU. Not executable on CPU-only hosts in practice.

Usage:
    python train.py --data-dir data --base-model nvidia/Nemotron-Mini-4B-Instruct \
        --out out/adapter
"""

from __future__ import annotations

import argparse
import json
import os
import sys


SAMPLE_MARKER = "SYNTHETIC"


def load_pairs(data_dir: str) -> list[dict]:
    rows: list[dict] = []
    real_files = 0
    for fname in ("train.jsonl", "redteam.jsonl"):
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            print(f"[!] missing {path}", file=sys.stderr)
            continue
        real_files += 1
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"[!] {path}:{lineno}: invalid JSON: {exc}"
                    ) from exc
                if SAMPLE_MARKER in json.dumps(obj):
                    raise SystemExit(
                        f"[!] {path}:{lineno}: refusing to train on rows "
                        f"labeled {SAMPLE_MARKER}. Supply the real pairs."
                    )
                if "instruction" not in obj or "output" not in obj:
                    raise SystemExit(
                        f"[!] {path}:{lineno}: row must contain "
                        f"'instruction' and 'output'"
                    )
                rows.append(obj)
    if real_files == 0:
        raise SystemExit(
            "[!] no data files found. Place the real pairs at "
            "data/train.jsonl and data/redteam.jsonl (see data/README.md)."
        )
    return rows


def to_chat_text(row: dict) -> str:
    instruction = row["instruction"].strip()
    output = row["output"].strip()
    return (
        f"<|user|>\n{instruction}\n<|assistant|>\n{output}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data",
                   help="directory holding train.jsonl + redteam.jsonl")
    p.add_argument("--base-model",
                   default="nvidia/Nemotron-Mini-4B-Instruct",
                   help="base model id (must match GGUF provenance)")
    p.add_argument("--out", default="out/adapter",
                   help="output directory for LoRA adapters")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  TrainingArguments)
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "[!] missing training dependency: "
            f"{exc}. Install: pip install torch transformers datasets "
            "peft trl accelerate bitsandbytes"
        ) from exc

    if not torch.cuda.is_available():
        print("[!] WARNING: no CUDA device detected. Training a 4B model "
              "on CPU is not practical; continuing anyway.", file=sys.stderr)

    rows = load_pairs(args.data_dir)
    print(f"[*] loaded {len(rows)} pairs from {args.data_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    dataset = Dataset.from_list(
        [{"text": to_chat_text(r)} for r in rows]
    )

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="adamw_torch",
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=100,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        max_seq_length=args.max_seq_len,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"[*] adapters saved to {args.out}")


if __name__ == "__main__":
    main()
