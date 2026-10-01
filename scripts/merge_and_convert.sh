#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-SnapKitty-Commercial-1.0
# Copyright (C) 2026 SnapKitty Collective
#
# merge_and_convert.sh — merge LoRA adapters into the base model, convert
# the merged checkpoint to GGUF, and quantize to Q4_K_M.
#
# Stages (run individually or all):
#   merge     merge LoRA adapters -> full fp16 HF checkpoint
#   convert   run llama.cpp convert_hf_to_gguf.py -> f16 GGUF
#   quantize  run llama-quantize -> Q4_K_M GGUF
#
# Env / args:
#   ADAPTER     dir with LoRA adapters (default: out/adapter)
#   BASE_MODEL  base model id (default: nvidia/Nemotron-Mini-4B-Instruct)
#   MERGED      output dir for merged checkpoint (default: out/merged)
#   LLAMACPP    path to a llama.cpp checkout (required for convert/quantize)
#   F16GGUF     input f16 GGUF for quantize (default: out/gguf/snapkitty-harness-f16.gguf)
#   OUTGGUF     output dir for final GGUF (default: out/gguf)
#
# Usage:
#   ./merge_and_convert.sh merge
#   ./merge_and_convert.sh convert
#   ./merge_and_convert.sh quantize
#   ./merge_and_convert.sh all
set -euo pipefail

ADAPTER="${ADAPTER:-out/adapter}"
BASE_MODEL="${BASE_MODEL:-nvidia/Nemotron-Mini-4B-Instruct}"
MERGED="${MERGED:-out/merged}"
LLAMACPP="${LLAMACPP:-}"
F16GGUF="${F16GGUF:-out/gguf/snapkitty-harness-f16.gguf}"
OUTGGUF="${OUTGGUF:-out/gguf}"

stage_merge() {
    echo "[*] merging adapters: $ADAPTER into $BASE_MODEL"
    python3 - "$ADAPTER" "$BASE_MODEL" "$MERGED" <<'EOF'
import sys
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

adapter_dir, base_id, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.float16, trust_remote_code=True)
model = PeftModel.from_pretrained(base, adapter_dir)
merged = model.merge_and_unload()
merged.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
print(f"[*] merged checkpoint saved to {out_dir}")
EOF
}

stage_convert() {
    if [ -z "$LLAMACPP" ]; then
        echo "[!] LLAMACPP is not set (path to llama.cpp checkout)" >&2
        exit 1
    fi
    mkdir -p "$OUTGGUF"
    echo "[*] converting $MERGED -> $OUTGGUF/snapkitty-harness-f16.gguf"
    python3 "$LLAMACPP/convert_hf_to_gguf.py" \
        "$MERGED" \
        --outfile "$OUTGGUF/snapkitty-harness-f16.gguf" \
        --outtype f16
}

stage_quantize() {
    if [ -z "$LLAMACPP" ]; then
        echo "[!] LLAMACPP is not set (path to llama.cpp checkout)" >&2
        exit 1
    fi
    local quant="$LLAMACPP/build/bin/llama-quantize"
    if [ ! -x "$quant" ]; then
        quant="$LLAMACPP/llama-quantize"
    fi
    if [ ! -x "$quant" ]; then
        echo "[!] llama-quantize not found; build llama.cpp first" >&2
        exit 1
    fi
    echo "[*] quantizing $F16GGUF -> $OUTGGUF/snapkitty-harness.Q4_K_M.gguf"
    "$quant" "$F16GGUF" "$OUTGGUF/snapkitty-harness.Q4_K_M.gguf" Q4_K_M
}

case "${1:-}" in
    merge)    stage_merge ;;
    convert)  stage_convert ;;
    quantize) stage_quantize ;;
    all)      stage_merge; stage_convert; stage_quantize ;;
    *)
        echo "usage: $0 {merge|convert|quantize|all}" >&2
        exit 2
        ;;
esac
