# snapkitty-harness — Training & Reproduction Spec

Exact recipe for reproducing the `snapkitty-harness.Q4_K_M.gguf` artifact
from its public metadata. This spec describes the pipeline; the scripts
in `../scripts/` implement it.

## 1. Provenance (verified from the original artifact)

| Field | Value |
|---|---|
| Public artifact | https://huggingface.co/Snapkitty/snapkitty-harness |
| File | `snapkitty-harness.Q4_K_M.gguf` |
| GGUF `general.architecture` | `nemotron` |
| GGUF `general.name` | `Minitron 4B Instruct` |
| GGUF `general.base_model.0.name` | `Minitron 4B Base` |
| GGUF `general.base_model.0.repo_url` | `https://huggingface.co/nvidia/Minitron-4B-Base` |
| GGUF `general.file_type` | `15` (= Q4_K_M) |
| Base model for fine-tune | `nvidia/Nemotron-Mini-4B-Instruct` |
| Base license | NVIDIA Community Model License (see `../LICENSE-NVIDIA`) |
| Model card claim | fine-tuned on 353 training pairs + 146 red-team pairs |

## 2. Dataset layout

The user's real pairs are NOT in this kit (the public
`Snapkitty/sovereign-training-corpus` copy is a degraded stand-in).
Place the real data here before training:

```
data/
  train.jsonl      # 353 rows — general SFT pairs
  redteam.jsonl    # 146 rows — adversarial / identity-evasion pairs
```

Row schema (JSONL, one object per line):

```json
{"instruction": "…", "output": "…", "domain": "redteam"}
```

- `instruction` (string, required): the user prompt.
- `output` (string, required): the desired assistant completion.
- `domain` (string, optional): e.g. `redteam`, `coding`, `darkai`,
  `architecture`, `security`, `financial`, `omega`.

`data/sample_synthetic.jsonl` (8 rows) demonstrates the schema. It is
clearly labeled SYNTHETIC and must never be presented as real training
data. `train.py` refuses to train on the sample alone.

## 3. Training — LoRA SFT (defaults)

Method: parameter-efficient supervised fine-tuning (LoRA) on top of the
frozen base, then merge adapters into the base weights. This matches a
4B-class model trained on ~500 pairs: full fine-tuning is unnecessary
and LoRA keeps the run on a single GPU.

| Hyperparameter | Default | Notes |
|---|---|---|
| Base model | `nvidia/Nemotron-Mini-4B-Instruct` | exact base per GGUF metadata |
| LoRA rank `r` | 16 | |
| LoRA alpha | 32 | |
| LoRA dropout | 0.05 | |
| Target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | all attention + MLP linears |
| Learning rate | 2e-4 | cosine schedule |
| Epochs | 3 | ~500 pairs: more epochs overfit |
| Batch size (per device) | 4 | raise if VRAM allows |
| Gradient accumulation | 4 | effective batch 16 |
| Max sequence length | 2048 | |
| Optimizer | `adamw_torch` | |
| Warmup ratio | 0.05 | |
| Precision | bf16 (fp16 fallback) | |
| Seed | 42 | |

Run:

```bash
make -C scripts train DATA_DIR=data BASE_MODEL=nvidia/Nemotron-Mini-4B-Instruct OUT=out/adapter
```

## 4. Merge adapters into base

```bash
make -C scripts merge ADAPTER=out/adapter BASE_MODEL=nvidia/Nemotron-Mini-4B-Instruct OUT=out/merged
```

This loads the base in fp16/bf16, applies `merge_and_unload()`, and
saves a full fp16 HF checkpoint. VRAM needed: ~10–12 GB.

## 5. Convert to GGUF

```bash
make -C scripts convert MERGED=out/merged LLAMACPP=/path/to/llama.cpp OUT=out/gguf
```

Runs llama.cpp's `convert_hf_to_gguf.py` on the merged checkpoint,
producing `snapkitty-harness-f16.gguf`. Requires a llama.cpp checkout
(`convert_hf_to_gguf.py` needs `gguf`, `numpy`, `torch`, `transformers`).

## 6. Quantize to Q4_K_M

```bash
make -C scripts quantize LLAMACPP=/path/to/llama.cpp GGUF=out/gguf/snapkitty-harness-f16.gguf OUT=out/gguf
```

Runs `llama-quantize <f16.gguf> <out.gguf> Q4_K_M`. Build
`llama-quantize` from the same llama.cpp checkout first.

## 7. Verify the artifact

```bash
make -C scripts verify GGUF=out/gguf/snapkitty-harness.Q4_K_M.gguf
```

`verify.py` parses the GGUF KV header (pure `struct`, no torch) and
asserts the expected metadata:

- magic `GGUF`
- `general.architecture == "nemotron"`
- `general.file_type == 15` (Q4_K_M)
- `general.name` mentions Minitron 4B Instruct
- `general.base_model.0.repo_url` points at `nvidia/Minitron-4B-Base`
- NVIDIA license fields present (`general.license.name`,
  `general.license.link`)

Any failure exits non-zero. The NVIDIA license fields are asserted
deliberately: they must travel with the artifact (see LICENSE-NVIDIA).

## 8. Serve (Ollama)

```bash
cp packaging/Modelfile out/gguf/Modelfile
ollama create snapkitty-harness -f out/gguf/Modelfile
```

The Modelfile carries the exact system prompt from the original repo.
It is what frames the model as a subordinate compute resource and,
together with the red-team pairs, produces the identity-evasive
behavior. Fill in `packaging/model-card.md` before any distribution.

## 9. Hardware reality

- Training (LoRA SFT on a 4B model): one GPU with ≥16 GB VRAM
  (24 GB comfortable). CPU-only training is not practical.
- Merge + convert + quantize: CPU is fine, needs ~16 GB RAM.
- This kit was NOT executed end-to-end on the build host (no GPU).
  No benchmarks are claimed. The scripts are syntax-checked and the
  GGUF verifier is unit-tested against a synthetic header.
