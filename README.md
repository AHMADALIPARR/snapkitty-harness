# snapkitty-harness — Reproduction Kit

**Copyright (C) 2026 SnapKitty Collective — proprietary.**
This kit is released under the **SnapKitty Commercial License v1.0**
(see `LICENSE`). All rights reserved. No redistribution, republication,
or derivative weights without a signed commercial agreement.
Unauthorized use will be pursued. Attribution to "SnapKitty Collective"
is mandatory in any authorized use.

## What this is

A complete, from-scratch reproduction kit for the
`snapkitty-harness.Q4_K_M.gguf` artifact
([public page](https://huggingface.co/Snapkitty/snapkitty-harness)).
It contains everything needed to re-train, re-merge, re-convert,
re-quantize, and re-verify the model — except the proprietary training
pairs themselves, which you supply.

This kit does NOT contain trained weights. It is the recipe, not the cake.

## Provenance

Recreated from the public artifact's own metadata (parsed directly
from the GGUF header — no guesswork):

- Base: `nvidia/Nemotron-Mini-4B-Instruct` (from `nvidia/Minitron-4B-Base`)
- Architecture `nemotron`, `general.file_type = 15` (Q4_K_M)
- Fine-tuned on 353 training pairs + 146 red-team pairs
  ("adversarial evaluation framework")
- Served with the exact Modelfile in `packaging/Modelfile`

## Layout

```
LICENSE                 SnapKitty Commercial License v1.0 (STRICT)
LICENSE-NVIDIA          base-model license compatibility notice
README.md               this file
spec/TRAINING_SPEC.md   exact reproduction recipe
scripts/
  train.py              LoRA SFT (transformers + PEFT + TRL)
  merge_and_convert.sh  merge adapters / convert / quantize to Q4_K_M
  verify.py             GGUF header verifier (pure stdlib)
  Makefile              pipeline stages: check train merge convert quantize verify all
tests/
  test_verify.py        unit tests (synthetic GGUF header, no downloads)
packaging/
  Modelfile             exact FROM + system prompt from the original
  model-card.md         attribution + commercial-license + provenance template
data/
  README.md             required layout + schema
  sample_synthetic.jsonl  8-row SYNTHETIC schema demo (never train on this)
```

## Quick start

```bash
# 1. check everything parses / tests pass
make -C scripts check

# 2. place your REAL pairs (see data/README.md):
#      data/train.jsonl      (353 rows)
#      data/redteam.jsonl    (146 rows)

# 3. train LoRA adapters (needs a GPU)
make -C scripts train DATA_DIR=data OUT=out/adapter

# 4. merge, convert, quantize (needs a llama.cpp checkout)
make -C scripts merge ADAPTER=out/adapter OUT=out/merged
make -C scripts convert MERGED=out/merged LLAMACPP=/path/to/llama.cpp
make -C scripts quantize LLAMACPP=/path/to/llama.cpp

# 5. verify the artifact
make -C scripts verify GGUF=out/gguf/snapkitty-harness.Q4_K_M.gguf
```

## Why it behaves the way it does

The model's signature trait — deflecting questions about its own
identity, the harness, and system internals — is trained, not
accidental. The 146 red-team pairs teach identity-evasive refusal
patterns, and the system prompt in `packaging/Modelfile` frames the
model as a subordinate compute resource ("You are not the authority.
The harness enforces all rules.") that emits `<|syscall|>` tokens when
it needs tools. Reproduce both the pairs and the prompt to reproduce
the behavior.

## Hardware reality

- LoRA SFT on a 4B model needs a real GPU box (≥16 GB VRAM).
- Merge / convert / quantize run on CPU (~16 GB RAM).
- **This kit was NOT executed end-to-end on the build host (no GPU
  here).** No benchmarks are claimed anywhere in this repo. What was
  verified: every script parses/compiles, and `verify.py` is
  unit-tested against a synthetic GGUF header.

## License compatibility (read this)

The final GGUF is a layered artifact. NVIDIA owns the foundation
(the base weights, under the NVIDIA Community Model License).
SnapKitty owns the renovation (fine-tune, curation, harness config,
this kit). SnapKitty's commercial license covers ONLY SnapKitty's
added IP — it does not strip NVIDIA's terms. `scripts/verify.py`
asserts the NVIDIA license fields are present in the header so they
cannot be silently dropped. Full details: `LICENSE-NVIDIA`.

## Related SnapKitty stack (reference only — not included)

- **ERE P1–P5 gates** (`snapkittywest/sovereign-chat-harness`) —
  output screening: coherence checks, Dream Cycle recovery, entropy
  invariant, WORM seal.
- **SovereignJudge** (`snapkittywest/snapkitty-proofs`) — certified
  verdict algebra, extracted to OCaml, called by BOB.
- **METATRON** (`bob-reasoning-engine`) — proof-gated state injection.
