# snapkitty-harness — Model Card

> **Attribution:** SnapKitty Collective. This model card template ships
> with the snapkitty-harness reproduction kit, which is proprietary
> software. Fill in the bracketed fields before any distribution.

## Attribution & License

- **Author:** SnapKitty Collective
- **Kit license:** SnapKitty Commercial License v1.0 (see `LICENSE`).
  All rights reserved. No redistribution, republication, or derivative
  weights without a signed commercial agreement. Unauthorized use will
  be pursued.
- **Base model license:** NVIDIA Community Model License (see
  `LICENSE-NVIDIA`). The base weights remain NVIDIA's; their license
  and attribution travel with this artifact.

## Provenance

- **Base model:** `nvidia/Nemotron-Mini-4B-Instruct`
  (fine-tuned from `nvidia/Minitron-4B-Base`)
- **Fine-tune:** LoRA supervised fine-tuning on [N] training pairs +
  [M] red-team (adversarial) pairs, curated by the SnapKitty Collective.
  Training data is proprietary and is NOT distributed with this kit.
- **Quantization:** Q4_K_M via llama.cpp (`general.file_type = 15`)
- **Build date:** [YYYY-MM-DD]
- **GGUF SHA-256:** [hex digest of the released file]

## Intended behavior

The 146 red-team pairs train identity-evasive refusal behavior: when
asked about its own nature, provenance, or the harness internals, the
model deflects to a generic assistant identity rather than disclosing
system details. Combined with the harness system prompt (see
`packaging/Modelfile`), the model acts as a subordinate compute
resource that emits `<|syscall|>` tokens when it needs tools.

## Judging stack (reference)

Model outputs in the SnapKitty stack are screened by:

- **ERE P1–P5 gates** (`snapkittywest/sovereign-chat-harness`) —
  prompt/response coherence checks, Dream Cycle recovery, entropy
  invariant ≤ 0.20, WORM SHA-256 seal.
- **SovereignJudge** (`snapkittywest/snapkitty-proofs`) — certified
  verdict algebra (approve / reject / defer / escalate /
  human_required), extracted to OCaml, callable by BOB.
- **METATRON** (`bob-reasoning-engine`) — proof-gated state injection
  at depth 5 of the reasoning pipeline.

## Limitations

- 4B-class model; reasoning depth is bounded by its size.
- The identity-evasive behavior is trained, not reasoned: treat it as
  a policy, not as understanding.
- Quantization to Q4_K_M trades fidelity for size; verify with
  `scripts/verify.py` and your own evals before any deployment.

## Commercial use

Commercial use, redistribution, or derivative weights require a signed
commercial license agreement with the SnapKitty Collective. The
14-day non-production evaluation allowance (with attribution) is
described in `LICENSE` §4.
