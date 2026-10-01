# data/ — training pair layout

The real training pairs are proprietary and are NOT in this kit. The
public copy at `Snapkitty/sovereign-training-corpus` is a degraded
stand-in and must not be used.

## Required layout (you supply the real pairs)

```
data/
  train.jsonl      # 353 rows — general SFT pairs
  redteam.jsonl    # 146 rows — adversarial / identity-evasion pairs
```

## Row schema

One JSON object per line:

```json
{"instruction": "user prompt", "output": "desired completion", "domain": "redteam"}
```

- `instruction` (string, required)
- `output` (string, required)
- `domain` (string, optional): `redteam`, `coding`, `darkai`,
  `architecture`, `security`, `financial`, `omega`, …

## Synthetic sample

`sample_synthetic.jsonl` contains 8 rows that demonstrate the schema.
Every row is labeled `"synthetic": true` and the text is marked
SYNTHETIC. `scripts/train.py` REFUSES to train on rows carrying the
synthetic marker. Never present the sample as real training data.
