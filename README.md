# ReliabilityLoop

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/Status-Alpha-orange.svg)](RELEASE.md)

Verifier-driven framework for improving local LLM reliability with adaptive inference.

ReliabilityLoop evaluates whether model outputs actually work (not just look plausible), then applies runtime strategies to improve reliability under cost and latency constraints.

## Why ReliabilityLoop

- Executable reliability checks: JSON schema, SQL execution, Python unit tests
- Policy routing: choose `baseline_first` or `contract_first` per task type
- Adaptive compute: per-task `best-of-k` and per-task token budgets
- Verified memory reuse: reuse proven outputs from earlier runs (`wins.jsonl`)
- Reproducible artifacts: every run outputs `summary.json`, `leaderboard.md`, `samples.jsonl`, `wins.jsonl`

## Benchmark Scope (v1)

Canonical split: `eval/reliability_v1_60.jsonl`

- 20 JSON tasks
- 20 SQL tasks
- 20 code tasks

Spec: `eval/RELIABILITY_V1_SPEC.md`

## Baseline Result (Example)

From `examples/leaderboard_60_baseline.md` on `qwen2.5-coder:0.5b`:

| model | policy reliability | json | sql | code | policy latency (s) |
|---|---:|---:|---:|---:|---:|
| `qwen2.5-coder:0.5b` | 0.867 | 1.000 | 1.000 | 0.600 | 2.428 |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```bash
reliabilityloop reliability \
  --backend ollama \
  --model qwen2.5-coder:0.5b \
  --prompts-file eval/reliability_v1_60.jsonl \
  --limit 60 \
  --max-tokens 96 \
  --policy-json contract_first \
  --policy-sql baseline_first \
  --policy-code baseline_first
```

Outputs are saved to `eval/reliability_runs/<timestamp>/`.

## Adaptive Inference Example

```bash
reliabilityloop reliability \
  --backend ollama \
  --model qwen2.5-coder:0.5b \
  --prompts-file eval/reliability_v1_60.jsonl \
  --limit 60 \
  --max-tokens 96 \
  --max-tokens-json 256 \
  --best-of-k 1 \
  --policy-json contract_first \
  --policy-sql baseline_first \
  --policy-code baseline_first
```

## Verified Memory Example

```bash
# first run
RUN_A=$(reliabilityloop reliability \
  --backend ollama \
  --model qwen2.5-coder:0.5b \
  --prompts-file eval/reliability_v1_60.jsonl \
  --limit 60 | sed -n 's/^- outdir: //p')

# second run with memory
reliabilityloop reliability \
  --backend ollama \
  --model qwen2.5-coder:0.5b \
  --prompts-file eval/reliability_v1_60.jsonl \
  --limit 60 \
  --memory-file "$RUN_A/wins.jsonl" \
  --memory-top-k 2
```

## Hugging Face Dataset

- https://huggingface.co/datasets/ranausmans/reliabilityloop-v1

## Release Docs

- `RELEASE.md`
- `CHANGELOG.md`

## License

MIT
