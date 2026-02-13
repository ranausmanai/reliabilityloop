# ReliabilityLoop Framework Release (v0.2.0)

This repo is now packaged as a reproducible framework for local LLM reliability.

## What You Can Run

- Benchmark local models on executable tasks (`json`, `sql`, `codestub`)
- Apply policy routing (`baseline_first` vs `contract_first`)
- Apply adaptive compute (`best-of-k`, per-task budgets)
- Reuse verified memory from prior runs (`wins.jsonl`)

## Canonical Benchmark

- Default split: `eval/reliability_v1_60.jsonl`
- Verifier spec: `eval/RELIABILITY_V1_SPEC.md`

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

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

Artifacts:
- `eval/reliability_runs/<timestamp>/summary.json`
- `eval/reliability_runs/<timestamp>/leaderboard.md`
- `eval/reliability_runs/<timestamp>/samples.jsonl`
- `eval/reliability_runs/<timestamp>/wins.jsonl`

## Example Artifacts

- `examples/leaderboard_60_baseline.md`
- `examples/summary_60_baseline.json`
