# Evaluation

This directory contains benchmark runners and benchmark data for TestGate.

## Canonical Reliability Benchmark

- Spec: `eval/RELIABILITY_V1_SPEC.md`
- Task split: `eval/reliability_v1_60.jsonl` (60 tasks, 20 per kind)

Kinds:
- `json`: schema + expected-field checks
- `sql`: executable SQL correctness on SQLite fixtures
- `codestub`: Python unit-test correctness

## Run (Recommended)

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

Artifacts are written to `eval/reliability_runs/<timestamp>/`.

## Quick Pipeline Test (No model needed)

```bash
reliabilityloop reliability --backend mock --model mock-fast --limit 6
```

## Other Existing Eval Scripts

- HumanEval runner: `eval/humaneval_runner.py`
- Contract benchmark: `eval/contract_bench.py`
