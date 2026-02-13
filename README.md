# ReliabilityLoop

Verifier-driven framework for improving local LLM reliability with adaptive inference, built from TestGate research components.

## What TestGate Does

- Evaluates models on executable tasks (`json`, `sql`, `codestub`)
- Uses strict verifiers (schema checks, SQL execution, Python unit tests)
- Supports policy routing (`baseline_first` / `contract_first`)
- Supports adaptive compute (`best-of-k`, per-task budgets)
- Supports verified memory reuse (`wins.jsonl`)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart (Framework)

Run the canonical 60-task split:

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

Artifacts written to `eval/reliability_runs/<timestamp>/`:
- `summary.json`
- `leaderboard.md`
- `samples.jsonl`
- `wins.jsonl`

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
# 1) Run once and produce wins
RUN_A=$(reliabilityloop reliability \
  --backend ollama \
  --model qwen2.5-coder:0.5b \
  --prompts-file eval/reliability_v1_60.jsonl \
  --limit 60 | sed -n 's/^- outdir: //p')

# 2) Re-run with memory from verified wins
reliabilityloop reliability \
  --backend ollama \
  --model qwen2.5-coder:0.5b \
  --prompts-file eval/reliability_v1_60.jsonl \
  --limit 60 \
  --memory-file "$RUN_A/wins.jsonl" \
  --memory-top-k 2
```

## Benchmark Assets

- Spec: `eval/RELIABILITY_V1_SPEC.md`
- Canonical split: `eval/reliability_v1_60.jsonl`
- Example output: `examples/leaderboard_60_baseline.md`

## Release Docs

- `RELEASE.md`
- `CHANGELOG.md`

## License

MIT
