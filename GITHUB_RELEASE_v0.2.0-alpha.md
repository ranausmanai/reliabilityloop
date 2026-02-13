## ReliabilityLoop v0.2.0-alpha

First alpha release of ReliabilityLoop, a verifier-driven framework for local LLM reliability and adaptive inference.

### What is included

- Reliability benchmark runner for:
  - JSON schema + expected-field checks
  - SQL execution correctness on SQLite fixtures
  - Python unit-test correctness
- Policy routing controls:
  - `--policy-json`, `--policy-sql`, `--policy-code`
- Adaptive inference controls:
  - `--best-of-k`
  - per-task `--best-of-k-json/sql/code`
  - per-task `--max-tokens-json/sql/code`
- Verified memory support:
  - `--memory-file`, `--memory-top-k`
  - run artifact: `wins.jsonl`
- Canonical benchmark split:
  - `eval/reliability_v1_60.jsonl` (60 tasks, 20 per type)

### Core command

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

### Artifacts generated per run

- `summary.json`
- `leaderboard.md`
- `samples.jsonl`
- `wins.jsonl`

### Links

- Repo: https://github.com/ranausmanai/reliabilityloop
- Dataset: https://huggingface.co/datasets/ranausmans/reliabilityloop-v1

### Notes

This is an alpha framework release focused on reproducible runtime reliability techniques, not model weight training.
