# ReliabilityLoop Reliability Benchmark v1

This benchmark measures whether local LLM outputs are *actually usable* for production-like tasks.

## Scope

- Canonical split size: 60 tasks (20 per kind)
- Task families:
  - `json`: schema-constrained structured extraction
  - `sql`: text-to-SQL with execution-based correctness
  - `codestub`: Python function generation with unit tests
- Default task file: `eval/reliability_v1_60.jsonl`
- Default CLI entrypoint: `reliabilityloop reliability`

## Scoring

- `baseline_ok_rate`: direct output validity/correctness
- `contract_ok_rate`: contract-first output validity/correctness
- `avg_latency_s`: average generation + validation latency
- `retry_rate`: fraction requiring repair retry
- `success_after_retry_rate`: fraction of retried samples that end up correct

Per-task-family scores are also reported for `json`, `sql`, and `codestub`.

## Oracle Rules

- JSON:
  - Must parse as JSON
  - Must satisfy schema
  - Must satisfy optional expected value checks
- SQL:
  - Must be a single `SELECT`
  - Query is executed on in-memory SQLite fixtures
  - Output columns/rows compared against expected results
- Code:
  - Must parse as Python
  - Executed in a restricted runtime
  - Named entrypoint function must pass task unit tests

## Reproducibility

Run:

```bash
PYTHONPATH=src python -m autoquality.cli reliability \
  --backend ollama \
  --models qwen2.5-coder:1.5b,llama3.2:3b \
  --limit 6 \
  --max-tokens 128 \
  --repair-attempts 1
```

Artifacts:

- `eval/reliability_runs/<timestamp>/summary.json`
- `eval/reliability_runs/<timestamp>/leaderboard.md`
- `eval/reliability_runs/<timestamp>/samples.jsonl`
