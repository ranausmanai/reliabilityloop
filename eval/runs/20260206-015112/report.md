# autoquality HumanEval report

- fast model: `qwen2.5-coder:7b`
- slow model: `qwen2.5-coder:14b`
- tasks: `50`

- completion normalization: `True`

| mode | pass@1 | gen seconds | eval seconds | total seconds | eval samples |
|---|---:|---:|---:|---:|---|
| `baseline_fast` | 0.5400 | 369.0 | 11.0 | 380.0 | `eval/runs/20260206-015112/samples.baseline_fast.normalized.jsonl` |
| `escalate` | 0.5400 | 419.2 | 11.3 | 430.5 | `eval/runs/20260206-015112/samples.escalate.normalized.jsonl` |
| `repair` | 0.5400 | 447.8 | 11.5 | 459.4 | `eval/runs/20260206-015112/samples.repair.normalized.jsonl` |

| mode | slow used | repairs attempted | repairs applied |
|---|---:|---:|---:|
| `baseline_fast` | 0 | 0 | 0 |
| `escalate` | 2 | 0 | 0 |
| `repair` | 0 | 2 | 1 |

Artifacts:
- `eval/runs/20260206-015112/samples.baseline_fast.jsonl`
- `eval/runs/20260206-015112/meta.baseline_fast.jsonl`
- `eval/runs/20260206-015112/metrics.baseline_fast.json`
- `eval/runs/20260206-015112/samples.escalate.jsonl`
- `eval/runs/20260206-015112/meta.escalate.jsonl`
- `eval/runs/20260206-015112/metrics.escalate.json`
- `eval/runs/20260206-015112/samples.repair.jsonl`
- `eval/runs/20260206-015112/meta.repair.jsonl`
- `eval/runs/20260206-015112/metrics.repair.json`
