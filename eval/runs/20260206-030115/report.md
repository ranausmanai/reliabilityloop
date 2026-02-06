# autoquality HumanEval report

- fast model: `qwen2.5-coder:7b`
- slow model: `qwen2.5-coder:14b`
- tasks: `50`

- completion normalization: `True`

| mode | pass@1 | gen seconds | eval seconds | total seconds | eval samples |
|---|---:|---:|---:|---:|---|
| `baseline_fast` | 0.5400 | 357.3 | 10.7 | 368.0 | `eval/runs/20260206-030115/samples.baseline_fast.normalized.jsonl` |
| `escalate` | 0.5200 | 873.6 | 13.7 | 887.4 | `eval/runs/20260206-030115/samples.escalate.normalized.jsonl` |
| `repair` | 0.4200 | 942.6 | 11.6 | 954.2 | `eval/runs/20260206-030115/samples.repair.normalized.jsonl` |

| mode | slow used | repairs attempted | repairs applied |
|---|---:|---:|---:|
| `baseline_fast` | 0 | 0 | 0 |
| `escalate` | 18 | 0 | 0 |
| `repair` | 0 | 18 | 15 |

Artifacts:
- `eval/runs/20260206-030115/samples.baseline_fast.jsonl`
- `eval/runs/20260206-030115/meta.baseline_fast.jsonl`
- `eval/runs/20260206-030115/metrics.baseline_fast.json`
- `eval/runs/20260206-030115/samples.escalate.jsonl`
- `eval/runs/20260206-030115/meta.escalate.jsonl`
- `eval/runs/20260206-030115/metrics.escalate.json`
- `eval/runs/20260206-030115/samples.repair.jsonl`
- `eval/runs/20260206-030115/meta.repair.jsonl`
- `eval/runs/20260206-030115/metrics.repair.json`
