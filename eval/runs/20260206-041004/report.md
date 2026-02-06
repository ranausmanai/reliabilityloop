# autoquality HumanEval report

- fast model: `qwen2.5-coder:7b`
- slow model: `qwen2.5-coder:14b`
- tasks: `10`

- completion normalization: `True`

| mode | pass@1 | gen seconds | eval seconds | total seconds | eval samples |
|---|---:|---:|---:|---:|---|
| `baseline_fast` | 0.8000 | 72.4 | 2.3 | 74.8 | `eval/runs/20260206-041004/samples.baseline_fast.normalized.jsonl` |
| `testgate` | 0.9000 | 136.9 | 2.7 | 139.6 | `eval/runs/20260206-041004/samples.testgate.normalized.jsonl` |

| mode | slow used | repairs attempted | repairs applied |
|---|---:|---:|---:|
| `baseline_fast` | 0 | 0 | 0 |
| `testgate` | 2 | 0 | 0 |

| mode | gate fast pass | gate slow pass |
|---|---:|---:|
| `baseline_fast` | 0 | 0 |
| `testgate` | 8 | 1 |

Artifacts:
- `eval/runs/20260206-041004/samples.baseline_fast.jsonl`
- `eval/runs/20260206-041004/meta.baseline_fast.jsonl`
- `eval/runs/20260206-041004/metrics.baseline_fast.json`
- `eval/runs/20260206-041004/samples.testgate.jsonl`
- `eval/runs/20260206-041004/meta.testgate.jsonl`
- `eval/runs/20260206-041004/metrics.testgate.json`
