# autoquality HumanEval report

- fast model: `qwen2.5-coder:1.5b`
- slow model: `qwen2.5-coder:14b`
- tasks: `100`

- completion normalization: `True`

| mode | pass@1 | gen seconds | eval seconds | total seconds | eval samples |
|---|---:|---:|---:|---:|---|
| `baseline_fast` | 0.5000 | 511.0 | 29.3 | 540.2 | `eval/runs/20260206-041930/samples.baseline_fast.normalized.jsonl` |
| `testgate` | 0.8200 | 1896.5 | 26.9 | 1923.4 | `eval/runs/20260206-041930/samples.testgate.normalized.jsonl` |

| mode | slow used | repairs attempted | repairs applied |
|---|---:|---:|---:|
| `baseline_fast` | 0 | 0 | 0 |
| `testgate` | 50 | 0 | 0 |

| mode | gate fast pass | gate slow pass |
|---|---:|---:|
| `baseline_fast` | 0 | 0 |
| `testgate` | 50 | 32 |

Artifacts:
- `eval/runs/20260206-041930/samples.baseline_fast.jsonl`
- `eval/runs/20260206-041930/meta.baseline_fast.jsonl`
- `eval/runs/20260206-041930/metrics.baseline_fast.json`
- `eval/runs/20260206-041930/samples.testgate.jsonl`
- `eval/runs/20260206-041930/meta.testgate.jsonl`
- `eval/runs/20260206-041930/metrics.testgate.json`
