# autoquality HumanEval report

- fast model: `qwen2.5-coder:7b`
- slow model: `qwen2.5-coder:14b`
- tasks: `100`

- completion normalization: `True`

| mode | pass@1 | gen seconds | eval seconds | total seconds | eval samples |
|---|---:|---:|---:|---:|---|
| `baseline_fast` | 0.6200 | 757.6 | 26.5 | 784.1 | `eval/runs/20260206-041726/samples.baseline_fast.normalized.jsonl` |
| `testgate` | 0.7900 | 1957.7 | 21.5 | 1979.2 | `eval/runs/20260206-041726/samples.testgate.normalized.jsonl` |

| mode | slow used | repairs attempted | repairs applied |
|---|---:|---:|---:|
| `baseline_fast` | 0 | 0 | 0 |
| `testgate` | 38 | 0 | 0 |

| mode | gate fast pass | gate slow pass |
|---|---:|---:|
| `baseline_fast` | 0 | 0 |
| `testgate` | 62 | 17 |

Artifacts:
- `eval/runs/20260206-041726/samples.baseline_fast.jsonl`
- `eval/runs/20260206-041726/meta.baseline_fast.jsonl`
- `eval/runs/20260206-041726/metrics.baseline_fast.json`
- `eval/runs/20260206-041726/samples.testgate.jsonl`
- `eval/runs/20260206-041726/meta.testgate.jsonl`
- `eval/runs/20260206-041726/metrics.testgate.json`
