# autoquality HumanEval report

- fast model: `qwen2.5-coder:14b`
- slow model: `qwen2.5-coder:14b`
- tasks: `50`

- completion normalization: `True`

| mode | pass@1 | gen seconds | eval seconds | total seconds | eval samples |
|---|---:|---:|---:|---:|---|
| `baseline_fast` | 0.6800 | 669.3 | 11.3 | 680.6 | `eval/runs/baseline-14b/samples.baseline_fast.normalized.jsonl` |

| mode | slow used | repairs attempted | repairs applied |
|---|---:|---:|---:|
| `baseline_fast` | 0 | 0 | 0 |

| mode | gate fast pass | gate slow pass |
|---|---:|---:|
| `baseline_fast` | 0 | 0 |

Artifacts:
- `eval/runs/baseline-14b/samples.baseline_fast.jsonl`
- `eval/runs/baseline-14b/meta.baseline_fast.jsonl`
- `eval/runs/baseline-14b/metrics.baseline_fast.json`
