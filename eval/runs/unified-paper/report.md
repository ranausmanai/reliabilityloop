# autoquality HumanEval report

- fast model: `qwen2.5-coder:7b`
- slow model: `qwen2.5-coder:14b`
- tasks: `50`

- completion normalization: `True`

| mode | pass@1 | gen seconds | eval seconds | total seconds | eval samples |
|---|---:|---:|---:|---:|---|
| `baseline_fast` | 0.5400 | 543.2 | 12.4 | 555.5 | `eval/runs/unified-paper/samples.baseline_fast.normalized.jsonl` |
| `testgate` | 0.7200 | 1093.4 | 15.5 | 1108.9 | `eval/runs/unified-paper/samples.testgate.normalized.jsonl` |

| mode | slow used | repairs attempted | repairs applied |
|---|---:|---:|---:|
| `baseline_fast` | 0 | 0 | 0 |
| `testgate` | 23 | 0 | 0 |

| mode | gate fast pass | gate slow pass |
|---|---:|---:|
| `baseline_fast` | 0 | 0 |
| `testgate` | 27 | 9 |

Artifacts:
- `eval/runs/unified-paper/samples.baseline_fast.jsonl`
- `eval/runs/unified-paper/meta.baseline_fast.jsonl`
- `eval/runs/unified-paper/metrics.baseline_fast.json`
- `eval/runs/unified-paper/samples.testgate.jsonl`
- `eval/runs/unified-paper/meta.testgate.jsonl`
- `eval/runs/unified-paper/metrics.testgate.json`
