# autoquality HumanEval report

- fast model: `qwen2.5-coder:14b`
- slow model: `glm-4.7-flash:latest`
- tasks: `50`

- completion normalization: `True`

| mode | pass@1 | gen seconds | eval seconds | total seconds | eval samples |
|---|---:|---:|---:|---:|---|
| `baseline_fast` | 0.6800 | 769.8 | 11.1 | 780.8 | `eval/runs/20260205-162415/samples.baseline_fast.normalized.jsonl` |
| `escalate` | 0.6800 | 1001.0 | 11.0 | 1011.9 | `eval/runs/20260205-162415/samples.escalate.normalized.jsonl` |
| `repair` | 0.6800 | 949.4 | 11.1 | 960.5 | `eval/runs/20260205-162415/samples.repair.normalized.jsonl` |

| mode | slow used | repairs attempted | repairs applied |
|---|---:|---:|---:|
| `baseline_fast` | 0 | 0 | 0 |
| `escalate` | 2 | 0 | 0 |
| `repair` | 0 | 2 | 2 |

Artifacts:
- `eval/runs/20260205-162415/samples.baseline_fast.jsonl`
- `eval/runs/20260205-162415/meta.baseline_fast.jsonl`
- `eval/runs/20260205-162415/metrics.baseline_fast.json`
- `eval/runs/20260205-162415/samples.escalate.jsonl`
- `eval/runs/20260205-162415/meta.escalate.jsonl`
- `eval/runs/20260205-162415/metrics.escalate.json`
- `eval/runs/20260205-162415/samples.repair.jsonl`
- `eval/runs/20260205-162415/meta.repair.jsonl`
- `eval/runs/20260205-162415/metrics.repair.json`
