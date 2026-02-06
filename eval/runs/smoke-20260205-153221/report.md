# autoquality HumanEval report

- fast model: `glm-4.7-flash:latest`
- slow model: `qwen2.5-coder:14b`
- tasks: `2`

- completion normalization: `True`

| mode | pass@1 | gen seconds | eval seconds | total seconds | eval samples |
|---|---:|---:|---:|---:|---|
| `baseline_fast` | 1.0000 | 291.7 | 0.7 | 292.5 | `eval/runs/smoke-20260205-153221/samples.baseline_fast.normalized.jsonl` |
| `escalate` | 1.0000 | 109.2 | 0.7 | 109.9 | `eval/runs/smoke-20260205-153221/samples.escalate.normalized.jsonl` |
| `repair` | 1.0000 | 220.8 | 0.7 | 221.5 | `eval/runs/smoke-20260205-153221/samples.repair.normalized.jsonl` |

| mode | slow used | repairs attempted | repairs applied |
|---|---:|---:|---:|
| `baseline_fast` | 0 | 0 | 0 |
| `escalate` | 0 | 0 | 0 |
| `repair` | 0 | 0 | 0 |

Artifacts:
- `eval/runs/smoke-20260205-153221/samples.baseline_fast.jsonl`
- `eval/runs/smoke-20260205-153221/meta.baseline_fast.jsonl`
- `eval/runs/smoke-20260205-153221/metrics.baseline_fast.json`
- `eval/runs/smoke-20260205-153221/samples.escalate.jsonl`
- `eval/runs/smoke-20260205-153221/meta.escalate.jsonl`
- `eval/runs/smoke-20260205-153221/metrics.escalate.json`
- `eval/runs/smoke-20260205-153221/samples.repair.jsonl`
- `eval/runs/smoke-20260205-153221/meta.repair.jsonl`
- `eval/runs/smoke-20260205-153221/metrics.repair.json`
