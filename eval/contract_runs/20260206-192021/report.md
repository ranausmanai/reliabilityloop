# autoquality Contract Benchmark

- model: `qwen2.5-coder:1.5b`
- slow model: `qwen2.5-coder:7b`
- tasks: `50`
- prompts: `eval/contract_prompts.jsonl`

## Overall

| mode | valid rate | avg seconds | avg chars | repair rate |
|---|---:|---:|---:|---:|
| `baseline` | 0.86 | 2.17 | 211.6 | 0.00 |
| `contract` | 0.98 | 4.58 | 140.1 | 0.16 |

## By kind

| kind | mode | valid rate | avg seconds | avg chars | repair rate |
|---|---|---:|---:|---:|---:|
| `codestub` | `baseline` | 1.00 | 3.36 | 395.2 | 0.00 |
| `codestub` | `contract` | 0.93 | 9.21 | 175.5 | 0.53 |
| `json` | `baseline` | 1.00 | 2.14 | 155.3 | 0.00 |
| `json` | `contract` | 1.00 | 2.11 | 150.2 | 0.00 |
| `sql` | `baseline` | 0.53 | 1.03 | 102.9 | 0.00 |
| `sql` | `contract` | 1.00 | 3.27 | 91.1 | 0.00 |

Artifacts:
- `eval/contract_runs/20260206-192021/samples.jsonl`
- `eval/contract_runs/20260206-192021/summary.json`