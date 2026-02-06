# autoquality Contract Benchmark

- model: `qwen2.5-coder:7b`
- tasks: `50`
- prompts: `eval/contract_prompts.jsonl`

## Overall

| mode | valid rate | avg seconds | avg chars | repair rate |
|---|---:|---:|---:|---:|
| `baseline` | 0.70 | 4.72 | 198.4 | 0.00 |
| `contract` | 1.00 | 8.03 | 153.3 | 0.06 |

## By kind

| kind | mode | valid rate | avg seconds | avg chars | repair rate |
|---|---|---:|---:|---:|---:|
| `codestub` | `baseline` | 1.00 | 6.96 | 367.4 | 0.00 |
| `codestub` | `contract` | 1.00 | 12.93 | 228.5 | 0.20 |
| `json` | `baseline` | 1.00 | 4.79 | 141.8 | 0.00 |
| `json` | `contract` | 1.00 | 5.04 | 144.8 | 0.00 |
| `sql` | `baseline` | 0.00 | 2.37 | 104.9 | 0.00 |
| `sql` | `contract` | 1.00 | 7.13 | 89.5 | 0.00 |

Artifacts:
- `eval/contract_runs/20260206-185738/samples.jsonl`
- `eval/contract_runs/20260206-185738/summary.json`