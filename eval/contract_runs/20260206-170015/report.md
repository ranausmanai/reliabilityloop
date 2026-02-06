# autoquality Contract Benchmark

- model: `qwen2.5-coder:7b`
- tasks: `50`
- prompts: `eval/contract_prompts.jsonl`

## Overall

| mode | valid rate | avg seconds | avg chars | repair rate |
|---|---:|---:|---:|---:|
| `baseline` | 0.70 | 4.57 | 198.4 | 0.00 |
| `contract` | 0.98 | 7.93 | 144.9 | 0.06 |

## By kind

| kind | mode | valid rate | avg seconds | avg chars | repair rate |
|---|---|---:|---:|---:|---:|
| `codestub` | `baseline` | 1.00 | 6.51 | 367.4 | 0.00 |
| `codestub` | `contract` | 0.93 | 12.17 | 200.4 | 0.20 |
| `json` | `baseline` | 1.00 | 4.76 | 141.8 | 0.00 |
| `json` | `contract` | 1.00 | 5.16 | 144.8 | 0.00 |
| `sql` | `baseline` | 0.00 | 2.40 | 104.9 | 0.00 |
| `sql` | `contract` | 1.00 | 7.37 | 89.5 | 0.00 |

Artifacts:
- `eval/contract_runs/20260206-170015/samples.jsonl`
- `eval/contract_runs/20260206-170015/summary.json`