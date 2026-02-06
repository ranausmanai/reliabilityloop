# autoquality Contract Benchmark

- model: `llama3.2:3b`
- tasks: `50`
- prompts: `eval/contract_prompts.jsonl`

## Overall

| mode | valid rate | avg seconds | avg chars | repair rate |
|---|---:|---:|---:|---:|
| `baseline` | 0.96 | 3.44 | 183.6 | 0.00 |
| `contract` | 0.68 | 12.42 | 143.3 | 0.36 |

## By kind

| kind | mode | valid rate | avg seconds | avg chars | repair rate |
|---|---|---:|---:|---:|---:|
| `codestub` | `baseline` | 0.87 | 5.92 | 366.9 | 0.00 |
| `codestub` | `contract` | 0.73 | 14.88 | 216.7 | 0.40 |
| `json` | `baseline` | 1.00 | 2.90 | 116.8 | 0.00 |
| `json` | `contract` | 1.00 | 4.12 | 144.6 | 0.05 |
| `sql` | `baseline` | 1.00 | 1.66 | 89.5 | 0.00 |
| `sql` | `contract` | 0.20 | 21.02 | 68.1 | 0.73 |

Artifacts:
- `eval/contract_runs/llama32-50/samples.jsonl`
- `eval/contract_runs/llama32-50/summary.json`