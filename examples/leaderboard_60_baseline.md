# testgate reliability leaderboard

- backend: `ollama`
- tasks: `json,sql,codestub`
- task_count: `60`

| model | policy reliability | json | sql | code | policy latency (s) | contract latency (s) | avg policy candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5-coder:0.5b` | 0.867 | 1.000 | 1.000 | 0.600 | 2.428 | 4.573 | 1.133 |

Artifacts:
- `summary.json`
- `samples.jsonl`
- `wins.jsonl`
