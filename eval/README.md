# Evaluation

This folder contains a HumanEval runner that compares:
- `baseline_fast`: fast model only
- `escalate`: autoquality escalate (full slow rewrite when uncertain)
- `repair`: autoquality selective repair (JSON patch when uncertain)
- `testgate`: evidence-gated routing (run tests on the fast output; only escalate if tests fail)

## Launch in the background

```bash
PYTHONPATH=src python eval/launch_humaneval.py --limit 50
```

Evidence-gated run (recommended for coding tasks):

```bash
PYTHONPATH=src python eval/launch_humaneval.py --limit 50 --modes baseline_fast,testgate --nice 20
```

## Contract benchmark (no tests required)

Run the structured-output benchmark (JSON/SQL/code stubs):

```bash
PYTHONPATH=src python eval/contract_bench.py --model qwen2.5-coder:7b --limit 50
```

## Don’t freeze your Mac (recommended)

The launcher already runs with increased niceness by default. You can make it even gentler:

```bash
PYTHONPATH=src python eval/launch_humaneval.py --limit 50 --nice 20
```

## Recommended model pair (for coding benchmarks)

- fast: `qwen2.5-coder:7b`
- slow: `qwen2.5-coder:14b`

Override if needed:

```bash
PYTHONPATH=src python eval/launch_humaneval.py --limit 50 --fast-model qwen2.5-coder:7b --slow-model qwen2.5-coder:14b
```

Or run smaller nightly batches:

```bash
taskpolicy -b nice -n 20 PYTHONPATH=src python eval/launch_humaneval.py --limit 10
```

## Resume / evaluate-only

If you already have `samples.<mode>.jsonl`, you can evaluate without regenerating:

```bash
PYTHONPATH=src python eval/humaneval_runner.py \
  --backend ollama --fast-model YOUR_FAST --slow-model YOUR_SLOW \
  --outdir eval/runs/YOUR_RUN --limit 50 --modes baseline_fast --resume
```

It prints the output directory (under `eval/runs/`), which contains:
- `report.md`
- `run.log`
- `summary.json`
- `samples.*.jsonl` (for the HumanEval evaluator)
