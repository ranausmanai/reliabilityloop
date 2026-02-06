# TestGate

**Why Evidence Beats Uncertainty for Local LLM Reliability**

TestGate is a framework for improving local LLM reliability through adaptive strategies. Our key finding: **test-based gating improves code generation by +18%**, while uncertainty-based routing actually hurts performance.

Paper: [TestGate: Why Evidence Beats Uncertainty for Local LLM Reliability](paper/testgate_paper.pdf)

## Key Findings

| Strategy | HumanEval Pass@1 | Change |
|----------|------------------|--------|
| Baseline (fast only) | 0.54 | — |
| Uncertainty → Escalate | 0.52 | -2% |
| Uncertainty → Repair | 0.42 | -12% |
| **Evidence-Gating** | **0.72** | **+18%** |

**Counter-intuitive result**: Models are often confident when wrong and uncertain when correct. Using logprob-based uncertainty to decide escalation makes things worse. But running tests and escalating on failure works.

## Three Strategies

1. **Evidence-Gated Routing** (recommended for code)
   - Run fast model → execute tests → escalate to slow model only if tests fail
   - +18 percentage points on HumanEval

2. **Contract-First Generation** (recommended for structured output)
   - Request JSON matching schema → validate → repair → compile to SQL/code
   - +30% for code-specialized models (Qwen)

3. **Uncertainty-Based Routing** (not recommended for code)
   - Use logprob entropy/margin to trigger escalation
   - Actually hurts code generation (-2% to -12%)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

### Evidence-Gated HumanEval (recommended)

```bash
PYTHONPATH=src python eval/humaneval_runner.py \
  --backend ollama \
  --fast-model qwen2.5-coder:7b \
  --slow-model qwen2.5-coder:14b \
  --modes baseline_fast,testgate \
  --limit 50
```

### Contract-First Generation

```bash
# SQL (model outputs JSON AST, compiled to SQL)
testgate contract \
  --backend ollama --model qwen2.5-coder:7b \
  --contract sql \
  --prompt "Get top 5 users by revenue from users table."

# JSON schema validation
testgate contract \
  --backend ollama --model qwen2.5-coder:7b \
  --contract json --schema-file schema.json \
  --prompt "Extract meeting action items."
```

### Basic Generation with Routing

```bash
testgate generate \
  --backend ollama \
  --fast-model qwen2.5-coder:7b \
  --slow-model qwen2.5-coder:14b \
  --prompt "Write a function to check if a number is prime"
```

## When to Use What

| Task | Tests Available? | Model Type | Strategy |
|------|------------------|------------|----------|
| Code | Yes | Any | **Evidence-gating** |
| Code | No | Any | Baseline only |
| SQL/JSON | No | Code-specialized | **Contract-first** |
| SQL/JSON | No | General-purpose | Baseline only |

## Hardware

All experiments run on consumer hardware (Apple M4, 24GB RAM) using local models via Ollama.

## Citation

```bibtex
@article{usman2026testgate,
  title={TestGate: Why Evidence Beats Uncertainty for Local LLM Reliability},
  author={Usman, Rana Muhammad},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT
