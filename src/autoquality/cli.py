from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from autoquality.bench import run_bench
from autoquality.contracts import run_contract
from autoquality.reliability import ReliabilityConfig, run_reliability
from autoquality.repair import RepairConfig
from autoquality.router import RouteConfig, Router


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reliabilityloop")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Generate text with auto-quality routing.")
    gen.add_argument("--prompt", help="Prompt text. If omitted, read from stdin.")
    gen.add_argument("--backend", choices=["mock", "llamacpp", "ollama"], default="mock")
    gen.add_argument("--model", help="Single model identifier (GGUF path for llamacpp, model name for ollama).")
    gen.add_argument("--fast-model", help="Fast model identifier (GGUF path for llamacpp, model name for ollama).")
    gen.add_argument("--slow-model", help="Slow model identifier (GGUF path for llamacpp, model name for ollama).")
    gen.add_argument("--max-tokens", type=int, default=256)
    gen.add_argument("--temperature", type=float, default=0.2)
    gen.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    gen.add_argument("--entropy-threshold", type=float, default=0.65)
    gen.add_argument("--margin-threshold", type=float, default=0.08)
    gen.add_argument("--min-uncertain-tokens", type=int, default=4)
    gen.add_argument("--escalate-always", action="store_true")
    gen.add_argument("--no-draft", action="store_true", help="Do not pass the fast draft to slow model.")
    gen.add_argument("--strategy", choices=["escalate", "repair"], default="escalate")
    gen.add_argument("--repair-max-tokens", type=int, default=512)
    gen.add_argument("--repair-max-spans", type=int, default=4)
    gen.add_argument("--repair-window-tokens", type=int, default=8)
    gen.add_argument("--repair-max-edits", type=int, default=2)

    bench = sub.add_parser("bench", help="Run a small prompt suite and report routing stats.")
    bench.add_argument("--backend", choices=["mock", "llamacpp", "ollama"], default="mock")
    bench.add_argument("--model", help="Single model identifier (GGUF path for llamacpp, model name for ollama).")
    bench.add_argument("--fast-model", help="Fast model identifier (GGUF path for llamacpp, model name for ollama).")
    bench.add_argument("--slow-model", help="Slow model identifier (GGUF path for llamacpp, model name for ollama).")
    bench.add_argument("--prompts-file", default="eval/prompts.jsonl", help="JSONL: {\"id\":...,\"prompt\":...}")
    bench.add_argument("--max-tokens", type=int, default=256)
    bench.add_argument("--temperature", type=float, default=0.2)
    bench.add_argument("--json", action="store_true")

    bench.add_argument("--entropy-threshold", type=float, default=0.65)
    bench.add_argument("--margin-threshold", type=float, default=0.08)
    bench.add_argument("--min-uncertain-tokens", type=int, default=4)
    bench.add_argument("--escalate-always", action="store_true")
    bench.add_argument("--no-draft", action="store_true")
    bench.add_argument("--strategy", choices=["escalate", "repair"], default="escalate")
    bench.add_argument("--repair-max-tokens", type=int, default=512)
    bench.add_argument("--repair-max-spans", type=int, default=4)
    bench.add_argument("--repair-window-tokens", type=int, default=8)
    bench.add_argument("--repair-max-edits", type=int, default=2)

    contract = sub.add_parser("contract", help="Contract-first generation with validation/compilation.")
    contract.add_argument("--backend", choices=["mock", "llamacpp", "ollama"], default="mock")
    contract.add_argument("--model", required=True, help="Model identifier (GGUF path for llamacpp, model name for ollama).")
    contract.add_argument("--slow-model", help="Optional slow model for repair attempts.")
    contract.add_argument("--prompt", help="Prompt text. If omitted, read from stdin.")
    contract.add_argument("--contract", choices=["json", "sql", "codestub"], required=True)
    contract.add_argument("--schema-file", help="JSON schema file (only for --contract json).")
    contract.add_argument("--language", help="Language for code stubs (python/javascript).")
    contract.add_argument("--repair-attempts", type=int, default=1)
    contract.add_argument("--max-tokens", type=int, default=256)
    contract.add_argument("--temperature", type=float, default=0.0)
    contract.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    reliability = sub.add_parser("reliability", help="Run reliability benchmark for JSON/SQL/code tasks.")
    reliability.add_argument("--backend", choices=["mock", "llamacpp", "ollama"], default="mock")
    reliability.add_argument("--model", help="Single model identifier.")
    reliability.add_argument("--models", help="Comma-separated model identifiers.")
    reliability.add_argument("--slow-model", help="Optional slow model for contract repairs.")
    reliability.add_argument("--prompts-file", default="eval/reliability_v1_60.jsonl")
    reliability.add_argument("--tasks", default="json,sql,codestub", help="Comma-separated subset of: json,sql,codestub")
    reliability.add_argument("--limit", type=int, default=50)
    reliability.add_argument("--max-tokens", type=int, default=256)
    reliability.add_argument("--max-tokens-json", type=int, default=None)
    reliability.add_argument("--max-tokens-sql", type=int, default=None)
    reliability.add_argument("--max-tokens-code", type=int, default=None)
    reliability.add_argument("--temperature", type=float, default=0.0)
    reliability.add_argument("--repair-attempts", type=int, default=1)
    reliability.add_argument("--best-of-k", type=int, default=1, help="Try up to K candidates and stop early on first pass.")
    reliability.add_argument("--best-of-k-json", type=int, default=None)
    reliability.add_argument("--best-of-k-sql", type=int, default=None)
    reliability.add_argument("--best-of-k-code", type=int, default=None)
    reliability.add_argument("--memory-file", default=None, help="JSONL of verified wins to retrieve from.")
    reliability.add_argument("--memory-top-k", type=int, default=0, help="How many retrieved wins to prepend.")
    reliability.add_argument("--policy-json", choices=["baseline_first", "contract_first"], default="contract_first")
    reliability.add_argument("--policy-sql", choices=["baseline_first", "contract_first"], default="baseline_first")
    reliability.add_argument("--policy-code", choices=["baseline_first", "contract_first"], default="baseline_first")
    reliability.add_argument("--outdir", default=None)
    reliability.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")

    return parser


def _read_prompt(arg_prompt: str | None) -> str:
    if arg_prompt is not None:
        return arg_prompt
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("No --prompt provided and stdin is empty.")
    return data


def main() -> None:
    args = _build_parser().parse_args()

    if args.cmd == "generate":
        prompt = _read_prompt(args.prompt)
        repair_cfg = RepairConfig(
            max_spans=args.repair_max_spans,
            window_tokens=args.repair_window_tokens,
            max_edits=args.repair_max_edits,
            max_tokens=args.repair_max_tokens,
        )
        cfg = RouteConfig(
            entropy_threshold=args.entropy_threshold,
            margin_threshold=args.margin_threshold,
            min_uncertain_tokens=args.min_uncertain_tokens,
            escalate_always=args.escalate_always,
            use_draft=not args.no_draft,
            strategy=args.strategy,
            repair=repair_cfg,
        )

        router = Router(backend=args.backend, model=args.model, fast_model=args.fast_model, slow_model=args.slow_model, config=cfg)
        result = router.generate(prompt=prompt, max_tokens=args.max_tokens, temperature=args.temperature)

        if args.json:
            sys.stdout.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        else:
            sys.stdout.write(result.text.rstrip() + "\n")
            rep = ""
            if result.repair is not None:
                rep = f" repair(applied={result.repair.applied_edits},model={result.repair.model_used},err={result.repair.parse_error})"
            sys.stderr.write(f"[autoquality] used={result.used} uncertain={result.uncertainty.uncertain} reason={result.uncertainty.reason}{rep}\n")
        return

    if args.cmd == "bench":
        repair_cfg = RepairConfig(
            max_spans=args.repair_max_spans,
            window_tokens=args.repair_window_tokens,
            max_edits=args.repair_max_edits,
            max_tokens=args.repair_max_tokens,
        )
        cfg = RouteConfig(
            entropy_threshold=args.entropy_threshold,
            margin_threshold=args.margin_threshold,
            min_uncertain_tokens=args.min_uncertain_tokens,
            escalate_always=args.escalate_always,
            use_draft=not args.no_draft,
            strategy=args.strategy,
            repair=repair_cfg,
        )
        report = run_bench(
            backend=args.backend,
            model=args.model,
            fast_model=args.fast_model,
            slow_model=args.slow_model,
            prompts_file=args.prompts_file,
            config=cfg,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        if args.json:
            sys.stdout.write(json.dumps(report, ensure_ascii=False) + "\n")
        else:
            sys.stdout.write(
                "reliabilityloop bench\n"
                f"- prompts: {report['prompts']}\n"
                f"- used_fast: {report['used_fast']}\n"
                f"- used_slow: {report['used_slow']}\n"
                f"- escalation_rate: {report['escalation_rate']:.1%}\n"
                f"- avg_ms_fast: {report['avg_ms_fast']:.1f}\n"
                f"- avg_ms_slow: {report['avg_ms_slow']:.1f}\n"
                f"- repairs_attempted: {report['repairs_attempted']}\n"
                f"- repairs_with_edits: {report['repairs_with_edits']}\n"
            )
        return

    if args.cmd == "contract":
        prompt = _read_prompt(args.prompt)
        schema = None
        if args.schema_file:
            with open(args.schema_file, "r", encoding="utf-8") as f:
                schema = json.load(f)
        result = run_contract(
            backend=args.backend,
            model=args.model,
            slow_model=args.slow_model,
            kind=args.contract,
            prompt=prompt,
            schema=schema,
            language=args.language,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            repair_attempts=args.repair_attempts,
        )
        if args.json:
            sys.stdout.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        else:
            if result.compiled:
                sys.stdout.write(result.compiled.rstrip() + "\n")
            if not result.ok:
                sys.stderr.write(f"[autoquality] contract failed: {', '.join(result.errors)}\n")
            else:
                sys.stderr.write(
                    f"[autoquality] contract ok kind={result.kind} attempts={result.attempts} repaired={result.repaired} model={result.used_model}\n"
                )
        return

    if args.cmd == "reliability":
        model_list: list[str] = []
        if args.model:
            model_list.append(args.model)
        if args.models:
            model_list.extend([m.strip() for m in args.models.split(",") if m.strip()])
        if not model_list:
            raise SystemExit("Provide --model or --models for reliability benchmark.")
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        valid_tasks = {"json", "sql", "codestub"}
        if not tasks or any(t not in valid_tasks for t in tasks):
            raise SystemExit("Invalid --tasks. Allowed values: json,sql,codestub")

        task_k_map: dict[str, int] = {}
        if args.best_of_k_json is not None:
            task_k_map["json"] = max(1, args.best_of_k_json)
        if args.best_of_k_sql is not None:
            task_k_map["sql"] = max(1, args.best_of_k_sql)
        if args.best_of_k_code is not None:
            task_k_map["codestub"] = max(1, args.best_of_k_code)

        policy_mode_map = {
            "json": args.policy_json,
            "sql": args.policy_sql,
            "codestub": args.policy_code,
        }
        max_tokens_map: dict[str, int] = {}
        if args.max_tokens_json is not None:
            max_tokens_map["json"] = max(1, args.max_tokens_json)
        if args.max_tokens_sql is not None:
            max_tokens_map["sql"] = max(1, args.max_tokens_sql)
        if args.max_tokens_code is not None:
            max_tokens_map["codestub"] = max(1, args.max_tokens_code)

        cfg = ReliabilityConfig(
            backend=args.backend,
            models=model_list,
            prompts_file=args.prompts_file,
            tasks=tasks,
            limit=args.limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            repair_attempts=args.repair_attempts,
            slow_model=args.slow_model,
            best_of_k=max(1, args.best_of_k),
            task_k_map=task_k_map or None,
            memory_file=args.memory_file,
            memory_top_k=max(0, args.memory_top_k),
            policy_mode_map=policy_mode_map,
            max_tokens_map=max_tokens_map or None,
        )
        summary = run_reliability(cfg, outdir=args.outdir)
        if args.json:
            sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")
        else:
            outdir = summary["outdir"]
            leader = summary["leaderboard"][0]
            sys.stdout.write(
                "reliabilityloop reliability\n"
                f"- outdir: {outdir}\n"
                f"- models: {len(summary['leaderboard'])}\n"
                f"- tasks: {summary['task_count']}\n"
                f"- best_model: {leader['model']}\n"
                f"- best_reliability: {leader['overall']['policy_ok_rate']:.3f}\n"
                f"- leaderboard: {outdir}/leaderboard.md\n"
            )
        return


if __name__ == "__main__":
    main()
