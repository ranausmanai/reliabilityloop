from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
import gzip

from human_eval import data, evaluation
from human_eval.execution import check_correctness

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autoquality.repair import RepairConfig
from autoquality.router import RouteConfig, Router


def _make_model_prompt(code_prompt: str) -> str:
    return (
        "You are a senior Python developer.\n"
        "Complete the following function.\n"
        "Return ONLY the code that should be appended after the prompt (no explanations, no markdown fences).\n\n"
        + code_prompt
    )


def _extract_code(text: str) -> str:
    text = text.strip()
    if "```" not in text:
        return text
    # Take first fenced block.
    parts = text.split("```", 2)
    if len(parts) < 3:
        return text
    block = parts[1]
    # Drop optional language tag.
    if "\n" in block:
        first, rest = block.split("\n", 1)
        if len(first) < 20 and all(c.isalnum() or c in "_-+" for c in first.strip()):
            return rest.strip()
    return block.strip()

def _normalize_indentation_for_function_body(code: str) -> str:
    lines = code.splitlines()
    out: list[str] = []
    for line in lines:
        if not line.strip():
            out.append("")
            continue
        if line.startswith("\t"):
            out.append(line)
            continue
        lead_spaces = len(line) - len(line.lstrip(" "))
        if lead_spaces < 4:
            out.append((" " * (4 - lead_spaces)) + line)
        else:
            out.append(line)
    return "\n".join(out).rstrip()


def _completion_only(text: str, entry_point: str) -> str:
    code = _extract_code(text)
    # If model returned the full function, keep only the body part.
    marker = f"def {entry_point}"
    if marker in code:
        after = code.split(marker, 1)[1]
        # Find the first newline after the signature line.
        nl = after.find("\n")
        if nl >= 0:
            body = after[nl + 1 :].rstrip()
            return _normalize_indentation_for_function_body(body)
    return _normalize_indentation_for_function_body(code.rstrip())


def _make_fix_prompt(base_prompt: str, failure: str) -> str:
    return (
        base_prompt.rstrip()
        + "\n\nThe previous attempt failed tests with error:\n"
        + failure.strip()
        + "\n\nReturn ONLY the corrected code that should be appended after the prompt."
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_problem_subset_gz(path: Path, problems: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for _, prob in problems.items():
            f.write(json.dumps(prob, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama", choices=["ollama"])
    ap.add_argument("--fast-model", required=True)
    ap.add_argument("--slow-model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument(
        "--modes",
        default="baseline_fast,escalate,repair",
        help="Comma-separated: baseline_fast,escalate,repair,testgate",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If samples.*.jsonl exists for a mode, skip generation and only evaluate/report.",
    )
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--normalize-completions",
        action="store_true",
        help="Write normalized sample files (indentation fixes) and evaluate those.",
    )

    # Router sensitivity knobs
    ap.add_argument("--entropy-threshold", type=float, default=0.65)
    ap.add_argument("--margin-threshold", type=float, default=0.08)
    ap.add_argument("--min-uncertain-tokens", type=int, default=10)

    # Repair knobs
    ap.add_argument("--repair-max-tokens", type=int, default=256)
    ap.add_argument("--repair-max-spans", type=int, default=3)
    ap.add_argument("--repair-window-tokens", type=int, default=8)
    ap.add_argument("--repair-max-edits", type=int, default=2)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    problems = data.read_problems()
    task_ids = sorted(problems.keys())[: args.limit]
    problems_subset = {tid: problems[tid] for tid in task_ids}
    subset_problem_file = outdir / "HumanEval.subset.jsonl.gz"
    _write_problem_subset_gz(subset_problem_file, problems_subset)

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    allowed = {"baseline_fast", "escalate", "repair", "testgate"}
    bad = [m for m in modes if m not in allowed]
    if bad:
        raise SystemExit(f"Unknown mode(s): {bad}. Allowed: {sorted(allowed)}")

    results_summary: dict[str, dict] = {}

    base_cfg = RouteConfig(
        entropy_threshold=args.entropy_threshold,
        margin_threshold=args.margin_threshold,
        min_uncertain_tokens=args.min_uncertain_tokens,
        strategy="escalate",
    )
    repair_cfg = RepairConfig(
        max_tokens=args.repair_max_tokens,
        max_spans=args.repair_max_spans,
        window_tokens=args.repair_window_tokens,
        max_edits=args.repair_max_edits,
    )

    for mode in modes:
        samples: list[dict] = []
        per_task: list[dict] = []

        t_mode0 = time.perf_counter()
        gen_seconds = 0.0
        slow_used = 0
        repair_attempted = 0
        repair_applied = 0
        gate_attempted = 0
        gate_fast_passed = 0
        gate_slow_passed = 0

        if mode == "baseline_fast":
            router = Router(backend=args.backend, model=args.fast_model, fast_model=None, slow_model=None, config=base_cfg)
        elif mode == "escalate":
            cfg = base_cfg
            router = Router(backend=args.backend, model=None, fast_model=args.fast_model, slow_model=args.slow_model, config=cfg)
        elif mode == "testgate":
            fast_router = Router(backend=args.backend, model=args.fast_model, fast_model=None, slow_model=None, config=base_cfg)
            slow_router = Router(backend=args.backend, model=args.slow_model, fast_model=None, slow_model=None, config=base_cfg)
        else:
            cfg = RouteConfig(
                entropy_threshold=args.entropy_threshold,
                margin_threshold=args.margin_threshold,
                min_uncertain_tokens=args.min_uncertain_tokens,
                strategy="repair",
                repair=repair_cfg,
            )
            router = Router(backend=args.backend, model=None, fast_model=args.fast_model, slow_model=args.slow_model, config=cfg)

        samples_path = outdir / f"samples.{mode}.jsonl"
        per_task_path = outdir / f"meta.{mode}.jsonl"

        if args.resume and samples_path.exists():
            # Best-effort: recover generation timing from meta if present.
            if per_task_path.exists():
                try:
                    gen_seconds = 0.0
                    for line in per_task_path.read_text(encoding="utf-8").splitlines():
                        obj = json.loads(line)
                        gen_seconds += float(obj.get("sec", 0.0))
                        if obj.get("used") == "slow":
                            slow_used += 1
                        rep = obj.get("repair")
                        if isinstance(rep, dict) and rep.get("attempted") is True:
                            repair_attempted += 1
                            if int(rep.get("applied_edits") or 0) > 0:
                                repair_applied += 1
                        gate = obj.get("gate")
                        if isinstance(gate, dict):
                            gate_attempted += 1
                            fast = gate.get("fast")
                            if isinstance(fast, dict) and fast.get("passed") is True:
                                gate_fast_passed += 1
                            slow = gate.get("slow")
                            if isinstance(slow, dict) and slow.get("passed") is True:
                                gate_slow_passed += 1
                except Exception:
                    gen_seconds = 0.0
        else:
            for i, task_id in enumerate(task_ids, start=1):
                prob = problems_subset[task_id]
                model_prompt = _make_model_prompt(prob["prompt"])

                t0 = time.perf_counter()
                if mode == "testgate":
                    fast_out = fast_router.generate(prompt=model_prompt, max_tokens=args.max_tokens, temperature=args.temperature)
                    completion_fast = _completion_only(fast_out.text, prob["entry_point"])
                    gate_fast = check_correctness(prob, completion_fast, args.timeout)
                    gate_attempted += 1
                    if gate_fast["passed"]:
                        gate_fast_passed += 1

                    used = "fast"
                    completion = completion_fast
                    gate_info: dict[str, object] = {"fast": gate_fast, "slow": None}

                    if not gate_fast["passed"]:
                        slow_prompt = _make_fix_prompt(model_prompt, gate_fast["result"])
                        slow_out = slow_router.generate(
                            prompt=slow_prompt,
                            max_tokens=args.max_tokens,
                            temperature=min(0.2, args.temperature),
                        )
                        completion_slow = _completion_only(slow_out.text, prob["entry_point"])
                        gate_slow = check_correctness(prob, completion_slow, args.timeout)
                        if gate_slow["passed"]:
                            gate_slow_passed += 1
                        used = "slow"
                        completion = completion_slow
                        gate_info["slow"] = gate_slow
                        slow_used += 1

                    dt = time.perf_counter() - t0
                    gen_seconds += dt

                    samples.append({"task_id": task_id, "completion": completion})
                    per_task.append(
                        {
                            "task_id": task_id,
                            "used": used,
                            "sec": dt,
                            "uncertainty": asdict(fast_out.uncertainty),
                            "repair": None,
                            "gate": gate_info,
                            "completion_chars": len(completion),
                        }
                    )
                else:
                    out = router.generate(prompt=model_prompt, max_tokens=args.max_tokens, temperature=args.temperature)
                    dt = time.perf_counter() - t0
                    gen_seconds += dt

                    completion = _completion_only(out.text, prob["entry_point"])

                    samples.append({"task_id": task_id, "completion": completion})
                    per_task.append(
                        {
                            "task_id": task_id,
                            "used": out.used,
                            "sec": dt,
                            "uncertainty": asdict(out.uncertainty),
                            "repair": asdict(out.repair) if out.repair is not None else None,
                            "gate": None,
                            "completion_chars": len(completion),
                        }
                    )
                    if out.used == "slow":
                        slow_used += 1
                    if out.repair is not None and out.repair.attempted:
                        repair_attempted += 1
                        if out.repair.applied_edits > 0:
                            repair_applied += 1

                if i % 5 == 0 or i == len(task_ids):
                    print(f"[{mode}] generated {i}/{len(task_ids)} tasks", flush=True)

            _write_jsonl(samples_path, samples)
            _write_jsonl(per_task_path, per_task)

        eval_sample_path = samples_path
        if args.normalize_completions:
            normalized_path = outdir / f"samples.{mode}.normalized.jsonl"
            rows = []
            for line in Path(samples_path).read_text(encoding="utf-8").splitlines():
                obj = json.loads(line)
                # Apply indentation normalization again for safety.
                obj["completion"] = _normalize_indentation_for_function_body(str(obj.get("completion", "")))
                rows.append(obj)
            _write_jsonl(normalized_path, rows)
            eval_sample_path = normalized_path

        t_eval0 = time.perf_counter()
        metrics = evaluation.evaluate_functional_correctness(
            sample_file=str(eval_sample_path),
            k=[args.k],
            n_workers=args.n_workers,
            timeout=args.timeout,
            problem_file=str(subset_problem_file),
        )
        eval_seconds = time.perf_counter() - t_eval0

        t_mode1 = time.perf_counter()

        results_summary[mode] = {
            "metrics": metrics,
            "seconds_generate": gen_seconds,
            "seconds_eval": eval_seconds,
            "seconds_total": gen_seconds + eval_seconds,
            "tasks": len(task_ids),
            "samples_path": str(samples_path),
            "eval_samples_path": str(eval_sample_path),
            "slow_used": slow_used,
            "repair_attempted": repair_attempted,
            "repair_applied": repair_applied,
            "gate_attempted": gate_attempted,
            "gate_fast_passed": gate_fast_passed,
            "gate_slow_passed": gate_slow_passed,
        }

        with open(outdir / f"metrics.{mode}.json", "w", encoding="utf-8") as f:
            json.dump(results_summary[mode], f, indent=2)

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "fast_model": args.fast_model,
                "slow_model": args.slow_model,
                "limit": args.limit,
                "timestamp": int(time.time()),
                "results": results_summary,
                "config": {
                    "entropy_threshold": args.entropy_threshold,
                    "margin_threshold": args.margin_threshold,
                    "min_uncertain_tokens": args.min_uncertain_tokens,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "repair": asdict(repair_cfg),
                },
            },
            f,
            indent=2,
        )

    # Write a tiny markdown report.
    lines = []
    lines.append("# autoquality HumanEval report")
    lines.append("")
    lines.append(f"- fast model: `{args.fast_model}`")
    lines.append(f"- slow model: `{args.slow_model}`")
    lines.append(f"- tasks: `{args.limit}`")
    lines.append("")
    lines.append(f"- completion normalization: `{bool(args.normalize_completions)}`")
    lines.append("")
    lines.append("| mode | pass@1 | gen seconds | eval seconds | total seconds | eval samples |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for mode in modes:
        m = results_summary[mode]["metrics"]
        pass1 = m.get("pass@1")
        gen_s = results_summary[mode]["seconds_generate"]
        eval_s = results_summary[mode]["seconds_eval"]
        sec = results_summary[mode]["seconds_total"]
        eval_samples = results_summary[mode]["eval_samples_path"]
        lines.append(f"| `{mode}` | {pass1:.4f} | {gen_s:.1f} | {eval_s:.1f} | {sec:.1f} | `{eval_samples}` |")
    lines.append("")
    lines.append("| mode | slow used | repairs attempted | repairs applied |")
    lines.append("|---|---:|---:|---:|")
    for mode in modes:
        lines.append(
            f"| `{mode}` | {results_summary[mode]['slow_used']} | {results_summary[mode]['repair_attempted']} | {results_summary[mode]['repair_applied']} |"
        )
    lines.append("")
    lines.append("| mode | gate fast pass | gate slow pass |")
    lines.append("|---|---:|---:|")
    for mode in modes:
        lines.append(
            f"| `{mode}` | {results_summary[mode]['gate_fast_passed']} | {results_summary[mode]['gate_slow_passed']} |"
        )
    lines.append("")
    lines.append("Artifacts:")
    for mode in modes:
        lines.append(f"- `{results_summary[mode]['samples_path']}`")
        lines.append(f"- `{outdir / f'meta.{mode}.jsonl'}`")
        lines.append(f"- `{outdir / f'metrics.{mode}.json'}`")
    lines.append("")

    with open(outdir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
