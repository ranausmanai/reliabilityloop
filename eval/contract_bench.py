from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path
from statistics import mean

from autoquality.contracts import run_contract, _extract_json, _validate_schema


def _resolve_backend(backend: str, model: str):
    if backend == "mock":
        from autoquality.backends.mock import MockBackend

        return MockBackend(mode="fast")
    if backend == "llamacpp":
        from autoquality.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend(model_path=model)
    if backend == "ollama":
        from autoquality.backends.ollama import OllamaBackend

        return OllamaBackend(model=model)
    raise ValueError(f"Unknown backend: {backend}")


def _iter_prompts(path: Path, limit: int | None):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if limit is not None and i > limit:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _baseline_prompt(item: dict) -> str:
    kind = item["kind"]
    prompt = str(item["prompt"]).strip()
    if kind == "json":
        schema = item.get("schema")
        schema_block = json.dumps(schema, ensure_ascii=False, indent=2) if schema else None
        if schema_block:
            return (
                "Return ONLY valid JSON that matches this schema (no explanations, no markdown):\n"
                + schema_block
                + "\n\n"
                + prompt
            )
        return "Return ONLY valid JSON (no explanations, no markdown):\n\n" + prompt
    if kind == "sql":
        return "Write the SQL query only (no explanations, no markdown):\n\n" + prompt
    if kind == "codestub":
        return "Return ONLY code (no explanations, no markdown):\n\n" + prompt
    raise ValueError(f"Unknown kind: {kind}")


def _extract_code(text: str) -> str:
    text = text.strip()
    if "```" not in text:
        return text
    parts = text.split("```", 2)
    if len(parts) < 3:
        return text
    block = parts[1]
    if "\n" in block:
        first, rest = block.split("\n", 1)
        if len(first) < 20 and all(c.isalnum() or c in "_-+" for c in first.strip()):
            return rest.strip()
    return block.strip()


def _sql_is_valid(text: str) -> bool:
    sql = _extract_code(text).strip().strip(";")
    upper = sql.upper()
    if not upper.startswith("SELECT "):
        return False
    if " FROM " not in upper:
        return False
    # basic check for accidental JSON or prose
    if "{" in sql and "}" in sql:
        return False
    return True


def _python_is_valid(text: str) -> bool:
    code = _extract_code(text)
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def _validate_output(kind: str, text: str, schema: dict | None, language: str | None) -> bool:
    if kind == "json":
        obj, err = _extract_json(text)
        if err:
            return False
        if schema is None:
            return True
        return _validate_schema(obj, schema) is None
    if kind == "sql":
        return _sql_is_valid(text)
    if kind == "codestub":
        lang = (language or "python").lower()
        if lang != "python":
            return False
        return _python_is_valid(text)
    return False


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize(rows: list[dict], *, mode: str, kind: str | None = None) -> dict:
    filtered = [r for r in rows if r["mode"] == mode and (kind is None or r["kind"] == kind)]
    total = len(filtered)
    ok = sum(1 for r in filtered if r["ok"])
    avg_sec = mean([r["seconds"] for r in filtered]) if filtered else 0.0
    avg_chars = mean([r["chars"] for r in filtered]) if filtered else 0.0
    repairs = sum(1 for r in filtered if r.get("repaired"))
    return {
        "count": total,
        "ok_rate": ok / total if total else 0.0,
        "avg_seconds": avg_sec,
        "avg_chars": avg_chars,
        "repair_rate": repairs / total if total else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "llamacpp", "ollama"], default="ollama")
    ap.add_argument("--model", required=True)
    ap.add_argument("--slow-model", help="Optional slow model for contract repair.")
    ap.add_argument("--prompts-file", default="eval/contract_prompts.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repair-attempts", type=int, default=1)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir or f"eval/contract_runs/{ts}")
    outdir.mkdir(parents=True, exist_ok=True)

    prompts_path = Path(args.prompts_file)
    items = list(_iter_prompts(prompts_path, args.limit))

    backend = _resolve_backend(args.backend, args.model)

    rows: list[dict] = []

    for item in items:
        kind = item["kind"]
        prompt = _baseline_prompt(item)

        t0 = time.perf_counter()
        out = backend.generate(prompt=prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        dt = time.perf_counter() - t0
        text = out.text
        ok = _validate_output(kind, text, item.get("schema"), item.get("language"))
        rows.append(
            {
                "id": item.get("id"),
                "kind": kind,
                "mode": "baseline",
                "ok": ok,
                "seconds": dt,
                "chars": len(text),
            }
        )

    for item in items:
        kind = item["kind"]
        t0 = time.perf_counter()
        result = run_contract(
            backend=args.backend,
            model=args.model,
            slow_model=args.slow_model,
            kind=kind,
            prompt=item["prompt"],
            schema=item.get("schema"),
            language=item.get("language"),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            repair_attempts=args.repair_attempts,
        )
        dt = time.perf_counter() - t0
        compiled = result.compiled or ""
        ok = result.ok and _validate_output(kind, compiled, item.get("schema"), item.get("language"))
        rows.append(
            {
                "id": item.get("id"),
                "kind": kind,
                "mode": "contract",
                "ok": ok,
                "seconds": dt,
                "chars": len(compiled),
                "attempts": result.attempts,
                "repaired": result.repaired,
                "errors": result.errors,
            }
        )

    _write_jsonl(outdir / "samples.jsonl", rows)

    summary = {
        "config": {
            "backend": args.backend,
            "model": args.model,
            "slow_model": args.slow_model,
            "prompts_file": str(prompts_path),
            "limit": args.limit,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "repair_attempts": args.repair_attempts,
        },
        "overall": {
            "baseline": _summarize(rows, mode="baseline"),
            "contract": _summarize(rows, mode="contract"),
        },
        "by_kind": {},
    }

    kinds = sorted({r["kind"] for r in rows})
    for kind in kinds:
        summary["by_kind"][kind] = {
            "baseline": _summarize(rows, mode="baseline", kind=kind),
            "contract": _summarize(rows, mode="contract", kind=kind),
        }

    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Report
    lines = []
    lines.append("# autoquality Contract Benchmark")
    lines.append("")
    lines.append(f"- model: `{args.model}`")
    if args.slow_model:
        lines.append(f"- slow model: `{args.slow_model}`")
    lines.append(f"- tasks: `{len(items)}`")
    lines.append(f"- prompts: `{prompts_path}`")
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.append("| mode | valid rate | avg seconds | avg chars | repair rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for mode in ["baseline", "contract"]:
        s = summary["overall"][mode]
        lines.append(
            f"| `{mode}` | {s['ok_rate']:.2f} | {s['avg_seconds']:.2f} | {s['avg_chars']:.1f} | {s['repair_rate']:.2f} |"
        )

    lines.append("")
    lines.append("## By kind")
    lines.append("")
    lines.append("| kind | mode | valid rate | avg seconds | avg chars | repair rate |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for kind in kinds:
        for mode in ["baseline", "contract"]:
            s = summary["by_kind"][kind][mode]
            lines.append(
                f"| `{kind}` | `{mode}` | {s['ok_rate']:.2f} | {s['avg_seconds']:.2f} | {s['avg_chars']:.1f} | {s['repair_rate']:.2f} |"
            )

    lines.append("")
    lines.append("Artifacts:")
    lines.append(f"- `{outdir / 'samples.jsonl'}`")
    lines.append(f"- `{outdir / 'summary.json'}`")

    with (outdir / "report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(str(outdir))


if __name__ == "__main__":
    main()
