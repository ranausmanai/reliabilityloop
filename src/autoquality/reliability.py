from __future__ import annotations

import ast
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from autoquality.contracts import _extract_json, _validate_schema, run_contract


@dataclass(frozen=True)
class ReliabilityConfig:
    backend: str
    models: list[str]
    prompts_file: str
    tasks: list[str]
    limit: int
    max_tokens: int
    temperature: float
    repair_attempts: int
    slow_model: str | None = None
    best_of_k: int = 1
    task_k_map: dict[str, int] | None = None
    memory_file: str | None = None
    memory_top_k: int = 0
    policy_mode_map: dict[str, str] | None = None
    max_tokens_map: dict[str, int] | None = None


def run_reliability(config: ReliabilityConfig, outdir: str | None = None) -> dict[str, Any]:
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = Path(outdir or f"eval/reliability_runs/{ts}")
    out_path.mkdir(parents=True, exist_ok=True)

    items = _load_items(Path(config.prompts_file), config.tasks, config.limit)
    if not items:
        raise ValueError("No benchmark tasks selected. Check --tasks/--prompts-file/--limit.")

    memory_entries = _load_memory_entries(config.memory_file)

    results: list[dict[str, Any]] = []
    per_model_rows: list[dict[str, Any]] = []

    for model in config.models:
        backend = _resolve_backend(config.backend, model)
        rows: list[dict[str, Any]] = []
        baseline_by_id: dict[str, dict[str, Any]] = {}
        contract_by_id: dict[str, dict[str, Any]] = {}

        for item in items:
            kind = str(item["kind"])
            original_prompt = str(item["prompt"])
            augmented_prompt = _augment_prompt_with_memory(
                prompt=original_prompt,
                kind=kind,
                memory_entries=memory_entries,
                top_k=config.memory_top_k,
            )
            prompt_item = dict(item)
            prompt_item["prompt"] = augmented_prompt

            baseline_prompt = _baseline_prompt(prompt_item)
            cached = _lookup_memory_exact(prompt=original_prompt, kind=kind, memory_entries=memory_entries)
            if cached is not None:
                baseline_output = cached
                baseline_ok = _validate_output(item, cached)
                baseline_seconds = 0.0
                baseline_chars = len(cached)
                baseline_candidates = 0
            else:
                baseline_ok, baseline_seconds, baseline_chars, baseline_candidates, baseline_output = _run_best_of_k_generation(
                    backend=backend,
                    item=item,
                    prompt=baseline_prompt,
                    max_tokens=_max_tokens_for_kind(config, kind),
                    temperature=config.temperature,
                    best_of_k=_best_of_k_for_kind(config, kind),
                )
            row = {
                "model": model,
                "id": item["id"],
                "kind": kind,
                "mode": "baseline",
                "ok": baseline_ok,
                "seconds": baseline_seconds,
                "chars": baseline_chars,
                "attempts": 1,
                "repaired": False,
                "candidates_used": baseline_candidates,
                "prompt": original_prompt,
                "prompt_used": augmented_prompt,
                "output": baseline_output,
            }
            rows.append(row)
            baseline_by_id[str(item["id"])] = row

        for item in items:
            kind = str(item["kind"])
            original_prompt = str(item["prompt"])
            augmented_prompt = _augment_prompt_with_memory(
                prompt=original_prompt,
                kind=kind,
                memory_entries=memory_entries,
                top_k=config.memory_top_k,
            )

            (
                contract_ok,
                contract_seconds,
                contract_chars,
                contract_attempts,
                contract_repaired,
                contract_candidates,
                contract_output,
            ) = _run_best_of_k_contract(
                config=config,
                item=item,
                model=model,
                prompt=augmented_prompt,
            )
            row = {
                "model": model,
                "id": item["id"],
                "kind": kind,
                "mode": "contract",
                "ok": contract_ok,
                "seconds": contract_seconds,
                "chars": contract_chars,
                "attempts": contract_attempts,
                "repaired": contract_repaired,
                "candidates_used": contract_candidates,
                "prompt": original_prompt,
                "prompt_used": augmented_prompt,
                "output": contract_output,
            }
            rows.append(row)
            contract_by_id[str(item["id"])] = row

        for item in items:
            item_id = str(item["id"])
            kind = str(item["kind"])
            b = baseline_by_id[item_id]
            c = contract_by_id[item_id]
            mode = _policy_mode_for_kind(config, kind)
            code_repair_attempted = False

            if mode == "contract_first":
                use_contract = bool(c["ok"])
                chosen = c if use_contract else b
                decision = "contract" if use_contract else "baseline_fallback"
                seconds = float(c["seconds"]) + (float(b["seconds"]) if not c["ok"] else 0.0)
                candidates_used = int(c.get("candidates_used", 1)) + (int(b.get("candidates_used", 1)) if not c["ok"] else 0)
                attempts = int(c.get("attempts", 1)) if use_contract else 1
                repaired = bool(c.get("repaired", False)) if use_contract else False
            else:
                use_contract = not bool(b["ok"])
                chosen = c if use_contract else b
                decision = "contract" if use_contract else "baseline"
                seconds = float(b["seconds"]) + (float(c["seconds"]) if use_contract else 0.0)
                candidates_used = int(b.get("candidates_used", 1)) + (int(c.get("candidates_used", 1)) if use_contract else 0)
                attempts = int(c.get("attempts", 1)) if use_contract else 1
                repaired = bool(c.get("repaired", False)) if use_contract else False

            # Targeted test-feedback repair for code tasks when baseline+contract both fail.
            if kind == "codestub" and (not bool(b["ok"])) and (not bool(c["ok"])):
                code_repair_attempted = True
                repair_ok, repair_seconds, repair_chars, repair_output = _run_code_test_feedback_repair(
                    backend=backend,
                    item=item,
                    original_prompt=str(item.get("prompt") or ""),
                    previous_output=str(b.get("output") or ""),
                    max_tokens=_max_tokens_for_kind(config, kind),
                )
                seconds += repair_seconds
                candidates_used += 1
                if repair_ok:
                    chosen = {
                        "chars": repair_chars,
                        "prompt_used": str(item.get("prompt") or ""),
                        "output": repair_output,
                    }
                    decision = "code_repair"
                    repaired = True

            policy_row = {
                "model": model,
                "id": item_id,
                "kind": kind,
                "mode": "policy",
                "ok": bool(b["ok"] or c["ok"] or (decision == "code_repair")),
                "seconds": seconds,
                "chars": int(chosen.get("chars", 0)),
                "attempts": attempts,
                "repaired": repaired,
                "candidates_used": candidates_used,
                "prompt": str(item.get("prompt") or ""),
                "prompt_used": str(chosen.get("prompt_used") or item.get("prompt") or ""),
                "output": str(chosen.get("output") or ""),
                "decision": decision,
                "policy_mode": mode,
                "code_repair_attempted": code_repair_attempted,
            }
            rows.append(policy_row)

        per_model_rows.extend(rows)
        results.append(_summarize_model(model=model, rows=rows))

    leaderboard = sorted(results, key=lambda x: x["overall"]["policy_ok_rate"], reverse=True)
    summary: dict[str, Any] = {
        "config": {
            "backend": config.backend,
            "models": config.models,
            "slow_model": config.slow_model,
            "prompts_file": config.prompts_file,
            "tasks": config.tasks,
            "limit": config.limit,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "repair_attempts": config.repair_attempts,
            "best_of_k": config.best_of_k,
            "task_k_map": config.task_k_map or {},
            "memory_file": config.memory_file,
            "memory_top_k": config.memory_top_k,
            "memory_entries_loaded": len(memory_entries),
            "policy_mode_map": config.policy_mode_map or {},
            "max_tokens_map": config.max_tokens_map or {},
        },
        "task_count": len(items),
        "leaderboard": leaderboard,
        "generated_at": ts,
        "outdir": str(out_path),
    }

    _write_jsonl(out_path / "samples.jsonl", per_model_rows)
    _write_wins_jsonl(out_path / "wins.jsonl", per_model_rows)
    with (out_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_path / "leaderboard.md").open("w", encoding="utf-8") as f:
        f.write(_leaderboard_markdown(summary))

    return summary


def _load_items(path: Path, tasks: list[str], limit: int) -> list[dict[str, Any]]:
    allow = [t for t in tasks]
    allow_set = set(allow)
    pools: dict[str, list[dict[str, Any]]] = {t: [] for t in allow}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kind = str(obj.get("kind", ""))
            if kind not in allow_set:
                continue
            pools[kind].append(
                {
                    "id": str(obj.get("id", "")),
                    "kind": kind,
                    "prompt": str(obj.get("prompt", "")),
                    "schema": obj.get("schema"),
                    "language": obj.get("language"),
                    "expected": obj.get("expected"),
                    "db": obj.get("db"),
                    "entry_point": obj.get("entry_point"),
                    "tests": obj.get("tests"),
                }
            )
    if limit <= 0:
        return []

    selected: list[dict[str, Any]] = []
    kinds = [k for k in allow if pools.get(k)]
    if not kinds:
        return []

    per_kind = max(1, limit // len(kinds))
    for kind in kinds:
        selected.extend(pools[kind][:per_kind])

    if len(selected) < limit:
        idx = {k: per_kind for k in kinds}
        while len(selected) < limit:
            made_progress = False
            for kind in kinds:
                p = pools[kind]
                i = idx[kind]
                if i < len(p):
                    selected.append(p[i])
                    idx[kind] += 1
                    made_progress = True
                    if len(selected) >= limit:
                        break
            if not made_progress:
                break

    return selected[:limit]


def _load_memory_entries(memory_file: str | None) -> list[dict[str, str]]:
    if not memory_file:
        return []
    path = Path(memory_file)
    if not path.exists():
        return []
    entries: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            kind = str(obj.get("kind") or "")
            prompt = str(obj.get("prompt") or "")
            output = str(obj.get("output") or "")
            if kind and prompt and output:
                entries.append({"kind": kind, "prompt": prompt, "output": output})
    return entries


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


def _best_of_k_for_kind(config: ReliabilityConfig, kind: str) -> int:
    if config.task_k_map and kind in config.task_k_map:
        return max(1, int(config.task_k_map[kind]))
    return max(1, int(config.best_of_k))


def _policy_mode_for_kind(config: ReliabilityConfig, kind: str) -> str:
    if config.policy_mode_map and kind in config.policy_mode_map:
        mode = str(config.policy_mode_map[kind])
        if mode in {"baseline_first", "contract_first"}:
            return mode
    return "baseline_first"


def _max_tokens_for_kind(config: ReliabilityConfig, kind: str) -> int:
    if config.max_tokens_map and kind in config.max_tokens_map:
        return max(1, int(config.max_tokens_map[kind]))
    return max(1, int(config.max_tokens))


def _run_best_of_k_generation(
    *,
    backend: Any,
    item: dict[str, Any],
    prompt: str,
    max_tokens: int,
    temperature: float,
    best_of_k: int,
) -> tuple[bool, float, int, int, str]:
    total_seconds = 0.0
    best_chars = 0
    candidates_used = 0
    last_text = ""
    k = max(1, best_of_k)
    for i in range(k):
        t0 = time.perf_counter()
        out = backend.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        total_seconds += (time.perf_counter() - t0)
        candidates_used = i + 1
        best_chars = len(out.text)
        last_text = out.text
        if _validate_output(item, out.text):
            return True, total_seconds, best_chars, candidates_used, out.text
    return False, total_seconds, best_chars, candidates_used, last_text


def _run_code_test_feedback_repair(
    *,
    backend: Any,
    item: dict[str, Any],
    original_prompt: str,
    previous_output: str,
    max_tokens: int,
) -> tuple[bool, float, int, str]:
    tests = item.get("tests") or []
    prompt = (
        "You must return ONLY Python code. No markdown.\n"
        "Fix the function so it passes these tests exactly.\n\n"
        f"Original task:\n{original_prompt}\n\n"
        f"Previous attempt:\n{previous_output}\n\n"
        f"Tests JSON:\n{json.dumps(tests, ensure_ascii=False)}\n"
    )
    t0 = time.perf_counter()
    out = backend.generate(prompt=prompt, max_tokens=max_tokens, temperature=0.0)
    dt = time.perf_counter() - t0
    ok = _validate_output(item, out.text)
    return ok, dt, len(out.text), out.text


def _run_best_of_k_contract(
    *,
    config: ReliabilityConfig,
    item: dict[str, Any],
    model: str,
    prompt: str,
) -> tuple[bool, float, int, int, bool, int, str]:
    total_seconds = 0.0
    best_chars = 0
    candidates_used = 0
    attempts = 1
    repaired = False
    kind = str(item["kind"])
    last_compiled = ""
    k = _best_of_k_for_kind(config, kind)
    for i in range(k):
        t0 = time.perf_counter()
        contract = run_contract(
            backend=config.backend,
            model=model,
            slow_model=config.slow_model,
            kind=kind,  # type: ignore[arg-type]
            prompt=prompt,
            schema=item.get("schema"),
            language=item.get("language"),
            max_tokens=_max_tokens_for_kind(config, kind),
            temperature=config.temperature,
            repair_attempts=config.repair_attempts,
        )
        total_seconds += (time.perf_counter() - t0)
        candidates_used = i + 1
        attempts = contract.attempts
        repaired = contract.repaired
        compiled = contract.compiled or ""
        best_chars = len(compiled)
        last_compiled = compiled
        if contract.ok and _validate_output(item, compiled):
            return True, total_seconds, best_chars, attempts, repaired, candidates_used, compiled
    return False, total_seconds, best_chars, attempts, repaired, candidates_used, last_compiled


def _lookup_memory_exact(*, prompt: str, kind: str, memory_entries: list[dict[str, str]]) -> str | None:
    for e in memory_entries:
        if e.get("kind") == kind and e.get("prompt") == prompt:
            out = str(e.get("output") or "").strip()
            if out:
                return out
    return None


def _augment_prompt_with_memory(*, prompt: str, kind: str, memory_entries: list[dict[str, str]], top_k: int) -> str:
    if top_k <= 0 or not memory_entries:
        return prompt

    candidates = [e for e in memory_entries if e.get("kind") == kind]
    if not candidates:
        return prompt

    target = _token_set(prompt)
    scored: list[tuple[int, dict[str, str]]] = []
    for e in candidates:
        overlap = len(target & _token_set(e["prompt"]))
        if overlap > 0:
            scored.append((overlap, e))
    if not scored:
        return prompt

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [x[1] for x in scored[: max(1, top_k)]]

    parts: list[str] = []
    parts.append("Use the verified examples below only as guidance for structure and correctness.")
    parts.append("Do not copy blindly; solve the new task.")
    for i, e in enumerate(picked, start=1):
        parts.append(f"Example {i} prompt:\n{e['prompt']}")
        parts.append(f"Example {i} answer:\n{_truncate(e['output'], 700)}")
    parts.append(f"New task:\n{prompt}")
    return "\n\n".join(parts)


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def _truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[:n] + "..."


def _baseline_prompt(item: dict[str, Any]) -> str:
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


def _validate_output(item: dict[str, Any], text: str) -> bool:
    kind = str(item["kind"])
    if kind == "json":
        schema = item.get("schema")
        obj, err = _extract_json(text)
        if err:
            return False
        if schema is not None and _validate_schema(obj, schema) is not None:
            return False
        expected = item.get("expected")
        if isinstance(expected, dict):
            return _json_matches_expected(obj, expected)
        return True
    if kind == "sql":
        return _sql_matches_expected(text, item)
    if kind == "codestub":
        language = str(item.get("language") or "python").lower()
        if language != "python":
            return False
        return _python_tests_pass(text, item)
    return False


def _json_matches_expected(obj: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    for key, value in expected.items():
        if key not in obj:
            return False
        if not _values_equal(obj[key], value):
            return False
    return True


def _sql_matches_expected(text: str, item: dict[str, Any]) -> bool:
    expected = item.get("expected")
    db = item.get("db")
    if not isinstance(expected, dict) or not isinstance(db, dict):
        sql = _extract_code(text).strip().strip(";")
        upper = sql.upper()
        return upper.startswith("SELECT ") and (" FROM " in upper) and ("{" not in sql and "}" not in sql)

    sql = _extract_code(text).strip().strip(";")
    upper = sql.upper()
    if not upper.startswith("SELECT "):
        return False
    if ";" in sql:
        return False

    try:
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        for table in db.get("tables", []):
            name = table["name"]
            cols = table["columns"]
            col_defs = ", ".join(f"{c[0]} {c[1]}" for c in cols)
            cur.execute(f"CREATE TABLE {name} ({col_defs})")
            rows = table.get("rows", [])
            if rows:
                placeholders = ", ".join("?" for _ in cols)
                cur.executemany(f"INSERT INTO {name} VALUES ({placeholders})", rows)

        cur.execute(sql)
        actual_rows = cur.fetchall()
        desc = [d[0] for d in (cur.description or [])]
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

    expected_cols = expected.get("columns")
    expected_rows = expected.get("rows")
    order_sensitive = bool(expected.get("order_sensitive", False))

    if isinstance(expected_cols, list):
        if [str(c).lower() for c in desc] != [str(c).lower() for c in expected_cols]:
            return False
    if not isinstance(expected_rows, list):
        return True

    if len(actual_rows) != len(expected_rows):
        return False

    actual = [_normalize_row(r) for r in actual_rows]
    target = [_normalize_row(tuple(r)) for r in expected_rows]
    if order_sensitive:
        return all(_values_equal(a, b) for a, b in zip(actual, target))

    return sorted(actual) == sorted(target)


def _normalize_row(row: tuple[Any, ...]) -> tuple[Any, ...]:
    out: list[Any] = []
    for v in row:
        if isinstance(v, float):
            out.append(round(v, 6))
        else:
            out.append(v)
    return tuple(out)


def _python_tests_pass(text: str, item: dict[str, Any]) -> bool:
    code = _extract_code(text)
    try:
        ast.parse(code)
    except Exception:
        return False

    entry_point = item.get("entry_point")
    tests = item.get("tests")
    if not isinstance(entry_point, str) or not isinstance(tests, list):
        return True

    safe_modules = {"math", "re", "itertools", "functools", "collections"}

    def _safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in safe_modules:
            raise ImportError(f"module not allowed: {name}")
        return __import__(name, globals, locals, fromlist, level)

    safe_builtins = {
        "__import__": _safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    env: dict[str, Any] = {"__builtins__": safe_builtins}
    try:
        exec(compile(code, "<model_code>", "exec"), env, env)
    except Exception:
        return False

    fn = env.get(entry_point)
    if not callable(fn):
        return False
    for test in tests:
        if not isinstance(test, dict):
            return False
        args = test.get("args", [])
        kwargs = test.get("kwargs", {})
        expected = test.get("equals")
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            return False
        try:
            got = fn(*args, **kwargs)
        except Exception:
            return False
        if not _values_equal(got, expected):
            return False
    return True


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) <= 1e-6
        except Exception:
            return False
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_values_equal(left[k], right[k]) for k in left.keys())
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_values_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            return False
        return all(_values_equal(a, b) for a, b in zip(left, right))
    return left == right


def _summarize_model(*, model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = [r for r in rows if r["mode"] == "baseline"]
    contract = [r for r in rows if r["mode"] == "contract"]
    policy = [r for r in rows if r["mode"] == "policy"]
    kinds = sorted({r["kind"] for r in rows})
    kind_summary: dict[str, Any] = {}
    for kind in kinds:
        b = [r for r in base if r["kind"] == kind]
        c = [r for r in contract if r["kind"] == kind]
        p = [r for r in policy if r["kind"] == kind]
        kind_summary[kind] = {
            "baseline_ok_rate": _ok_rate(b),
            "contract_ok_rate": _ok_rate(c),
            "policy_ok_rate": _ok_rate(p),
            "avg_latency_s": _avg_seconds(c),
            "policy_avg_latency_s": _avg_seconds(p),
            "retry_rate": _retry_rate(c),
            "avg_candidates": _avg_candidates(c),
            "policy_avg_candidates": _avg_candidates(p),
        }

    retried = [r for r in contract if int(r.get("attempts", 1)) > 1]
    retried_success = [r for r in retried if r["ok"]]
    return {
        "model": model,
        "overall": {
            "baseline_ok_rate": _ok_rate(base),
            "contract_ok_rate": _ok_rate(contract),
            "policy_ok_rate": _ok_rate(policy),
            "avg_latency_s": _avg_seconds(contract),
            "policy_avg_latency_s": _avg_seconds(policy),
            "avg_attempts": mean([int(r.get("attempts", 1)) for r in contract]) if contract else 0.0,
            "retry_rate": _retry_rate(contract),
            "success_after_retry_rate": (len(retried_success) / len(retried)) if retried else 0.0,
            "avg_candidates_baseline": _avg_candidates(base),
            "avg_candidates_contract": _avg_candidates(contract),
            "avg_candidates_policy": _avg_candidates(policy),
        },
        "by_task": kind_summary,
    }


def _ok_rate(rows: list[dict[str, Any]]) -> float:
    return (sum(1 for r in rows if r["ok"]) / len(rows)) if rows else 0.0


def _avg_seconds(rows: list[dict[str, Any]]) -> float:
    return mean([float(r["seconds"]) for r in rows]) if rows else 0.0


def _retry_rate(rows: list[dict[str, Any]]) -> float:
    return (sum(1 for r in rows if int(r.get("attempts", 1)) > 1) / len(rows)) if rows else 0.0


def _avg_candidates(rows: list[dict[str, Any]]) -> float:
    return mean([int(r.get("candidates_used", 1)) for r in rows]) if rows else 0.0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_wins_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if row.get("mode") != "policy":
                continue
            if not row.get("ok"):
                continue
            out = str(row.get("output") or "").strip()
            prompt = str(row.get("prompt") or "").strip()
            if not out or not prompt:
                continue
            rec = {
                "kind": row.get("kind"),
                "prompt": prompt,
                "output": out,
                "id": row.get("id"),
                "model": row.get("model"),
                "mode": row.get("mode"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _leaderboard_markdown(summary: dict[str, Any]) -> str:
    def _fmt_task(row: dict[str, Any], key: str) -> str:
        task = row["by_task"].get(key)
        if not task:
            return "n/a"
        return f"{task.get('policy_ok_rate', 0.0):.3f}"

    lines: list[str] = []
    lines.append("# reliabilityloop reliability leaderboard")
    lines.append("")
    lines.append(f"- backend: `{summary['config']['backend']}`")
    lines.append(f"- tasks: `{','.join(summary['config']['tasks'])}`")
    lines.append(f"- task_count: `{summary['task_count']}`")
    lines.append("")
    lines.append("| model | policy reliability | json | sql | code | policy latency (s) | contract latency (s) | avg policy candidates |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["leaderboard"]:
        lines.append(
            f"| `{row['model']}` | {row['overall']['policy_ok_rate']:.3f} | "
            f"{_fmt_task(row, 'json')} | "
            f"{_fmt_task(row, 'sql')} | "
            f"{_fmt_task(row, 'codestub')} | "
            f"{row['overall']['policy_avg_latency_s']:.3f} | {row['overall']['avg_latency_s']:.3f} | "
            f"{row['overall']['avg_candidates_policy']:.3f} |"
        )
    lines.append("")
    lines.append("Artifacts:")
    lines.append("- `summary.json`")
    lines.append("- `samples.jsonl`")
    lines.append("- `wins.jsonl`")
    return "\n".join(lines) + "\n"
