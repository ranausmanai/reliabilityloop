from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any, Literal

ContractKind = Literal["json", "sql", "codestub"]


@dataclass(frozen=True)
class ContractResult:
    ok: bool
    kind: ContractKind
    attempts: int
    repaired: bool
    errors: list[str]
    compiled: str | None
    parsed: Any | None
    used_model: str


def run_contract(
    *,
    backend: str,
    model: str,
    slow_model: str | None,
    kind: ContractKind,
    prompt: str,
    schema: dict[str, Any] | None = None,
    language: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    repair_attempts: int = 1,
) -> ContractResult:
    fast_backend = _resolve_backend(backend, model)
    slow_backend = _resolve_backend(backend, slow_model) if slow_model else None

    base_prompt = _build_contract_prompt(prompt=prompt, kind=kind, schema=schema, language=language)
    errors: list[str] = []
    attempts = 0
    repaired = False

    parsed = None
    compiled = None
    last_raw = ""
    last_used_model = model

    for attempt in range(repair_attempts + 1):
        attempts += 1
        if attempt == 0:
            out = fast_backend.generate(prompt=base_prompt, max_tokens=max_tokens, temperature=temperature)
            used_model = model
        else:
            repaired = True
            fix_prompt = _build_repair_prompt(
                kind=kind,
                schema=schema,
                language=language,
                last_raw=last_raw,
                errors=errors,
            )
            backend_to_use = slow_backend or fast_backend
            used_model = slow_model or model
            out = backend_to_use.generate(prompt=fix_prompt, max_tokens=max_tokens, temperature=0.0)

        last_raw = out.text
        last_used_model = used_model
        parsed, err = _parse_contract_output(kind=kind, text=out.text)
        if err:
            errors.append(err)
            continue

        err = _validate_contract(kind=kind, obj=parsed, schema=schema, language=language)
        if err:
            errors.append(err)
            continue

        compiled = _compile_contract(kind=kind, obj=parsed, language=language)
        if compiled is None:
            errors.append("compile_failed")
            continue

        return ContractResult(
            ok=True,
            kind=kind,
            attempts=attempts,
            repaired=repaired,
            errors=errors,
            compiled=compiled,
            parsed=parsed,
            used_model=used_model,
        )

    fallback = _code_fallback(kind=kind, text=last_raw, language=language)
    if fallback is not None:
        errors.append("fallback_code")
        return ContractResult(
            ok=True,
            kind=kind,
            attempts=attempts,
            repaired=True,
            errors=errors,
            compiled=fallback,
            parsed=None,
            used_model=last_used_model,
        )

    return ContractResult(
        ok=False,
        kind=kind,
        attempts=attempts,
        repaired=repaired,
        errors=errors or ["invalid_output"],
        compiled=None,
        parsed=parsed,
        used_model=last_used_model,
    )


def _resolve_backend(backend: str, model: str | None):
    if model is None:
        raise ValueError("Model is required for contract generation.")
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


def _build_contract_prompt(
    *,
    prompt: str,
    kind: ContractKind,
    schema: dict[str, Any] | None,
    language: str | None,
) -> str:
    header = (
        "CONTRACT_FIRST_OUTPUT\n"
        "You must return ONLY valid JSON. No markdown, no extra text.\n"
    )
    if kind == "json":
        schema_block = json.dumps(schema, ensure_ascii=False, indent=2) if schema else None
        if schema_block:
            return (
                header
                + "The JSON must satisfy this schema:\n"
                + schema_block
                + "\n\nUser request:\n"
                + prompt.strip()
            )
        return header + "Return a JSON value that fulfills the user request.\n\nUser request:\n" + prompt.strip()

    if kind == "sql":
        sql_schema = _sql_schema()
        schema_block = json.dumps(sql_schema, ensure_ascii=False, indent=2)
        return (
            header
            + "Return a JSON SQL AST matching this schema:\n"
            + schema_block
            + "\n\nUser request:\n"
            + prompt.strip()
        )

    if kind == "codestub":
        lang = language or "python"
        code_schema = _codestub_schema()
        schema_block = json.dumps(code_schema, ensure_ascii=False, indent=2)
        return (
            header
            + f"Return a JSON code stub for language: {lang}\n"
            + "Use this schema:\n"
            + schema_block
            + "\n\nUser request:\n"
            + prompt.strip()
        )

    raise ValueError(f"Unknown contract kind: {kind}")


def _build_repair_prompt(
    *,
    kind: ContractKind,
    schema: dict[str, Any] | None,
    language: str | None,
    last_raw: str,
    errors: list[str],
) -> str:
    header = (
        "CONTRACT_REPAIR\n"
        "Fix the JSON so it satisfies the contract. Return ONLY valid JSON.\n"
    )
    err_block = "\n".join(f"- {e}" for e in errors[-3:]) or "- unknown_error"
    body = f"Recent errors:\n{err_block}\n\nBroken output:\n{last_raw.strip()}\n"
    if kind == "json":
        schema_block = json.dumps(schema, ensure_ascii=False, indent=2) if schema else None
        if schema_block:
            return header + "Schema:\n" + schema_block + "\n\n" + body
        return header + body
    if kind == "sql":
        return header + "SQL AST schema:\n" + json.dumps(_sql_schema(), ensure_ascii=False, indent=2) + "\n\n" + body
    if kind == "codestub":
        lang = language or "python"
        return header + f"Language: {lang}\nSchema:\n" + json.dumps(_codestub_schema(), ensure_ascii=False, indent=2) + "\n\n" + body
    raise ValueError(f"Unknown contract kind: {kind}")


def _parse_contract_output(*, kind: ContractKind, text: str) -> tuple[Any | None, str | None]:
    obj, err = _extract_json(text)
    if err:
        return None, err
    return obj, None


def _validate_contract(*, kind: ContractKind, obj: Any, schema: dict[str, Any] | None, language: str | None) -> str | None:
    if kind == "json":
        if schema is None:
            return None
        return _validate_schema(obj, schema)
    if kind == "sql":
        return _validate_schema(obj, _sql_schema())
    if kind == "codestub":
        err = _validate_schema(obj, _codestub_schema())
        if err:
            return err
        if not isinstance(obj, dict):
            return "code_stub_not_object"
        lang = str(obj.get("language") or language or "python").lower()
        if lang not in {"python", "javascript"}:
            return f"unsupported_language:{lang}"
    return None


def _compile_contract(*, kind: ContractKind, obj: Any, language: str | None) -> str | None:
    if kind == "json":
        return json.dumps(obj, ensure_ascii=False, indent=2)
    if kind == "sql":
        return _compile_sql(obj)
    if kind == "codestub":
        lang = language
        if isinstance(obj, dict):
            lang = str(obj.get("language") or language or "python").lower()
        return _compile_codestub(obj, lang or "python")
    return None


def _code_fallback(*, kind: ContractKind, text: str, language: str | None) -> str | None:
    if kind != "codestub":
        return None
    code = _extract_code(text)
    if not code.strip():
        return None
    lang = (language or "python").lower()
    if lang == "python":
        try:
            ast.parse(code)
            return code.rstrip() + "\n"
        except Exception:
            return None
    if lang == "javascript":
        lowered = code.lower()
        if "function" in lowered or "=>" in code or "const " in lowered or "let " in lowered:
            return code.rstrip() + "\n"
    return None


def _extract_json(text: str) -> tuple[Any | None, str | None]:
    text = text.strip()
    if not text:
        return None, "empty_output"

    # Try direct JSON parse.
    try:
        return json.loads(text), None
    except Exception:
        pass

    # Try to locate the first JSON object/array and parse it.
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None, "no_json_found"

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
        return obj, None
    except Exception:
        return None, "json_parse_error"


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


def _validate_schema(obj: Any, schema: dict[str, Any], path: str = "$") -> str | None:
    stype = schema.get("type")
    if stype == "object":
        if not isinstance(obj, dict):
            return f"{path}:expected_object"
        required = schema.get("required") or []
        for key in required:
            if key not in obj:
                return f"{path}:missing:{key}"
        props = schema.get("properties") or {}
        additional = schema.get("additionalProperties", True)
        if additional is False:
            for key in obj.keys():
                if key not in props:
                    return f"{path}:unexpected:{key}"
        for key, sub in props.items():
            if key in obj:
                err = _validate_schema(obj[key], sub, f"{path}.{key}")
                if err:
                    return err
        return None
    if stype == "array":
        if not isinstance(obj, list):
            return f"{path}:expected_array"
        item_schema = schema.get("items")
        if item_schema is None:
            return None
        for i, item in enumerate(obj):
            err = _validate_schema(item, item_schema, f"{path}[{i}]")
            if err:
                return err
        return None
    if stype == "string":
        if not isinstance(obj, str):
            return f"{path}:expected_string"
        return None
    if stype == "number":
        if not isinstance(obj, (int, float)) or isinstance(obj, bool):
            return f"{path}:expected_number"
        return None
    if stype == "integer":
        if not isinstance(obj, int) or isinstance(obj, bool):
            return f"{path}:expected_integer"
        return None
    if stype == "boolean":
        if not isinstance(obj, bool):
            return f"{path}:expected_boolean"
        return None
    if stype == "null":
        if obj is not None:
            return f"{path}:expected_null"
        return None
    return None


def _sql_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["select", "from"],
        "properties": {
            "select": {"type": "array", "items": {"type": "string"}},
            "from": {"type": "string"},
            "where": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["col", "op", "val"],
                    "properties": {
                        "col": {"type": "string"},
                        "op": {"type": "string"},
                        "val": {},
                    },
                    "additionalProperties": False,
                },
            },
            "order_by": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["col"],
                    "properties": {
                        "col": {"type": "string"},
                        "dir": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            "limit": {"type": "integer"},
        },
        "additionalProperties": False,
    }


def _codestub_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["language", "functions"],
        "properties": {
            "language": {"type": "string"},
            "imports": {"type": "array", "items": {"type": "string"}},
            "functions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "args"],
                    "properties": {
                        "name": {"type": "string"},
                        "args": {"type": "array", "items": {"type": "string"}},
                        "returns": {"type": "string"},
                        "doc": {"type": "string"},
                        "body": {"type": "array", "items": {"type": "string"}},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


def _compile_sql(obj: Any) -> str | None:
    if not isinstance(obj, dict):
        return None
    select_cols = obj.get("select")
    if not isinstance(select_cols, list) or not select_cols:
        return None
    from_tbl = obj.get("from")
    if not isinstance(from_tbl, str) or not from_tbl.strip():
        return None
    parts = ["SELECT " + ", ".join(_sql_ident(c) for c in select_cols), "FROM " + _sql_ident(from_tbl)]

    where = obj.get("where") or []
    if isinstance(where, list) and where:
        clauses: list[str] = []
        for cond in where:
            if not isinstance(cond, dict):
                continue
            col = cond.get("col")
            op = cond.get("op")
            val = cond.get("val")
            if not isinstance(col, str) or not isinstance(op, str):
                continue
            clauses.append(f"{_sql_ident(col)} {op.strip()} {_sql_value(val)}")
        if clauses:
            parts.append("WHERE " + " AND ".join(clauses))

    order_by = obj.get("order_by") or []
    if isinstance(order_by, list) and order_by:
        order_clauses: list[str] = []
        for item in order_by:
            if not isinstance(item, dict):
                continue
            col = item.get("col")
            if not isinstance(col, str):
                continue
            direction = str(item.get("dir") or "ASC").upper()
            if direction not in {"ASC", "DESC"}:
                direction = "ASC"
            order_clauses.append(f"{_sql_ident(col)} {direction}")
        if order_clauses:
            parts.append("ORDER BY " + ", ".join(order_clauses))

    limit = obj.get("limit")
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        parts.append(f"LIMIT {limit}")

    return " ".join(parts)


def _sql_ident(name: str) -> str:
    cleaned = name.strip()
    if cleaned.replace("_", "").isalnum():
        return cleaned
    return '"' + cleaned.replace('"', '""') + '"'


def _sql_value(val: Any) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return str(val)
    if isinstance(val, str):
        return "'" + val.replace("'", "''") + "'"
    return "'" + str(val).replace("'", "''") + "'"


def _compile_codestub(obj: Any, language: str) -> str | None:
    if not isinstance(obj, dict):
        return None
    language = language.lower()
    if language == "python":
        return _compile_python_stub(obj)
    if language == "javascript":
        return _compile_js_stub(obj)
    return None


def _compile_python_stub(obj: dict[str, Any]) -> str:
    lines: list[str] = []
    imports = obj.get("imports") or []
    if isinstance(imports, list) and imports:
        for imp in imports:
            if isinstance(imp, str) and imp.strip():
                lines.append(f"import {imp.strip()}")
        lines.append("")
    funcs = obj.get("functions") or []
    if not isinstance(funcs, list):
        funcs = []
    for fn in funcs:
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "func")
        args = fn.get("args") or []
        if not isinstance(args, list):
            args = []
        args_s = ", ".join(str(a) for a in args)
        returns = fn.get("returns")
        ret_s = f" -> {returns}" if isinstance(returns, str) and returns.strip() else ""
        lines.append(f"def {name}({args_s}){ret_s}:")
        doc = fn.get("doc")
        if isinstance(doc, str) and doc.strip():
            lines.append(f'    """{doc.strip()}"""')
        body = fn.get("body")
        if isinstance(body, list) and body:
            for b in body:
                if isinstance(b, str) and b.strip():
                    lines.append("    " + b.rstrip())
        else:
            lines.append("    pass")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _compile_js_stub(obj: dict[str, Any]) -> str:
    lines: list[str] = []
    imports = obj.get("imports") or []
    if isinstance(imports, list) and imports:
        for imp in imports:
            if isinstance(imp, str) and imp.strip():
                lines.append(f"const {imp.strip()} = require('{imp.strip()}');")
        lines.append("")
    funcs = obj.get("functions") or []
    if not isinstance(funcs, list):
        funcs = []
    for fn in funcs:
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "func")
        args = fn.get("args") or []
        if not isinstance(args, list):
            args = []
        args_s = ", ".join(str(a) for a in args)
        lines.append(f"function {name}({args_s}) {{")
        doc = fn.get("doc")
        if isinstance(doc, str) and doc.strip():
            lines.append(f"  // {doc.strip()}")
        body = fn.get("body")
        if isinstance(body, list) and body:
            for b in body:
                if isinstance(b, str) and b.strip():
                    lines.append("  " + b.rstrip())
        else:
            lines.append("  // TODO: implement")
        lines.append("}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
