from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RepairConfig:
    max_spans: int = 4
    window_tokens: int = 8
    max_edits: int = 2
    max_tokens: int = 512


@dataclass(frozen=True)
class RepairReport:
    attempted: bool
    applied_edits: int
    parse_error: str | None
    model_used: str


def build_flagged_spans(
    logprobs: list[dict[str, Any]] | None,
    *,
    uncertain_token_indices: list[int],
    max_spans: int,
    window_tokens: int,
    max_chars: int = 240,
) -> list[str]:
    if not logprobs or not uncertain_token_indices:
        return []

    tokens: list[str] = []
    for item in logprobs:
        tok = item.get("token")
        if isinstance(tok, str):
            tokens.append(tok)
        else:
            tokens.append("")

    # Group indices into contiguous runs.
    runs: list[tuple[int, int]] = []
    for idx in sorted(set(uncertain_token_indices)):
        if not runs or idx > runs[-1][1] + 1:
            runs.append((idx, idx))
        else:
            runs[-1] = (runs[-1][0], idx)

    spans: list[str] = []
    for start, end in runs:
        if len(spans) >= max_spans:
            break
        a = max(0, start - window_tokens)
        b = min(len(tokens), end + window_tokens + 1)
        snippet = "".join(tokens[a:b]).strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip()
        if snippet and snippet not in spans:
            spans.append(snippet)
    return spans


def build_repair_prompt(*, user_prompt: str, draft: str, flagged_spans: list[str], max_edits: int) -> str:
    spans_block = "\n".join(f"{i+1}. {json.dumps(s, ensure_ascii=False)}" for i, s in enumerate(flagged_spans)) or "None"
    # The contract is intentionally strict so we can apply edits deterministically.
    return (
        "AUTOQUALITY_REPAIR\n"
        "You are fixing a draft answer from a coding assistant.\n"
        "Task: Apply minimal edits to correct mistakes, focusing on the flagged spans.\n\n"
        f"User prompt:\n{user_prompt.strip()}\n\n"
        f"Draft answer:\n{draft.strip()}\n\n"
        "Flagged spans (exact substrings from the draft):\n"
        f"{spans_block}\n\n"
        "Return ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "edits": [\n'
        '    {"before": "<one exact flagged span string>", "after": "<replacement text>"},\n'
        '    ... up to ' + str(max_edits) + " edits\n"
        "  ],\n"
        '  "notes": "<optional short note>"\n'
        "}\n\n"
        "Rules:\n"
        "- `before` MUST be exactly one of the flagged spans.\n"
        "- Keep `after` minimal; do not rewrite the whole answer.\n"
        "- If nothing needs fixing, return {\"edits\":[],\"notes\":\"ok\"}.\n"
        "- Do not include markdown fences.\n"
    )


def parse_repair_json(text: str) -> tuple[list[dict[str, str]] | None, str | None]:
    text = text.strip()
    if not text:
        return None, "empty_response"

    # Try direct JSON first.
    try:
        obj = json.loads(text)
        return _extract_edits(obj)
    except Exception:
        pass

    # Otherwise, extract the first balanced JSON object block.
    extracted = _extract_first_json_object(text)
    if extracted is None:
        # Common failure mode: response got truncated mid-JSON.
        if "{" in text and "}" not in text:
            return None, "incomplete_json"
        return None, "no_json_object_found"
    try:
        obj = json.loads(extracted)
        return _extract_edits(obj)
    except Exception as e:
        return None, f"json_parse_error:{type(e).__name__}"


def _extract_first_json_object(s: str) -> str | None:
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _extract_edits(obj: Any) -> tuple[list[dict[str, str]] | None, str | None]:
    if not isinstance(obj, dict):
        return None, "json_not_object"
    edits = obj.get("edits")
    if edits is None:
        return [], None
    if not isinstance(edits, list):
        return None, "edits_not_list"
    parsed: list[dict[str, str]] = []
    for e in edits:
        if not isinstance(e, dict):
            continue
        before = e.get("before")
        after = e.get("after")
        if isinstance(before, str) and isinstance(after, str):
            parsed.append({"before": before, "after": after})
    return parsed, None


def apply_edits(
    *,
    draft: str,
    edits: list[dict[str, str]],
    allowed_befores: list[str],
    max_before_chars: int = 600,
) -> tuple[str, int]:
    out = draft
    applied = 0
    allowed = set(allowed_befores)
    for e in edits:
        before = e["before"]
        after = e["after"]
        # Prefer strict application to flagged spans, but allow edits that clearly target the draft.
        if before not in allowed:
            if not before or before not in out or len(before) > max_before_chars:
                continue
        if before and before in out:
            out = out.replace(before, after, 1)
            applied += 1
    return out, applied
