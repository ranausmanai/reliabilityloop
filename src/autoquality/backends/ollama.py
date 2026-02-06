from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from autoquality.backends.base import BackendOutput


class OllamaBackend:
    def __init__(self, *, model: str, base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, *, prompt: str, max_tokens: int, temperature: float) -> BackendOutput:
        url = f"{self.base_url}/api/generate"
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # Ensure thinking-capable models put the final answer in `response`.
            # (Otherwise some models may emit only `thinking` and leave `response` empty.)
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "logprobs": True,
            "top_logprobs": 10,
        }

        # For repair calls, request structured output when supported.
        if "AUTOQUALITY_REPAIR" in prompt:
            body["format"] = {
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"before": {"type": "string"}, "after": {"type": "string"}},
                            "required": ["before", "after"],
                            "additionalProperties": False,
                        },
                    },
                    "notes": {"type": "string"},
                },
                "required": ["edits", "notes"],
                "additionalProperties": False,
            }

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        payload = None
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as e:
                last_err = e
                # Retry once without structured output in case the server/model doesn't support it.
                if attempt == 0 and "format" in body:
                    body.pop("format", None)
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(body).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    continue
                break

        if payload is None:
            raise RuntimeError(
                "Failed to call Ollama at localhost. Ensure Ollama is running "
                "(`ollama serve`) and the model name is correct."
            ) from last_err

        response = payload.get("response")
        text = str(response) if response is not None else ""
        logprobs_raw = payload.get("logprobs")
        logprobs = _normalize_ollama_logprobs(logprobs_raw)
        return BackendOutput(text=text, logprobs=logprobs)


def _normalize_ollama_logprobs(logprobs_raw: Any) -> list[dict[str, Any]] | None:
    if not isinstance(logprobs_raw, list):
        return None

    normalized: list[dict[str, Any]] = []
    for item in logprobs_raw:
        if not isinstance(item, dict):
            continue

        top = item.get("top_logprobs")
        top_map: dict[str, float] | None = None
        if isinstance(top, list):
            tmp: dict[str, float] = {}
            for t in top:
                if not isinstance(t, dict):
                    continue
                tok = t.get("token")
                lp = t.get("logprob")
                if isinstance(tok, str) and isinstance(lp, (int, float)):
                    tmp[tok] = float(lp)
            top_map = tmp if tmp else None

        normalized.append(
            {
                "token": item.get("token"),
                "logprob": item.get("logprob"),
                "top_logprobs": top_map,
            }
        )

    return normalized
