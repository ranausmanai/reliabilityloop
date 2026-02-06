from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class BackendOutput:
    text: str
    # List of per-token logprob objects (OpenAI-style-ish):
    # [{"token": "...", "logprob": -0.1, "top_logprobs": {"...": -0.2, ...}}, ...]
    logprobs: list[dict[str, Any]] | None = None


class LLMBackend(Protocol):
    def generate(self, *, prompt: str, max_tokens: int, temperature: float) -> BackendOutput: ...

