from __future__ import annotations

import random

from autoquality.backends.base import BackendOutput


class MockBackend:
    def __init__(self, *, mode: str) -> None:
        self.mode = mode

    def generate(self, *, prompt: str, max_tokens: int, temperature: float) -> BackendOutput:
        # Deterministic-ish behavior for tests/demos.
        seed = sum(ord(c) for c in (prompt + self.mode)) % 10_000
        rng = random.Random(seed)

        if self.mode == "fast":
            text = f"[FAST] {prompt.strip()} -> draft answer."
            # Make it "uncertain" by giving flat-ish top probs.
            logprobs = []
            toks = ["draft", " answer", "."]
            for i in range(3):
                top = {"A": -0.8 + rng.random() * 0.1, "B": -0.85 + rng.random() * 0.1, "C": -0.9 + rng.random() * 0.1}
                logprobs.append({"token": toks[i % len(toks)], "logprob": -0.8, "top_logprobs": top})
            return BackendOutput(text=text, logprobs=logprobs)

        if "AUTOQUALITY_REPAIR" in prompt:
            # Return a deterministic JSON patch that targets the flagged span "draft answer."
            return BackendOutput(
                text='{"edits":[{"before":"draft answer.","after":"final answer."}],"notes":"fixed"}',
                logprobs=None,
            )

        text = f"[SLOW] {prompt.strip()} -> careful answer."
        # Confident distribution.
        logprobs = []
        for _ in range(3):
            top = {"A": -0.01, "B": -4.0, "C": -6.0}
            logprobs.append({"token": "x", "logprob": -0.01, "top_logprobs": top})
        return BackendOutput(text=text, logprobs=logprobs)
