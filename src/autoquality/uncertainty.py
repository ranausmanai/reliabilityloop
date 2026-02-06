from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UncertaintyReport:
    uncertain: bool
    reason: str
    avg_entropy: float | None
    min_margin: float | None
    uncertain_tokens: int
    uncertain_token_indices: list[int]


class UncertaintyScorer:
    def __init__(self, *, entropy_threshold: float, margin_threshold: float, min_uncertain_tokens: int) -> None:
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold
        self.min_uncertain_tokens = min_uncertain_tokens

    def score(self, logprobs: list[dict[str, Any]] | None) -> UncertaintyReport:
        if not logprobs:
            # Fallback when backend cannot provide logprobs.
            return UncertaintyReport(
                uncertain=True,
                reason="no_logprobs",
                avg_entropy=None,
                min_margin=None,
                uncertain_tokens=0,
                uncertain_token_indices=[],
            )

        entropies: list[float] = []
        margins: list[float] = []
        uncertain = 0
        uncertain_token_indices: list[int] = []

        for i, tok in enumerate(logprobs):
            top = tok.get("top_logprobs")
            if not isinstance(top, dict) or len(top) < 2:
                continue
            entropy, margin = _entropy_and_margin_from_top_logprobs(top)
            entropies.append(entropy)
            margins.append(margin)
            if entropy > self.entropy_threshold or margin < self.margin_threshold:
                uncertain += 1
                uncertain_token_indices.append(i)

        if not entropies or not margins:
            return UncertaintyReport(
                uncertain=True,
                reason="insufficient_logprobs",
                avg_entropy=None,
                min_margin=None,
                uncertain_tokens=0,
                uncertain_token_indices=[],
            )

        avg_entropy = sum(entropies) / len(entropies)
        min_margin = min(margins)

        is_uncertain = uncertain >= self.min_uncertain_tokens
        reason = "ok"
        if is_uncertain:
            reason = f"uncertain_tokens>={self.min_uncertain_tokens}"

        return UncertaintyReport(
            uncertain=is_uncertain,
            reason=reason,
            avg_entropy=avg_entropy,
            min_margin=min_margin,
            uncertain_tokens=uncertain,
            uncertain_token_indices=uncertain_token_indices,
        )


def _entropy_and_margin_from_top_logprobs(top_logprobs: dict[str, float]) -> tuple[float, float]:
    # Approximate entropy using the provided top-k distribution (renormalized).
    lps = sorted(top_logprobs.values(), reverse=True)
    max_lp = lps[0]
    ps = [math.exp(lp - max_lp) for lp in lps]
    z = sum(ps)
    ps = [p / z for p in ps]

    entropy = 0.0
    for p in ps:
        entropy -= p * math.log(max(p, 1e-12))

    margin = ps[0] - ps[1] if len(ps) > 1 else 0.0
    return entropy, margin
