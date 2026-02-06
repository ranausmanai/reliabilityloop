from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from autoquality.repair import RepairConfig, RepairReport, apply_edits, build_flagged_spans, build_repair_prompt, parse_repair_json
from autoquality.uncertainty import UncertaintyReport, UncertaintyScorer


BackendName = Literal["mock", "llamacpp", "ollama"]
UsedPath = Literal["fast", "slow"]
StrategyName = Literal["escalate", "repair"]


@dataclass(frozen=True)
class RouteConfig:
    entropy_threshold: float = 0.65
    margin_threshold: float = 0.08
    min_uncertain_tokens: int = 4
    escalate_always: bool = False
    use_draft: bool = True
    strategy: StrategyName = "escalate"
    repair: RepairConfig = field(default_factory=RepairConfig)


@dataclass(frozen=True)
class GenerateResult:
    used: UsedPath
    text: str
    uncertainty: UncertaintyReport
    repair: RepairReport | None = None


class Router:
    def __init__(
        self,
        *,
        backend: BackendName,
        model: str | None,
        fast_model: str | None,
        slow_model: str | None,
        config: RouteConfig | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.fast_model = fast_model
        self.slow_model = slow_model
        self.config = config or RouteConfig()
        self.scorer = UncertaintyScorer(
            entropy_threshold=self.config.entropy_threshold,
            margin_threshold=self.config.margin_threshold,
            min_uncertain_tokens=self.config.min_uncertain_tokens,
        )

    def generate(self, *, prompt: str, max_tokens: int, temperature: float) -> GenerateResult:
        fast_backend, slow_backend = self._resolve_backends()

        if self.config.escalate_always and slow_backend is not None:
            out = slow_backend.generate(prompt=prompt, max_tokens=max_tokens, temperature=min(0.2, temperature))
            report = UncertaintyReport(
                uncertain=True,
                reason="escalate_always",
                avg_entropy=None,
                min_margin=None,
                uncertain_tokens=0,
                uncertain_token_indices=[],
            )
            return GenerateResult(used="slow", text=out.text, uncertainty=report, repair=None)

        if slow_backend is None:
            fast_out = fast_backend.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            report = self.scorer.score(fast_out.logprobs)
            if report.uncertain and self.config.strategy == "repair":
                repaired, rep = self._attempt_repair(
                    user_prompt=prompt,
                    draft=fast_out.text,
                    logprobs=fast_out.logprobs,
                    report=report,
                    repair_backend=fast_backend,
                    model_used="fast",
                )
                return GenerateResult(used="fast", text=repaired, uncertainty=report, repair=rep)
            return GenerateResult(used="fast", text=fast_out.text, uncertainty=report, repair=None)

        # Two-stage: fast pass, score, then optional repair/escalation.
        fast_out = fast_backend.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        report = self.scorer.score(fast_out.logprobs)
        if not report.uncertain:
            return GenerateResult(used="fast", text=fast_out.text, uncertainty=report, repair=None)

        if self.config.strategy == "repair":
            repaired, rep = self._attempt_repair(
                user_prompt=prompt,
                draft=fast_out.text,
                logprobs=fast_out.logprobs,
                report=report,
                repair_backend=slow_backend,
                model_used="slow",
            )
            return GenerateResult(used="fast", text=repaired, uncertainty=report, repair=rep)

        slow_prompt = prompt
        if self.config.use_draft:
            slow_prompt = (
                prompt.rstrip()
                + "\n\nDraft answer (may contain mistakes):\n"
                + fast_out.text.rstrip()
                + "\n\nNow answer carefully and correct any mistakes:\n"
            )

        slow_out = slow_backend.generate(prompt=slow_prompt, max_tokens=max_tokens, temperature=min(0.2, temperature))
        return GenerateResult(used="slow", text=slow_out.text, uncertainty=report, repair=None)

    def _attempt_repair(
        self,
        *,
        user_prompt: str,
        draft: str,
        logprobs,
        report: UncertaintyReport,
        repair_backend,
        model_used: str,
    ) -> tuple[str, RepairReport]:
        flagged = build_flagged_spans(
            logprobs,
            uncertain_token_indices=report.uncertain_token_indices,
            max_spans=self.config.repair.max_spans,
            window_tokens=self.config.repair.window_tokens,
        )
        if not flagged:
            rep = RepairReport(attempted=False, applied_edits=0, parse_error="no_flagged_spans", model_used=model_used)
            return draft, rep

        repair_prompt = build_repair_prompt(
            user_prompt=user_prompt,
            draft=draft,
            flagged_spans=flagged,
            max_edits=self.config.repair.max_edits,
        )
        repair_out = repair_backend.generate(
            prompt=repair_prompt,
            max_tokens=self.config.repair.max_tokens,
            temperature=0.0,
        )
        edits, err = parse_repair_json(repair_out.text)
        if edits is None:
            rep = RepairReport(attempted=True, applied_edits=0, parse_error=err, model_used=model_used)
            return draft, rep
        repaired, applied = apply_edits(draft=draft, edits=edits, allowed_befores=flagged)
        rep = RepairReport(attempted=True, applied_edits=applied, parse_error=None, model_used=model_used)
        return repaired, rep

    def _resolve_backends(self):
        if self.backend == "mock":
            from autoquality.backends.mock import MockBackend

            return MockBackend(mode="fast"), (MockBackend(mode="slow") if self.slow_model else None)

        if self.backend == "llamacpp":
            from autoquality.backends.llamacpp import LlamaCppBackend

            if self.fast_model or self.slow_model:
                if not self.fast_model or not self.slow_model:
                    raise ValueError("For routing, provide both --fast-model and --slow-model.")
                return LlamaCppBackend(model_path=self.fast_model), LlamaCppBackend(model_path=self.slow_model)

            if not self.model:
                raise ValueError("Provide --model or --fast-model/--slow-model.")
            return LlamaCppBackend(model_path=self.model), None

        if self.backend == "ollama":
            from autoquality.backends.ollama import OllamaBackend

            if self.fast_model or self.slow_model:
                if not self.fast_model or not self.slow_model:
                    raise ValueError("For routing, provide both --fast-model and --slow-model (Ollama model names).")
                return OllamaBackend(model=self.fast_model), OllamaBackend(model=self.slow_model)

            if not self.model:
                raise ValueError("Provide --model (Ollama model name) or --fast-model/--slow-model.")
            return OllamaBackend(model=self.model), None

        raise ValueError(f"Unknown backend: {self.backend}")
