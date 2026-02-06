from __future__ import annotations

from autoquality.backends.base import BackendOutput


class LlamaCppBackend:
    def __init__(self, *, model_path: str) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError('llama-cpp backend requires `pip install "autoquality[llamacpp]"`.') from e

        # Keep defaults conservative; users can tune via env/config later.
        self._llm = Llama(model_path=model_path, logits_all=True, n_ctx=4096)

    def generate(self, *, prompt: str, max_tokens: int, temperature: float) -> BackendOutput:
        # Best-effort logprob extraction; llama-cpp-python APIs vary by version.
        try:
            out = self._llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                logprobs=10,
            )
            choice = out["choices"][0]
            text = choice.get("text", "")
            logprobs = None
            lp = choice.get("logprobs")
            if isinstance(lp, dict) and isinstance(lp.get("top_logprobs"), list):
                tokens = lp.get("tokens") or []
                token_lps = lp.get("token_logprobs") or []
                top_lps = lp.get("top_logprobs") or []
                merged = []
                for i in range(min(len(tokens), len(top_lps))):
                    merged.append({"token": tokens[i], "logprob": token_lps[i] if i < len(token_lps) else None, "top_logprobs": top_lps[i]})
                logprobs = merged
            return BackendOutput(text=text, logprobs=logprobs)
        except Exception:
            # Fall back to plain generation without logprobs.
            text = self._llm(prompt, max_tokens=max_tokens, temperature=temperature)["choices"][0]["text"]
            return BackendOutput(text=text, logprobs=None)

