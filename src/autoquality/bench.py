from __future__ import annotations

import json
import time
from collections.abc import Iterable

from autoquality.router import RouteConfig, Router


def _iter_prompts(path: str) -> Iterable[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                continue
            yield {"id": str(obj.get("id", "")), "prompt": str(obj["prompt"])}


def run_bench(
    *,
    backend: str,
    model: str | None,
    fast_model: str | None,
    slow_model: str | None,
    prompts_file: str,
    config: RouteConfig,
    max_tokens: int,
    temperature: float,
) -> dict[str, object]:
    router = Router(backend=backend, model=model, fast_model=fast_model, slow_model=slow_model, config=config)

    used_fast = 0
    used_slow = 0
    ms_fast: list[float] = []
    ms_slow: list[float] = []
    repairs_attempted = 0
    repairs_with_edits = 0

    prompts = list(_iter_prompts(prompts_file))
    for item in prompts:
        t0 = time.perf_counter()
        res = router.generate(prompt=item["prompt"], max_tokens=max_tokens, temperature=temperature)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if res.repair is not None and res.repair.attempted:
            repairs_attempted += 1
            if res.repair.applied_edits > 0:
                repairs_with_edits += 1

        if res.used == "fast":
            used_fast += 1
            ms_fast.append(dt_ms)
        else:
            used_slow += 1
            ms_slow.append(dt_ms)

    avg_ms_fast = sum(ms_fast) / len(ms_fast) if ms_fast else 0.0
    avg_ms_slow = sum(ms_slow) / len(ms_slow) if ms_slow else 0.0
    total = len(prompts) if prompts else 0
    escalation_rate = (used_slow / total) if total else 0.0

    return {
        "prompts": total,
        "used_fast": used_fast,
        "used_slow": used_slow,
        "escalation_rate": escalation_rate,
        "avg_ms_fast": avg_ms_fast,
        "avg_ms_slow": avg_ms_slow,
        "repairs_attempted": repairs_attempted,
        "repairs_with_edits": repairs_with_edits,
    }
