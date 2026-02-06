from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast-model", default="qwen2.5-coder:7b")
    ap.add_argument("--slow-model", default="qwen2.5-coder:14b")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--outroot", default="eval/runs")
    ap.add_argument("--nice", type=int, default=15, help="Increase niceness to reduce UI impact (0-20).")
    ap.add_argument("--modes", default="baseline_fast,escalate,repair")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-normalize-completions", action="store_true")

    # Routing sensitivity (pass-through)
    ap.add_argument("--entropy-threshold", type=float, default=0.65)
    ap.add_argument("--margin-threshold", type=float, default=0.08)
    ap.add_argument("--min-uncertain-tokens", type=int, default=10)

    # Repair params (pass-through)
    ap.add_argument("--repair-max-tokens", type=int, default=256)
    ap.add_argument("--repair-max-spans", type=int, default=3)
    ap.add_argument("--repair-window-tokens", type=int, default=8)
    ap.add_argument("--repair-max-edits", type=int, default=2)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outroot) / ts
    outdir.mkdir(parents=True, exist_ok=True)

    log_path = outdir / "run.log"
    pid_path = outdir / "run.pid"

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "src")

    cmd = [
        sys.executable,
        "eval/humaneval_runner.py",
        "--backend",
        "ollama",
        "--fast-model",
        args.fast_model,
        "--slow-model",
        args.slow_model,
        "--limit",
        str(args.limit),
        "--outdir",
        str(outdir),
        "--modes",
        str(args.modes),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--entropy-threshold",
        str(args.entropy_threshold),
        "--margin-threshold",
        str(args.margin_threshold),
        "--min-uncertain-tokens",
        str(args.min_uncertain_tokens),
        "--repair-max-tokens",
        str(args.repair_max_tokens),
        "--repair-max-spans",
        str(args.repair_max_spans),
        "--repair-window-tokens",
        str(args.repair_window_tokens),
        "--repair-max-edits",
        str(args.repair_max_edits),
    ]
    if not args.no_normalize_completions:
        cmd.append("--normalize-completions")

    with open(log_path, "wb") as logf:
        def _preexec() -> None:
            try:
                os.nice(max(0, min(20, int(args.nice))))
            except Exception:
                pass

        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=logf,
            env=env,
            start_new_session=True,
            preexec_fn=_preexec,
        )

    pid_path.write_text(str(proc.pid), encoding="utf-8")
    print(str(outdir))


if __name__ == "__main__":
    main()
