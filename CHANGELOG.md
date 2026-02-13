# Changelog

## 0.2.0 - 2026-02-13

### Added
- Reliability framework mode with executable verifiers for:
  - JSON schema + expected-field checks
  - SQL execution correctness on SQLite fixtures
  - Python unit-test correctness
- Adaptive inference controls:
  - `--best-of-k`
  - per-task overrides (`--best-of-k-json/sql/code`)
- Policy routing controls:
  - `--policy-json/sql/code` (`baseline_first` or `contract_first`)
- Per-task token budgets:
  - `--max-tokens-json/sql/code`
- Verified memory support:
  - `--memory-file`
  - `--memory-top-k`
  - run artifact `wins.jsonl`
- 60-task benchmark split:
  - `eval/reliability_v1_60.jsonl`

### Changed
- CLI program name shown as `reliabilityloop`
- Reliability leaderboard reports `policy` metrics as the primary score

### Notes
- This is a framework release focused on reproducible local reliability evaluation and inference-time improvement techniques.
