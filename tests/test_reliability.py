import json

from autoquality.reliability import ReliabilityConfig, _validate_output, run_reliability


def test_run_reliability_mock(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    rows = [
        {
            "id": "json-1",
            "kind": "json",
            "prompt": "Extract user data.",
            "schema": {
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            },
        },
        {"id": "sql-1", "kind": "sql", "prompt": "Table users(id). Select id from users."},
        {"id": "code-1", "kind": "codestub", "language": "python", "prompt": "Write a python function foo()."},
    ]
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    outdir = tmp_path / "run"
    summary = run_reliability(
        ReliabilityConfig(
            backend="mock",
            models=["mock-fast"],
            prompts_file=str(prompts),
            tasks=["json", "sql", "codestub"],
            limit=3,
            max_tokens=64,
            temperature=0.0,
            repair_attempts=1,
            slow_model=None,
        ),
        outdir=str(outdir),
    )

    assert summary["task_count"] == 3
    assert len(summary["leaderboard"]) == 1
    assert (outdir / "summary.json").exists()
    assert (outdir / "samples.jsonl").exists()
    assert (outdir / "leaderboard.md").exists()


def test_validate_output_json_with_expected_fields():
    item = {
        "kind": "json",
        "schema": {
            "type": "object",
            "required": ["a", "b"],
            "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
            "additionalProperties": False,
        },
        "expected": {"a": 3},
    }
    assert _validate_output(item, '{"a": 3, "b": "ok"}')
    assert not _validate_output(item, '{"a": 4, "b": "ok"}')


def test_validate_output_sql_executes_and_matches():
    item = {
        "kind": "sql",
        "db": {
            "tables": [
                {
                    "name": "users",
                    "columns": [["id", "INTEGER"], ["name", "TEXT"]],
                    "rows": [[1, "Ada"], [2, "Mia"]],
                }
            ]
        },
        "expected": {"columns": ["name"], "rows": [["Ada"], ["Mia"]], "order_sensitive": True},
    }
    assert _validate_output(item, "SELECT name FROM users ORDER BY id")
    assert not _validate_output(item, "SELECT id FROM users ORDER BY id")


def test_validate_output_code_runs_tests():
    item = {
        "kind": "codestub",
        "language": "python",
        "entry_point": "clamp",
        "tests": [
            {"args": [5, 1, 10], "equals": 5},
            {"args": [-1, 0, 10], "equals": 0},
            {"args": [12, 0, 10], "equals": 10},
        ],
    }
    good = "def clamp(x, low, high):\n    return max(low, min(high, x))\n"
    bad = "def clamp(x, low, high):\n    return x\n"
    assert _validate_output(item, good)
    assert not _validate_output(item, bad)
