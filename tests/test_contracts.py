from autoquality.contracts import _code_fallback, _compile_codestub, _compile_sql, _extract_json, _validate_schema


def test_extract_json_embedded():
    obj, err = _extract_json("prefix {\"a\": 1, \"b\": [2, 3]} suffix")
    assert err is None
    assert obj["a"] == 1
    assert obj["b"] == [2, 3]


def test_validate_schema_object():
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"type": "integer"}},
        "additionalProperties": False,
    }
    assert _validate_schema({"a": 1}, schema) is None
    assert _validate_schema({"a": "1"}, schema) is not None
    assert _validate_schema({"a": 1, "b": 2}, schema) is not None


def test_compile_sql_basic():
    ast = {
        "select": ["id", "name"],
        "from": "users",
        "where": [{"col": "age", "op": ">", "val": 30}],
        "order_by": [{"col": "id", "dir": "desc"}],
        "limit": 5,
    }
    sql = _compile_sql(ast)
    assert sql == "SELECT id, name FROM users WHERE age > 30 ORDER BY id DESC LIMIT 5"


def test_compile_codestub_python():
    obj = {
        "language": "python",
        "imports": ["typing"],
        "functions": [
            {"name": "foo", "args": ["x"], "doc": "do x", "body": ["return x + 1"]},
        ],
    }
    code = _compile_codestub(obj, "python")
    assert "def foo(x):" in code
    assert "return x + 1" in code


def test_code_fallback_python():
    code = "def hello():\n    return 1\n"
    out = _code_fallback(kind="codestub", text=code, language="python")
    assert out is not None
