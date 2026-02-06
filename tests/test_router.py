from autoquality.router import RouteConfig, Router


def test_escalates_on_uncertainty_mock():
    router = Router(backend="mock", model=None, fast_model="x", slow_model="y", config=RouteConfig(min_uncertain_tokens=1))
    res = router.generate(prompt="write a parser", max_tokens=32, temperature=0.2)
    assert res.used == "slow"
    assert res.uncertainty.uncertain is True


def test_fast_when_no_slow_backend():
    router = Router(backend="mock", model=None, fast_model=None, slow_model=None, config=RouteConfig())
    res = router.generate(prompt="hello", max_tokens=16, temperature=0.2)
    assert res.used == "fast"


def test_ollama_logprobs_normalization():
    from autoquality.backends.ollama import _normalize_ollama_logprobs

    raw = [
        {
            "token": "hello",
            "logprob": -0.01,
            "top_logprobs": [{"token": "hello", "logprob": -0.01}, {"token": "hi", "logprob": -3.0}],
        }
    ]
    norm = _normalize_ollama_logprobs(raw)
    assert norm is not None
    assert norm[0]["top_logprobs"]["hello"] == -0.01
    assert norm[0]["top_logprobs"]["hi"] == -3.0


def test_repair_strategy_applies_patch():
    cfg = RouteConfig(min_uncertain_tokens=1, strategy="repair")
    router = Router(backend="mock", model=None, fast_model="x", slow_model="y", config=cfg)
    res = router.generate(prompt="write a parser", max_tokens=32, temperature=0.2)
    assert res.used == "fast"
    assert res.repair is not None
    assert res.repair.applied_edits == 1
    assert "final answer." in res.text
