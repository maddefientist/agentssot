"""Reranker timeout wiring: settings.reranker_timeout_seconds must reach both
the fast and deep OllamaRerankerProvider instances, and OllamaRerankerProvider
must actually use it on the outbound HTTP call -- not silently fall back to
the class default of 30s.
"""
import os
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@127.0.0.1/test")

from app.reranker.ollama_provider import OllamaRerankerProvider
from app.reranker.router import build_reranker_pair


def _settings(**overrides):
    base = dict(
        reranker_provider="ollama",
        ollama_reranker_base_url="http://ollama-host:11434",
        ollama_base_url="http://ollama-host:11434",
        ollama_reranker_model="deep-model",
        ollama_reranker_fast_base_url="",
        ollama_reranker_fast_model="fast-model",
        reranker_scoring_mode="generate",
        reranker_timeout_seconds=5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_reranker_pair_uses_configured_timeout_default_5():
    fast, deep = build_reranker_pair(_settings())
    assert fast.timeout_seconds == 5
    assert deep.timeout_seconds == 5


def test_build_reranker_pair_respects_custom_timeout():
    fast, deep = build_reranker_pair(_settings(reranker_timeout_seconds=2))
    assert fast.timeout_seconds == 2
    assert deep.timeout_seconds == 2


def test_build_reranker_pair_falls_back_to_5_when_settings_field_absent():
    """A settings object predating this field (no reranker_timeout_seconds
    attribute at all) must still get the safe default, not crash or 30s."""
    s = _settings()
    del s.reranker_timeout_seconds
    fast, deep = build_reranker_pair(s)
    assert fast.timeout_seconds == 5
    assert deep.timeout_seconds == 5


def test_ollama_provider_default_timeout_is_30_when_unconfigured():
    """The class default stays 30s for direct construction (back-compat) --
    only the router's wiring lowers it. This pins that build_reranker_pair is
    the one actually responsible for the safer default."""
    p = OllamaRerankerProvider(base_url="http://x", model="m")
    assert p.timeout_seconds == 30


def test_ollama_provider_post_uses_configured_timeout_seconds():
    p = OllamaRerankerProvider(base_url="http://x", model="m", timeout_seconds=5)
    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"response": "0.5"}

    def fake_post(url, json, timeout):
        captured["timeout"] = timeout
        return _Resp()

    with patch("app.reranker.ollama_provider.httpx.post", fake_post):
        p._post({"model": "m", "prompt": "x"})

    assert captured["timeout"] == 5
