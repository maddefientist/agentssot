from types import SimpleNamespace

import pytest

from app.runtime_config import HOT_KEYS, coerce_value


def _settings():
    return SimpleNamespace(
        synthesis_model="qwen3.5:397b-cloud",
        synthesis_fallback_model="qwen3:latest",
        synthesis_window_days=7,
        synthesis_similarity_threshold=0.65,
        synthesis_min_cluster_size=2,
        ollama_reranker_model="deep",
        ollama_reranker_base_url="http://host.docker.internal:11434",
        ollama_reranker_fast_model="fast",
        ollama_reranker_fast_base_url="",
        ollama_embed_model="nomic-embed-text",
        ollama_base_url="http://host.docker.internal:11434",
        classifier_model="gemma4:31b-cloud",
        classifier_base_url="",
        semantic_dedup_threshold=0.0,
    )


def test_hot_keys_are_allow_listed_and_typed():
    s = _settings()
    assert "database_url" not in HOT_KEYS
    assert "synthesis_window_days" in HOT_KEYS
    assert coerce_value(s, "synthesis_window_days", "14") == 14
    assert coerce_value(s, "synthesis_min_cluster_size", "3") == 3
    assert coerce_value(s, "synthesis_similarity_threshold", "0.7") == 0.7
    assert coerce_value(s, "synthesis_model", "qwen3.5:397b-cloud") == "qwen3.5:397b-cloud"


def test_rejects_unknown_key():
    with pytest.raises(ValueError):
        coerce_value(_settings(), "database_url", "postgres://example")


def test_rejects_bad_provider_urls():
    s = _settings()
    with pytest.raises(ValueError):
        coerce_value(s, "ollama_base_url", "file:///etc/passwd")
    with pytest.raises(ValueError):
        coerce_value(s, "ollama_base_url", "http://user:pass@example.local")
