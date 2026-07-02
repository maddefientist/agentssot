import os
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://ssot:test@localhost:5432/ssot")

import pytest
from app.llm import model_validation as mv


class _Resp:
    def __init__(self, payload): self._payload = payload
    def raise_for_status(self): pass
    def json(self): return self._payload


def _fake_get(payload):
    def _get(url, timeout):
        assert url.endswith("/api/tags")
        return _Resp(payload)
    return _get


def test_validate_models_present_and_missing(monkeypatch):
    payload = {"models": [{"name": "qwen3.6:27b"}, {"name": "qwen3.5:397b-cloud"}]}
    monkeypatch.setattr(mv.httpx, "get", _fake_get(payload))
    present, missing = mv.validate_models("http://ollama:11434", ["qwen3.6:27b", "qwen3.5:27b"])
    assert present == {"qwen3.6:27b"}
    assert missing == {"qwen3.5:27b"}


def test_validate_models_all_present(monkeypatch):
    payload = {"models": [{"name": "a"}, {"name": "b"}]}
    monkeypatch.setattr(mv.httpx, "get", _fake_get(payload))
    present, missing = mv.validate_models("http://ollama:11434", ["a", "b"])
    assert missing == set()


def test_list_unavailable_raises(monkeypatch):
    def raiser(url, timeout):
        raise RuntimeError("conn refused")
    monkeypatch.setattr(mv.httpx, "get", raiser)
    with pytest.raises(mv.ModelListUnavailable):
        mv.list_available_models("http://ollama:11434")

def test_malformed_200_raises_unavailable(monkeypatch):
    class _BadResp:
        def raise_for_status(self): pass
        def json(self): raise ValueError("not json")
    monkeypatch.setattr(mv.httpx, "get", lambda url, timeout: _BadResp())
    with pytest.raises(mv.ModelListUnavailable):
        mv.list_available_models("http://ollama:11434")
