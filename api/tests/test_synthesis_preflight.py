import os
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://ssot:test@localhost:5432/ssot")

import pytest
from app.synthesis import preflight
from app.llm.model_validation import ModelListUnavailable


def _patch(monkeypatch, present, missing):
    monkeypatch.setattr(preflight, "validate_models", lambda base, req, timeout=5.0: (set(present), set(missing)))


def test_all_present_proceeds_clean(monkeypatch):
    _patch(monkeypatch, ["p", "f"], [])
    r = preflight.evaluate("http://o", "p", "f")
    assert r.proceed and r.primary == "p" and r.fallback == "f" and r.severity is None


def test_primary_missing_degrades_to_fallback(monkeypatch):
    _patch(monkeypatch, ["f"], ["p"])
    r = preflight.evaluate("http://o", "p", "f")
    assert r.proceed and r.primary == "f" and r.fallback is None and r.severity == "warning"


def test_fallback_missing_runs_without_fallback(monkeypatch):
    _patch(monkeypatch, ["p"], ["f"])
    r = preflight.evaluate("http://o", "p", "f")
    assert r.proceed and r.primary == "p" and r.fallback is None and r.severity == "warning"


def test_both_missing_skips(monkeypatch):
    _patch(monkeypatch, [], ["p", "f"])
    r = preflight.evaluate("http://o", "p", "f")
    assert not r.proceed and r.severity == "error" and r.event == "synthesis.model_missing"


def test_unreachable_skips(monkeypatch):
    def raiser(base, req, timeout=5.0):
        raise ModelListUnavailable("down")
    monkeypatch.setattr(preflight, "validate_models", raiser)
    r = preflight.evaluate("http://o", "p", "f")
    assert not r.proceed and r.event == "synthesis.unreachable" and r.severity == "error"

def test_empty_fallback_treated_missing(monkeypatch):
    # empty fallback string is filtered from validate_models' required set;
    # preflight must still treat it as "no fallback" and run on primary only.
    _patch(monkeypatch, ["p"], [])
    r = preflight.evaluate("http://o", "p", "")
    assert r.proceed and r.primary == "p" and r.fallback is None and r.severity == "warning"


def test_empty_primary_promotes_fallback(monkeypatch):
    _patch(monkeypatch, ["f"], [])
    r = preflight.evaluate("http://o", "", "f")
    assert r.proceed and r.primary == "f" and r.fallback is None and r.severity == "warning"


def test_both_empty_skips(monkeypatch):
    _patch(monkeypatch, [], [])
    r = preflight.evaluate("http://o", "", "")
    assert not r.proceed and r.severity == "error" and r.event == "synthesis.model_missing"
