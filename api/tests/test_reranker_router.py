"""Two-tier reranker router routes by tier set.

Tests are pure-Python (no DB, no .env). select_reranker_model reads
settings.procedural_tiers; we monkeypatch get_settings with a lightweight
stub so the test never touches pydantic-settings at all.
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

# Allow importing app modules from host
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set dummy DATABASE_URL so Settings can be imported without error
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"


def _stub_settings(procedural_tiers=None):
    """Return a minimal settings stub with just the fields router.py needs."""
    if procedural_tiers is None:
        procedural_tiers = ["command", "rule", "entity"]
    return SimpleNamespace(procedural_tiers=procedural_tiers)


def test_procedural_only_uses_fast():
    with patch("app.reranker.router.get_settings", return_value=_stub_settings()):
        from app.reranker.router import select_reranker_model
        assert select_reranker_model(["command", "rule"]) == "fast"
        assert select_reranker_model(["entity"]) == "fast"
        assert select_reranker_model(["command", "entity", "rule"]) == "fast"


def test_includes_skill_uses_deep():
    with patch("app.reranker.router.get_settings", return_value=_stub_settings()):
        from app.reranker.router import select_reranker_model
        assert select_reranker_model(["command", "skill"]) == "deep"


def test_includes_decision_or_episodic_uses_deep():
    with patch("app.reranker.router.get_settings", return_value=_stub_settings()):
        from app.reranker.router import select_reranker_model
        assert select_reranker_model(["decision"]) == "deep"
        assert select_reranker_model(["episodic"]) == "deep"


def test_empty_tier_set_uses_deep():
    with patch("app.reranker.router.get_settings", return_value=_stub_settings()):
        from app.reranker.router import select_reranker_model
        assert select_reranker_model([]) == "deep"
