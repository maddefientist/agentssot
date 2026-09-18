"""Pure-function unit tests for the eligibility/rendering helpers shared by
the legacy (/recall) and bucketed (/api/v1/knowledge/recall) retrieval paths.

crud.knowledge_active_status_clause() is the single source of eligibility
policy for "is this item disputed" across both paths (S1: converge
eligibility instead of letting adapters implement separate rules).
crud.resolve_bounded_abstract() is the shared blank-abstract fallback.
"""
import os

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@127.0.0.1/test")

from app import crud


def test_active_status_clause_excludes_flagged_only():
    where = str(crud.knowledge_active_status_clause())
    assert "status" in where


def test_resolve_bounded_abstract_prefers_existing_abstract():
    assert crud.resolve_bounded_abstract("real abstract", "summary", "content") == "real abstract"


def test_resolve_bounded_abstract_falls_back_to_summary():
    assert crud.resolve_bounded_abstract(None, "a summary", "content") == "a summary"


def test_resolve_bounded_abstract_falls_back_to_bounded_content():
    long_content = "x" * 500
    result = crud.resolve_bounded_abstract(None, None, long_content, max_chars=160)
    assert result is not None
    assert len(result) <= 160
    assert result != long_content


def test_resolve_bounded_abstract_returns_none_when_nothing_to_show():
    assert crud.resolve_bounded_abstract(None, None, None) is None


def test_resolve_bounded_abstract_empty_string_abstract_treated_as_blank():
    """An empty-string abstract (falsy) must still fall back, not render blank."""
    assert crud.resolve_bounded_abstract("", None, "content here") == "content here"
