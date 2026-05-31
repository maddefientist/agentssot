"""Tests for query-time extractive sentence trimming (lift #3)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"

from app.sentence_trim import trim_to_query, trim_recall_items


SNIPPET = (
    "The deployment uses Kubernetes on GKE. "
    "We decided to use JWT with 15-minute access tokens for the auth service. "
    "The refresh token bug was caused by a missing httpOnly flag on the cookie. "
    "Lunch options near the office include three taco places. "
    "Postgres runs on a dedicated node pool."
)


def test_trims_to_relevant_sentences():
    out = trim_to_query(SNIPPET, "JWT refresh token cookie bug", max_sentences=2)
    assert "JWT" in out
    assert "refresh token" in out
    assert "taco" not in out  # irrelevant sentence dropped
    assert len(out) < len(SNIPPET)


def test_marks_elision():
    out = trim_to_query(SNIPPET, "JWT refresh token", max_sentences=2)
    assert "[…]" in out


def test_failopen_when_no_overlap():
    out = trim_to_query(SNIPPET, "quantum chromodynamics lattice", max_sentences=2)
    assert out == SNIPPET  # nothing matched → return original, never blank


def test_no_query_returns_original():
    assert trim_to_query(SNIPPET, None) == SNIPPET
    assert trim_to_query(SNIPPET, "   ") == SNIPPET


def test_short_snippet_untouched():
    short = "Just one sentence here."
    assert trim_to_query(short, "sentence", max_sentences=4) == short


def test_preserves_document_order():
    out = trim_to_query(SNIPPET, "Kubernetes Postgres node pool", max_sentences=2)
    # Kubernetes sentence precedes Postgres sentence in the source.
    assert out.index("Kubernetes") < out.index("Postgres")


def test_trim_recall_items_flags():
    items = [{"id": "a", "snippet": SNIPPET}]
    n = trim_recall_items(items, "JWT refresh token", max_sentences=2)
    assert n == 1
    assert items[0]["trimmed"] is True
    assert len(items[0]["snippet"]) < len(SNIPPET)
