"""Layer pre-compute: derive L0 abstract and L1 summary from content.

Reuses classifier output when present (classifier already returns abstract
and summary as part of its schema). Falls back to a head-of-content
heuristic when classifier is unavailable, so verbatim items always have
something to embed/display even if the LLM is down.
"""
from __future__ import annotations

from typing import Any


# Approximate char limits for token caps (matches classifier prompt limits).
_ABSTRACT_CHAR_CAP = 220   # ~50 tokens
_SUMMARY_CHAR_CAP = 2200   # ~500 tokens


def _truncate(text: str | None, cap: int) -> str | None:
    if not text:
        return None
    text = text.strip()
    if len(text) <= cap:
        return text
    # Cut at last full word boundary before cap
    head = text[:cap].rsplit(" ", 1)[0]
    return head + "…"


def _heuristic_abstract(content: str) -> str:
    """First sentence, truncated."""
    first = content.strip().split(". ", 1)[0]
    return _truncate(first, _ABSTRACT_CHAR_CAP) or content[:_ABSTRACT_CHAR_CAP]


def _heuristic_summary(content: str) -> str:
    """First ~500 tokens of content."""
    return _truncate(content, _SUMMARY_CHAR_CAP) or content[:_SUMMARY_CHAR_CAP]


def compute_layers(content: str, classifier_out: dict[str, Any] | None) -> dict[str, str | None]:
    """Return {abstract, summary, full_content} for an ingest payload.

    classifier_out: the dict returned by classifier.classify(). May have
    abstract/summary as None or empty strings on classifier failure.
    """
    classifier_out = classifier_out or {}
    abstract = _truncate(classifier_out.get("abstract"), _ABSTRACT_CHAR_CAP) \
        or _heuristic_abstract(content)
    summary = _truncate(classifier_out.get("summary"), _SUMMARY_CHAR_CAP) \
        or _heuristic_summary(content)
    return {
        "abstract": abstract,
        "summary": summary,
        "full_content": content,
    }
