"""Lifecycle helpers: supersession detection, soft-expire, promote.

Operates on KnowledgeItem-shaped objects (real ORM rows or test stubs).
"""
from __future__ import annotations

from typing import Any, Iterable


def find_supersession_candidates(new_item: Any, existing: Iterable[Any]) -> list[Any]:
    """Return existing items that are likely superseded by new_item.

    Match rule: same memory_type AND ≥1 entity_ref overlap. The classifier's
    supersedes_likely flag is informational; the actual decision is made by
    this deterministic match.
    """
    new_type = getattr(new_item, "memory_type", None)
    new_entities = set(getattr(new_item, "entity_refs", []) or [])
    if not new_type or not new_entities:
        return []
    out: list[Any] = []
    for it in existing:
        if it.id == new_item.id:
            continue
        if getattr(it, "superseded_by", None) is not None:
            continue
        if str(getattr(it, "memory_type", "")) != str(new_type):
            continue
        existing_entities = set(getattr(it, "entity_refs", []) or [])
        if existing_entities & new_entities:
            out.append(it)
    return out


def soft_expire(item: Any, reason: str) -> None:
    """Mark item expired (sets expires_at = now). Caller commits."""
    from datetime import datetime, timezone
    item.expires_at = datetime.now(timezone.utc)
    if reason:
        existing = list(item.tags or [])
        existing.append(f"expired:{reason[:40]}")
        item.tags = existing


def apply_supersession(old: Any, new: Any) -> None:
    """Mark old superseded by new. Decay old confidence, set 30d expiry."""
    from datetime import datetime, timedelta, timezone
    old.superseded_by = new.id
    old.confidence = float(old.confidence or 1.0) * 0.3
    old.expires_at = datetime.now(timezone.utc) + timedelta(days=30)
