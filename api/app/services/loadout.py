"""Loadout assembly: cwd→entity resolution, tier-aware fetch, budget pack.

Loadout = the cwd-aware push context that gets pre-loaded at SessionStart
or fetched explicitly via /loadout. Cheap, deterministic, prompt-cacheable.
"""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


# Approximate tokens per character. Matches OpenAI's tiktoken on English text
# closely enough for budget packing without pulling tiktoken into the runtime.
_TOKEN_PER_CHAR = 0.27


def estimate_tokens(text: str) -> int:
    """Rough token count; over-estimates slightly so we stay under budget."""
    return int(len(text) * _TOKEN_PER_CHAR) + 1


def resolve_cwd_entities(cwd: str, entities: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return entities whose cwd_hints prefix-match the given cwd.

    Match rule: cwd starts with the hint OR cwd contains the hint as a path
    segment (e.g. cwd '/home/hari/.claude' matches hint '/.claude').
    """
    matched: list[dict[str, Any]] = []
    cwd_norm = cwd.rstrip("/")
    for ent in entities:
        for hint in ent.get("cwd_hints", []):
            h = hint.rstrip("/")
            if not h:
                continue
            if cwd_norm == h or cwd_norm.startswith(h + "/") or h in cwd_norm:
                matched.append(ent)
                break
    return matched


def fetch_loadout_candidates(
    session: Session, namespace: str, entity_ids: list[str], device_id: str | None
) -> list[KnowledgeItem]:
    """Pull active items linked to any of the given entity_ids in the namespace.

    Active = not superseded, not expired, confidence >= 0.5.
    Includes rules unconditionally (rules are global to the namespace).
    """
    from sqlalchemy import select, or_, and_, func, cast
    from sqlalchemy.dialects.postgresql import ARRAY, TEXT
    # PostgreSQL JSONB ?| operator — safe bindparam expansion, no string concat
    from datetime import datetime, timezone
    from app.models import KnowledgeItem, MemoryType

    now = datetime.now(timezone.utc)
    base_filters = [
        KnowledgeItem.namespace == namespace,
        KnowledgeItem.confidence >= 0.5,
        KnowledgeItem.superseded_by.is_(None),
        or_(KnowledgeItem.expires_at.is_(None), KnowledgeItem.expires_at > now),
    ]

    # Rules: load all in this namespace (global rules)
    rules_stmt = select(KnowledgeItem).where(
        and_(*base_filters, KnowledgeItem.memory_type == MemoryType.rule)
    ).order_by(KnowledgeItem.loadout_priority.desc())
    rules = list(session.execute(rules_stmt).scalars())

    # Other tiers: filter by entity_refs intersection. JSONB containment via ?| operator.
    if not entity_ids:
        return rules

    # PostgreSQL JSONB ?| operator — matches when any element in the JSONB array
    # equals any of the supplied entity_ids. Safe bindparam expansion, no string concat.
    entity_filter = func.jsonb_exists_any(KnowledgeItem.entity_refs, cast(entity_ids, ARRAY(TEXT)))
    others_stmt = select(KnowledgeItem).where(
        and_(*base_filters,
             KnowledgeItem.memory_type.in_([
                 MemoryType.command, MemoryType.entity,
                 MemoryType.skill, MemoryType.decision,
             ]),
             entity_filter)
    ).order_by(KnowledgeItem.loadout_priority.desc())
    others = list(session.execute(others_stmt).scalars())
    return rules + others


def pack_loadout(
    items: list[dict[str, Any]], token_budget: int
) -> tuple[list[dict[str, Any]], int, int]:
    """Greedy pack by priority desc until token budget is exhausted.

    Returns (packed_items, overflow_count, tokens_used).
    """
    if not items:
        return [], 0, 0
    sorted_items = sorted(items, key=lambda x: -int(x.get("priority", 0)))
    packed: list[dict[str, Any]] = []
    used = 0
    for it in sorted_items:
        cost = estimate_tokens(f"[{it['memory_type']}] {it.get('title','')} — {it['abstract']}")
        if used + cost > token_budget:
            continue
        packed.append(it)
        used += cost
    overflow = len(sorted_items) - len(packed)
    return packed, overflow, used


def loadout_cache_key(cwd: str, device_id: str | None, item_ids: list[str]) -> str:
    """sha256 of (cwd, device, sorted item_ids) — stable across sessions."""
    payload = json.dumps(
        {"cwd": cwd, "device": device_id or "", "ids": sorted(item_ids)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
