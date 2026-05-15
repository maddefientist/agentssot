"""Adherence telemetry endpoints (Cortex GUI P5).

Surfaces the three adherence-failure classes computed from KnowledgeItem
telemetry fields. v1 is Ollama-free — pure SQL on populated counters.

Classes:
- under_fetched: high-confidence items never recalled past their incubation
- false_positives: items being recalled often but operator marks them wrong
- promotion_candidates: items recalled often, well-received, but not yet rules
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, desc, and_, or_, func
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import KnowledgeItem
from app.security import require_api_key, AuthContext, ApiRole

logger = logging.getLogger("agentssot.adherence")

router = APIRouter(prefix="/adherence", tags=["adherence"])


def _ki_summary(ki: KnowledgeItem) -> dict[str, Any]:
    return {
        "id": str(ki.id),
        "namespace": ki.namespace,
        "title": ki.abstract or (ki.content[:80] if ki.content else "(no title)"),
        "memory_type": ki.memory_type,
        "confidence": ki.confidence,
        "strength": ki.strength,
        "recall_count": ki.recall_count,
        "positive_feedback": ki.positive_feedback,
        "negative_feedback": ki.negative_feedback,
        "tags": ki.tags or [],
        "loadout_priority": ki.loadout_priority,
        "created_at": ki.created_at.isoformat() if ki.created_at else None,
        "last_recalled_at": ki.last_recalled_at.isoformat() if ki.last_recalled_at else None,
    }


@router.get("/stats")
async def adherence_stats(
    namespace: str = Query(default="claude-shared"),
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Return adherence telemetry split into three classes plus rollup counts."""
    if auth.role not in (ApiRole.writer.value, ApiRole.admin.value):
        raise HTTPException(status_code=403, detail="writer or admin role required")

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    incubation_cutoff = now - timedelta(days=7)

    base = [
        KnowledgeItem.namespace == namespace,
        KnowledgeItem.status == "active",
        KnowledgeItem.superseded_by.is_(None),
        or_(KnowledgeItem.expires_at.is_(None), KnowledgeItem.expires_at > now),
    ]

    # Under-fetched: high-confidence items past incubation that haven't been
    # recalled. Surfaces both retrieval gaps and topics the operator never asks
    # about (the GUI lets the human disambiguate).
    under_fetched_stmt = (
        select(KnowledgeItem)
        .where(and_(*base))
        .where(KnowledgeItem.confidence >= 0.8)
        .where(KnowledgeItem.recall_count == 0)
        .where(KnowledgeItem.created_at < incubation_cutoff)
        .where(
            KnowledgeItem.memory_type.in_(
                ["rule", "doctrine", "decision", "preference", "command", "entity", "fact"]
            )
        )
        .order_by(desc(KnowledgeItem.confidence), desc(KnowledgeItem.created_at))
        .limit(limit)
    )
    under_fetched = [_ki_summary(ki) for ki in session.execute(under_fetched_stmt).scalars()]

    # False positives: items being recalled but marked wrong. Net feedback
    # must be net-negative (negative >= positive).
    false_pos_stmt = (
        select(KnowledgeItem)
        .where(and_(*base))
        .where(KnowledgeItem.negative_feedback >= 1)
        .where(KnowledgeItem.negative_feedback >= KnowledgeItem.positive_feedback)
        .order_by(desc(KnowledgeItem.negative_feedback), desc(KnowledgeItem.recall_count))
        .limit(limit)
    )
    false_positives = [_ki_summary(ki) for ki in session.execute(false_pos_stmt).scalars()]

    # Promotion candidates: items being recalled often, well-received, but
    # haven't been elevated to rule status yet.
    promo_stmt = (
        select(KnowledgeItem)
        .where(and_(*base))
        .where(KnowledgeItem.recall_count >= 3)
        .where(KnowledgeItem.positive_feedback >= 2)
        .where(KnowledgeItem.negative_feedback == 0)
        .where(KnowledgeItem.memory_type != "rule")
        .where(KnowledgeItem.memory_type != "doctrine")
        .order_by(desc(KnowledgeItem.positive_feedback), desc(KnowledgeItem.recall_count))
        .limit(limit)
    )
    promotion_candidates = [_ki_summary(ki) for ki in session.execute(promo_stmt).scalars()]

    # Rollup counts (single namespace, single window)
    total_active = session.scalar(select(func.count()).where(and_(*base))) or 0
    total_recalled = session.scalar(
        select(func.count()).where(and_(*base)).where(KnowledgeItem.recall_count > 0)
    ) or 0
    coverage_pct = round(100.0 * total_recalled / total_active, 1) if total_active else 0.0

    return {
        "namespace": namespace,
        "window_days": days,
        "rollup": {
            "total_active": total_active,
            "ever_recalled": total_recalled,
            "coverage_pct": coverage_pct,
            "under_fetched_count": len(under_fetched),
            "false_positives_count": len(false_positives),
            "promotion_candidates_count": len(promotion_candidates),
        },
        "under_fetched": under_fetched,
        "false_positives": false_positives,
        "promotion_candidates": promotion_candidates,
    }
