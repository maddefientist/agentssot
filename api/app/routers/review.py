"""Synthesis Review v2 endpoints (Cortex GUI P6).

The existing /api/v1/knowledge/admin/review-queue handles KI-level review
(dup/supersede/low_conf/contradiction). This router adds the *concept*-level
human gate for synthesis-proposed concepts in the uncertainty band [0.5, 0.8) —
too uncertain to auto-promote to doctrine, too plausible to drop.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy import select, desc, and_
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import Concept
from app.security import require_api_key, AuthContext, ApiRole

logger = logging.getLogger("agentssot.review")

router = APIRouter(prefix="/review", tags=["review"])

BAND_LOW = 0.5
BAND_HIGH = 0.8


def _concept_summary(c: Concept) -> dict[str, Any]:
    return {
        "id": str(c.id),
        "namespace": c.namespace,
        "title": c.title,
        "content": c.content,
        "type": c.type.value if hasattr(c.type, "value") else str(c.type),
        "scope": c.scope.value if hasattr(c.scope, "value") else str(c.scope),
        "confidence": c.confidence,
        "tags": c.tags or [],
        "version": c.version,
        "evidence_count": len(c.evidence_ids or []),
        "confirming_agents": c.confirming_agents or [],
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
    }


@router.get("/concepts")
async def list_band(
    namespace: str = Query(default="claude-shared"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Concepts in the uncertainty band, sorted by confidence desc then recency."""
    if auth.role not in (ApiRole.writer.value, ApiRole.admin.value):
        raise HTTPException(status_code=403, detail="writer or admin role required")

    base = [
        Concept.namespace == namespace,
        Concept.confidence >= BAND_LOW,
        Concept.confidence < BAND_HIGH,
        ~Concept.tags.any("superseded"),
        ~Concept.tags.any("dismissed-by-operator"),
    ]
    stmt = (
        select(Concept)
        .where(and_(*base))
        .order_by(desc(Concept.confidence), desc(Concept.updated_at))
        .offset(offset)
        .limit(limit)
    )
    rows = list(session.execute(stmt).scalars())
    return {
        "namespace": namespace,
        "band": [BAND_LOW, BAND_HIGH],
        "count": len(rows),
        "items": [_concept_summary(c) for c in rows],
    }


@router.post("/concepts/{concept_id}/{action}")
async def gate_concept(
    concept_id: str = Path(...),
    action: str = Path(..., pattern="^(approve|demote|dismiss)$"),
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Human gate action on a band concept.

    approve: confidence -> 0.85 (above band, eligible for doctrine promotion on
             next synthesis-loop run), tag 'operator-approved'.
    demote:  confidence -> 0.4 (below band; concept stays around for re-evaluation
             but won't surface in the queue), tag 'operator-demoted'.
    dismiss: confidence unchanged, tag 'dismissed-by-operator' (excluded from
             future queue surfaces). Reversible by removing the tag.
    """
    if auth.role not in (ApiRole.writer.value, ApiRole.admin.value):
        raise HTTPException(status_code=403, detail="writer or admin role required")

    c = session.get(Concept, concept_id)
    if not c:
        raise HTTPException(status_code=404, detail="concept not found")

    new_tags = list(c.tags or [])
    if action == "approve":
        c.confidence = max(c.confidence, 0.85)
        if "operator-approved" not in new_tags:
            new_tags.append("operator-approved")
    elif action == "demote":
        c.confidence = min(c.confidence, 0.4)
        if "operator-demoted" not in new_tags:
            new_tags.append("operator-demoted")
    elif action == "dismiss":
        if "dismissed-by-operator" not in new_tags:
            new_tags.append("dismissed-by-operator")
    c.tags = sorted(set(new_tags))
    c.updated_at = datetime.now(timezone.utc)
    session.commit()
    logger.info("concept gated", extra={
        "concept_id": str(c.id), "action": action, "new_confidence": c.confidence,
        "namespace": c.namespace,
    })
    return {"id": str(c.id), "action": action, "confidence": c.confidence, "tags": c.tags}
