"""Doctrine viewer endpoint (Cortex GUI P3).

Surfaces synthesis-promoted doctrine KnowledgeItems with rotation metadata so the
operator can browse what's compounding in the loadout layer.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import KnowledgeItem
from app.security import require_api_key, AuthContext, ApiRole

logger = logging.getLogger("agentssot.doctrine")

router = APIRouter(prefix="/doctrine", tags=["doctrine"])


def _today_rotation_index(count: int) -> int:
    if count <= 0:
        return -1
    return int(hashlib.sha256(date.today().isoformat().encode()).hexdigest(), 16) % count


def _age_seconds(dt: datetime | None) -> int:
    if dt is None:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int((datetime.now(timezone.utc) - dt).total_seconds())


@router.get("/list")
async def list_doctrine(
    namespace: str = Query(default="claude-shared"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    search: str = Query(default=""),
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """List doctrine-typed KnowledgeItems with rotation metadata."""
    if auth.role not in (ApiRole.writer.value, ApiRole.admin.value):
        raise HTTPException(status_code=403, detail="writer or admin role required")

    now = datetime.now(timezone.utc)
    base = [
        KnowledgeItem.namespace == namespace,
        KnowledgeItem.memory_type == "doctrine",
        KnowledgeItem.status == "active",
        KnowledgeItem.superseded_by.is_(None),
        or_(KnowledgeItem.expires_at.is_(None), KnowledgeItem.expires_at > now),
    ]
    if search:
        like = f"%{search}%"
        base.append(or_(
            KnowledgeItem.abstract.ilike(like),
            KnowledgeItem.content.ilike(like),
        ))

    total = session.scalar(select(func.count()).where(and_(*base))) or 0
    rotation_index = _today_rotation_index(total)

    stmt = (
        select(KnowledgeItem)
        .where(and_(*base))
        .order_by(KnowledgeItem.created_at)
        .offset(offset)
        .limit(limit)
    )
    rows = list(session.execute(stmt).scalars())

    items: list[dict[str, Any]] = []
    for i, ki in enumerate(rows):
        absolute_position = offset + i
        items.append({
            "id": str(ki.id),
            "namespace": ki.namespace,
            "title": ki.abstract or (ki.content[:80] if ki.content else "(no title)"),
            "summary": ki.summary,
            "content": ki.content,
            "tags": ki.tags or [],
            "source": ki.source,
            "source_ref": ki.source_ref,
            "confidence": ki.confidence,
            "strength": ki.strength,
            "loadout_priority": ki.loadout_priority,
            "recall_count": ki.recall_count,
            "last_recalled_at": ki.last_recalled_at.isoformat() if ki.last_recalled_at else None,
            "created_at": ki.created_at.isoformat() if ki.created_at else None,
            "age_seconds": _age_seconds(ki.created_at),
            "position": absolute_position,
            "is_today": absolute_position == rotation_index,
        })

    return {
        "namespace": namespace,
        "total": total,
        "offset": offset,
        "limit": limit,
        "rotation_index_today": rotation_index,
        "items": items,
    }


@router.get("/{ki_id}")
async def get_doctrine(
    ki_id: str,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Single doctrine item detail."""
    if auth.role not in (ApiRole.writer.value, ApiRole.admin.value):
        raise HTTPException(status_code=403, detail="writer or admin role required")

    ki = session.get(KnowledgeItem, ki_id)
    if not ki or ki.memory_type != "doctrine":
        raise HTTPException(status_code=404, detail="doctrine item not found")
    return {
        "id": str(ki.id),
        "namespace": ki.namespace,
        "title": ki.abstract,
        "summary": ki.summary,
        "content": ki.content,
        "tags": ki.tags or [],
        "source": ki.source,
        "source_ref": ki.source_ref,
        "confidence": ki.confidence,
        "strength": ki.strength,
        "loadout_priority": ki.loadout_priority,
        "recall_count": ki.recall_count,
        "last_recalled_at": ki.last_recalled_at.isoformat() if ki.last_recalled_at else None,
        "created_at": ki.created_at.isoformat() if ki.created_at else None,
    }
