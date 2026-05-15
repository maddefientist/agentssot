"""Signals inbox endpoints (Cortex GUI P2).

Merged feed of operator-relevant signals: manual leads, adherence misses,
recurring errors, pattern detections, wonder queue digest.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, desc, or_, func
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import KnowledgeItem, Concept
from app.security import require_api_key, AuthContext, ApiRole

logger = logging.getLogger("agentssot.signals")

router = APIRouter(prefix="/signals", tags=["signals"])


def _age_seconds(dt: datetime | None) -> int:
    if dt is None:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int((datetime.now(timezone.utc) - dt).total_seconds())


def _ki_to_signal(ki: KnowledgeItem, kind: str) -> dict[str, Any]:
    return {
        "type": kind,
        "id": str(ki.id),
        "title": ki.abstract or (ki.content[:120] if ki.content else "(no content)"),
        "source": ki.source or "unknown",
        "age_seconds": _age_seconds(ki.created_at),
        "namespace": ki.namespace,
        "payload": {
            "content": ki.content,
            "tags": ki.tags or [],
            "memory_type": ki.memory_type,
        },
        "actions": ["dismiss", "promote-to-rule", "promote-to-doctrine", "send-to-wonder"],
    }


def _concept_to_signal(c: Concept) -> dict[str, Any]:
    return {
        "type": "pattern",
        "id": str(c.id),
        "title": c.title,
        "source": "synthesis",
        "age_seconds": _age_seconds(c.updated_at),
        "namespace": c.namespace,
        "payload": {
            "content": c.content,
            "confidence": c.confidence,
            "tags": c.tags or [],
            "concept_type": c.type.value if hasattr(c.type, "value") else str(c.type),
        },
        "actions": ["dismiss", "promote-to-doctrine"],
    }


@router.get("/feed")
async def list_signals(
    namespace: str = Query(default="claude-shared"),
    types: str = Query(default="manual,wonder,error,pattern,adherence"),
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Return merged signals feed."""
    if auth.role not in (ApiRole.writer.value, ApiRole.admin.value):
        raise HTTPException(status_code=403, detail="writer or admin role required")

    requested = {t.strip() for t in types.split(",") if t.strip()}
    out: list[dict[str, Any]] = []

    base_ki = (
        select(KnowledgeItem)
        .where(KnowledgeItem.namespace == namespace)
        .where(KnowledgeItem.status == "active")
    )

    if "manual" in requested:
        q = base_ki.where(KnowledgeItem.tags.any("manual-lead")).order_by(
            desc(KnowledgeItem.created_at)
        ).limit(limit)
        for ki in session.execute(q).scalars():
            out.append(_ki_to_signal(ki, "manual"))

    if "wonder" in requested:
        q = (
            base_ki.where(KnowledgeItem.memory_type == "wonder")
            .where(~KnowledgeItem.tags.any("manual-lead"))
            .order_by(desc(KnowledgeItem.created_at))
            .limit(limit)
        )
        for ki in session.execute(q).scalars():
            out.append(_ki_to_signal(ki, "wonder"))

    if "error" in requested:
        q = (
            base_ki.where(
                or_(
                    KnowledgeItem.tags.any("error"),
                    KnowledgeItem.tags.any("incident"),
                )
            )
            .order_by(desc(KnowledgeItem.created_at))
            .limit(limit)
        )
        for ki in session.execute(q).scalars():
            out.append(_ki_to_signal(ki, "error"))

    if "pattern" in requested:
        q = (
            select(Concept)
            .where(Concept.namespace == namespace)
            .where(
                or_(
                    Concept.tags.any("supersession-surge"),
                    Concept.tags.any("confidence-drop"),
                    Concept.tags.any("contradiction"),
                )
            )
            .order_by(desc(Concept.updated_at))
            .limit(limit)
        )
        for c in session.execute(q).scalars():
            out.append(_concept_to_signal(c))

    # adherence: deferred to P4/P5 — placeholder returns nothing

    out.sort(key=lambda s: s["age_seconds"])
    return {"namespace": namespace, "count": len(out), "items": out[:limit]}


class ManualLeadIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    note: str = Field(..., min_length=1)
    namespace: str = Field(default="claude-shared")
    tags: list[str] = Field(default_factory=list)


@router.post("/manual")
async def add_manual_lead(
    data: ManualLeadIn,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Operator adds a lead. Stored as KnowledgeItem with memory_type='wonder',
    tagged 'manual-lead'. Picked up by the merged feed as type='manual'."""
    if auth.role not in (ApiRole.writer.value, ApiRole.admin.value):
        raise HTTPException(status_code=403, detail="writer or admin role required")

    tags = sorted(set((data.tags or []) + ["manual-lead", "auto"]))
    ki = KnowledgeItem(
        namespace=data.namespace,
        content=data.note,
        abstract=data.title,
        memory_type="wonder",
        source="signals-manual",
        tags=tags,
        confidence=1.0,
        strength=1.0,
    )
    session.add(ki)
    session.commit()
    session.refresh(ki)
    logger.info("manual lead added", extra={"ki_id": str(ki.id), "namespace": data.namespace})
    return {"id": str(ki.id), "namespace": ki.namespace, "tags": ki.tags}


class SignalActionIn(BaseModel):
    action: str = Field(..., pattern="^(dismiss|promote-to-rule|promote-to-doctrine|send-to-wonder)$")
    kind: str = Field(default="knowledge", pattern="^(knowledge|concept)$")


@router.post("/{signal_id}/action")
async def signal_action(
    signal_id: UUID,
    data: SignalActionIn,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Dispatch chip action on a signal item.

    For knowledge-kind signals:
      dismiss             -> mark item status='dismissed'
      promote-to-rule     -> memory_type='rule', loadout_priority=5
      promote-to-doctrine -> memory_type='doctrine', loadout_priority=4
      send-to-wonder      -> memory_type='wonder', add tag 'wonder-queue'

    For concept-kind signals:
      dismiss             -> add tag 'dismissed-by-operator'
      promote-to-doctrine -> defer (nightly bridge will pick up)
    """
    if auth.role not in (ApiRole.writer.value, ApiRole.admin.value):
        raise HTTPException(status_code=403, detail="writer or admin role required")

    if data.kind == "knowledge":
        ki = session.get(KnowledgeItem, signal_id)
        if not ki:
            raise HTTPException(status_code=404, detail="knowledge item not found")
        if data.action == "dismiss":
            ki.status = "dismissed"
        elif data.action == "promote-to-rule":
            ki.memory_type = "rule"
            ki.tags = sorted(set((ki.tags or []) + ["promoted-by-operator"]))
        elif data.action == "promote-to-doctrine":
            ki.memory_type = "doctrine"
            ki.tags = sorted(set((ki.tags or []) + ["promoted-by-operator"]))
        elif data.action == "send-to-wonder":
            ki.memory_type = "wonder"
            ki.tags = sorted(set((ki.tags or []) + ["wonder-queue"]))
        session.commit()
        logger.info("signal action applied", extra={
            "ki_id": str(ki.id), "action": data.action, "namespace": ki.namespace
        })
        return {"id": str(ki.id), "action": data.action, "status": ki.status, "memory_type": ki.memory_type}

    # concept-kind
    concept = session.get(Concept, signal_id)
    if not concept:
        raise HTTPException(status_code=404, detail="concept not found")
    if data.action == "dismiss":
        concept.tags = sorted(set((concept.tags or []) + ["dismissed-by-operator"]))
    elif data.action == "promote-to-doctrine":
        concept.tags = sorted(set((concept.tags or []) + ["promote-now"]))
    else:
        raise HTTPException(status_code=400, detail=f"action {data.action} not supported for concepts")
    session.commit()
    logger.info("concept signal action applied", extra={
        "concept_id": str(concept.id), "action": data.action
    })
    return {"id": str(concept.id), "action": data.action, "tags": concept.tags}
