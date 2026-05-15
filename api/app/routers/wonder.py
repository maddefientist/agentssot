"""Wonder lineage endpoints (Cortex GUI P8).

Builds the full provenance graph for any KnowledgeItem so the inspector page
can render the chain: ingest receipt -> classification -> embedding -> concepts
that used it as evidence -> doctrine derived from those concepts -> recall
stats -> status/supersession.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import KnowledgeItem, Concept
from app.security import require_api_key, AuthContext, ApiRole

logger = logging.getLogger("agentssot.wonder")

router = APIRouter(prefix="/wonder", tags=["wonder"])


def _ki_card(ki: KnowledgeItem) -> dict[str, Any]:
    return {
        "id": str(ki.id),
        "namespace": ki.namespace,
        "title": ki.abstract or (ki.content[:80] if ki.content else "(no title)"),
        "content": ki.content,
        "summary": ki.summary,
        "memory_type": ki.memory_type,
        "status": ki.status,
        "confidence": ki.confidence,
        "strength": ki.strength,
        "tags": ki.tags or [],
        "source": ki.source,
        "source_ref": ki.source_ref,
        "loadout_priority": ki.loadout_priority,
        "recall_count": ki.recall_count,
        "positive_feedback": ki.positive_feedback,
        "negative_feedback": ki.negative_feedback,
        "has_embedding": ki.embedding is not None,
        "created_at": ki.created_at.isoformat() if ki.created_at else None,
        "last_recalled_at": ki.last_recalled_at.isoformat() if ki.last_recalled_at else None,
        "expires_at": ki.expires_at.isoformat() if ki.expires_at else None,
        "superseded_by": str(ki.superseded_by) if ki.superseded_by else None,
        "source_ki_id": str(ki.source_ki_id) if ki.source_ki_id else None,
    }


def _concept_card(c: Concept) -> dict[str, Any]:
    return {
        "id": str(c.id),
        "namespace": c.namespace,
        "title": c.title,
        "content": c.content,
        "type": c.type.value if hasattr(c.type, "value") else str(c.type),
        "confidence": c.confidence,
        "version": c.version,
        "tags": c.tags or [],
        "evidence_count": len(c.evidence_ids or []),
        "confirming_agents": c.confirming_agents or [],
        "has_embedding": c.embedding is not None,
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
    }


@router.get("/lineage/{ki_id}")
async def lineage(
    ki_id: UUID = Path(...),
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Return the full provenance graph for one KnowledgeItem.

    Sections:
      ki: the item itself
      origin_concept: if this KI is itself a doctrine item, the concept it was
        promoted from
      derived_concepts: concepts that used this KI as evidence (synthesis output)
      derived_doctrine: doctrine items derived from any of those concepts
      superseded_by_ki: if status='superseded' or superseded_by is set, the
        replacement KI summary
      ingest_receipt: signals that show how this item was processed
    """
    if auth.role not in (ApiRole.writer.value, ApiRole.admin.value):
        raise HTTPException(status_code=403, detail="writer or admin role required")

    ki = session.get(KnowledgeItem, ki_id)
    if not ki:
        raise HTTPException(status_code=404, detail="knowledge item not found")

    out: dict[str, Any] = {"ki": _ki_card(ki)}

    # Origin concept: if this KI was promoted from a concept
    origin_concept = None
    if ki.source == "synthesis-promotion" and ki.source_ref:
        try:
            origin_concept = session.get(Concept, UUID(ki.source_ref))
        except (ValueError, TypeError):
            origin_concept = None
    out["origin_concept"] = _concept_card(origin_concept) if origin_concept else None

    # Concepts using this KI as evidence — uses Postgres ARRAY containment
    derived_concepts_stmt = (
        select(Concept)
        .where(Concept.evidence_ids.op("@>")(text("ARRAY[:kid]::uuid[]")))
        .params(kid=str(ki_id))
    )
    derived_concepts = list(session.execute(derived_concepts_stmt).scalars())

    # Doctrine items derived from those concepts
    derived_doctrine = []
    if derived_concepts:
        concept_ids = [str(c.id) for c in derived_concepts]
        doctrine_stmt = (
            select(KnowledgeItem)
            .where(KnowledgeItem.source == "synthesis-promotion")
            .where(KnowledgeItem.source_ref.in_(concept_ids))
            .where(KnowledgeItem.memory_type == "doctrine")
        )
        derived_doctrine = list(session.execute(doctrine_stmt).scalars())

    out["derived_concepts"] = [_concept_card(c) for c in derived_concepts]
    out["derived_doctrine"] = [_ki_card(k) for k in derived_doctrine]

    # Supersession chain
    out["superseded_by_ki"] = None
    if ki.superseded_by:
        replacement = session.get(KnowledgeItem, ki.superseded_by)
        if replacement:
            out["superseded_by_ki"] = _ki_card(replacement)

    # Ingest receipt — synthesize what we know about how this item landed
    receipt = {
        "ingested_at": ki.created_at.isoformat() if ki.created_at else None,
        "source": ki.source or "unknown",
        "classified_as": ki.memory_type or "(unclassified)",
        "embedding_generated": ki.embedding is not None,
        "verbatim_mode": getattr(ki, "verbatim", False),
        "abstract_generated": ki.abstract is not None,
        "summary_generated": ki.summary is not None,
        "initial_tags": list(ki.tags or []),
    }
    out["ingest_receipt"] = receipt

    return out
