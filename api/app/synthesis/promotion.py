"""Concept -> doctrine KnowledgeItem promotion.

After each synthesis run, eligible high-confidence Concepts of type
principle / mental_model are mirrored into doctrine-typed KnowledgeItems
so they enter the daily loadout rotation.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from ..models import Concept, ConceptType, ContentLayer, KnowledgeItem

logger = logging.getLogger("agentssot.synthesis.promotion")

ELIGIBLE_TYPES = {ConceptType.principle, ConceptType.mental_model}
CONFIDENCE_THRESHOLD = 0.8
PROMOTION_SOURCE = "synthesis-promotion"
DOCTRINE_LOADOUT_PRIORITY = 4
DOCTRINE_TAGS = ["doctrine-promoted", "auto"]


def is_eligible(concept: Concept) -> bool:
    """Pure check, safe to unit-test."""
    if concept.confidence < CONFIDENCE_THRESHOLD:
        return False
    if concept.type not in ELIGIBLE_TYPES:
        return False
    if "superseded" in (concept.tags or []):
        return False
    if concept.embedding is None:
        return False
    return True


def _build_ki(concept: Concept) -> KnowledgeItem:
    tags = sorted(set((concept.tags or []) + DOCTRINE_TAGS))
    return KnowledgeItem(
        namespace=concept.namespace,
        content=concept.content,
        memory_type="doctrine",
        abstract=concept.title,
        summary=None,
        layer=ContentLayer.full,
        loadout_priority=DOCTRINE_LOADOUT_PRIORITY,
        confidence=1.0,
        strength=1.0,
        tags=tags,
        source=PROMOTION_SOURCE,
        source_ref=str(concept.id),
        embedding=list(concept.embedding) if concept.embedding is not None else None,
        verbatim=False,
    )


def promote_concepts_to_doctrine(
    session: Session,
    namespace: str,
    embedding_provider=None,  # accepted for symmetry; embedding copied from concept
) -> dict:
    """Promote eligible Concepts in `namespace` to doctrine KnowledgeItems.

    Returns: {"promoted": int, "superseded": int, "skipped": int}
    """
    stats = {"promoted": 0, "superseded": 0, "skipped": 0}

    concepts = session.scalars(
        select(Concept).where(Concept.namespace == namespace)
    ).all()

    for concept in concepts:
        if not is_eligible(concept):
            continue

        existing = session.scalar(
            select(KnowledgeItem).where(
                and_(
                    KnowledgeItem.namespace == namespace,
                    KnowledgeItem.source == PROMOTION_SOURCE,
                    KnowledgeItem.source_ref == str(concept.id),
                    KnowledgeItem.status == "active",
                    KnowledgeItem.superseded_by.is_(None),
                )
            )
        )

        if existing is not None and existing.content == concept.content:
            stats["skipped"] += 1
            continue

        new_ki = _build_ki(concept)
        session.add(new_ki)
        session.flush()  # materialize new_ki.id

        if existing is not None:
            existing.status = "superseded"
            existing.superseded_by = new_ki.id
            stats["superseded"] += 1
            logger.info(
                "promoted concept to doctrine",
                extra={
                    "concept_id": str(concept.id),
                    "ki_id": str(new_ki.id),
                    "namespace": namespace,
                    "action": "superseded",
                },
            )
        else:
            stats["promoted"] += 1
            logger.info(
                "promoted concept to doctrine",
                extra={
                    "concept_id": str(concept.id),
                    "ki_id": str(new_ki.id),
                    "namespace": namespace,
                    "action": "new",
                },
            )

    return stats
