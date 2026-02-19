import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from ..embeddings import EmbeddingProvider, EmbeddingProviderError
from ..models import Concept, ConceptScope, ConceptType

logger = logging.getLogger("agentssot.synthesis.reconciler")

_SCOPE_MAP = {"global": ConceptScope.global_, "project": ConceptScope.project, "device": ConceptScope.device}
_TYPE_MAP = {"mental_model": ConceptType.mental_model, "relationship": ConceptType.relationship, "principle": ConceptType.principle}


def reconcile_concepts(
    session: Session,
    namespace: str,
    proposals: list[dict],
    embedding_provider: EmbeddingProvider,
    embedding_provider_kind: str,
    embedding_dim: int,
) -> dict:
    """Reconcile synthesis proposals against existing concepts.

    Returns: {"new": int, "updated": int}
    """
    new_count = 0
    updated_count = 0

    for proposal in proposals:
        matched_id = proposal.get("matches_existing_id")
        is_contradiction = proposal.get("is_contradiction", False)

        concept_embedding = None
        if embedding_provider_kind != "none" and embedding_provider.is_available:
            try:
                text_to_embed = f"{proposal['title']}\n{proposal['content']}"
                concept_embedding = embedding_provider.embed_text(text_to_embed)
            except EmbeddingProviderError:
                logger.warning("failed to embed concept, storing without embedding")

        if matched_id:
            try:
                existing = session.scalar(
                    select(Concept).where(
                        and_(Concept.id == UUID(matched_id), Concept.namespace == namespace)
                    )
                )
            except (ValueError, Exception):
                existing = None

            if existing and not is_contradiction:
                existing.confidence = min(existing.confidence + 0.1, 1.0)
                new_evidence = [UUID(eid) for eid in proposal.get("evidence_item_ids", []) if eid]
                existing.evidence_ids = list(set(existing.evidence_ids or []) | set(new_evidence))
                existing.version += 1
                if proposal.get("content") and len(proposal["content"]) > len(existing.content):
                    existing.content = proposal["content"]
                    existing.title = proposal.get("title", existing.title)
                    if concept_embedding:
                        existing.embedding = concept_embedding
                updated_count += 1
                continue

            elif existing and is_contradiction:
                existing.confidence = max(existing.confidence - 0.2, 0.0)
                existing.tags = list(set(existing.tags or []) | {"superseded"})
                existing.embedding = None

                new_concept = Concept(
                    namespace=namespace,
                    type=_TYPE_MAP[proposal["type"]],
                    scope=_SCOPE_MAP.get(proposal.get("scope", "global"), ConceptScope.global_),
                    scope_ref=proposal.get("scope_ref"),
                    title=proposal["title"],
                    content=proposal["content"],
                    evidence_ids=[UUID(eid) for eid in proposal.get("evidence_item_ids", []) if eid],
                    confidence=proposal.get("confidence", 0.5),
                    version=existing.version + 1,
                    parent_id=existing.id,
                    tags=list(proposal.get("tags", [])),
                    embedding=concept_embedding,
                )
                session.add(new_concept)
                new_count += 1
                continue

        new_concept = Concept(
            namespace=namespace,
            type=_TYPE_MAP[proposal["type"]],
            scope=_SCOPE_MAP.get(proposal.get("scope", "global"), ConceptScope.global_),
            scope_ref=proposal.get("scope_ref"),
            title=proposal["title"],
            content=proposal["content"],
            evidence_ids=[UUID(eid) for eid in proposal.get("evidence_item_ids", []) if eid],
            confidence=proposal.get("confidence", 0.5),
            version=1,
            parent_id=None,
            tags=list(proposal.get("tags", [])),
            embedding=concept_embedding,
        )
        session.add(new_concept)
        new_count += 1

    session.commit()
    return {"new": new_count, "updated": updated_count}


def apply_feedback_signals(
    session: Session,
    namespace: str,
    since: datetime,
    feedback_protection_days: int = 180,
) -> tuple[set[UUID], int]:
    """Apply feedback signals to concept confidence. Returns (protected_ids, adjustments_made)."""
    from app.crud import get_feedback_summary
    from datetime import timedelta

    summary = get_feedback_summary(session, namespace, since)
    protection_cutoff = datetime.now(UTC) - timedelta(days=feedback_protection_days)
    protected_ids: set[UUID] = set()
    adjustments = 0

    for concept_id, signals in summary.items():
        concept = session.get(Concept, concept_id)
        if not concept or "superseded" in (concept.tags or []):
            continue

        delta = 0.0
        useful = min(signals["useful"], 2)  # cap +0.30/cycle
        delta += useful * 0.15
        noted = min(signals["noted"], 2)  # cap +0.10/cycle
        delta += noted * 0.05
        implicit = min(signals["implicit_recalls"], 5)  # cap +0.10/cycle
        delta += implicit * 0.02

        if delta > 0:
            concept.confidence = min(1.0, concept.confidence + delta)
            concept.updated_at = datetime.now(UTC)
            adjustments += 1

        if signals["useful"] > 0 or signals["noted"] > 0 or signals["implicit_recalls"] > 0:
            protected_ids.add(concept_id)

        if signals["wrong"] > 0 and "contested" not in (concept.tags or []):
            concept.tags = list(concept.tags or []) + ["contested"]

    # Also protect concepts with recent positive feedback in protection window
    from app.models import ConceptFeedback, FeedbackSignal
    recent_positive = (
        select(ConceptFeedback.concept_id)
        .where(ConceptFeedback.namespace == namespace)
        .where(ConceptFeedback.signal.in_([FeedbackSignal.useful, FeedbackSignal.noted]))
        .where(ConceptFeedback.created_at >= protection_cutoff)
        .distinct()
    )
    for (cid,) in session.execute(recent_positive):
        protected_ids.add(cid)

    session.flush()
    return protected_ids, adjustments


def decay_stale_concepts(
    session: Session,
    namespace: str,
    active_concept_ids: set[UUID],
    decay_rate: float = 0.02,
    min_age_days: int = 90,
    decay_floor: float = 0.15,
    protected_ids: set[UUID] | None = None,
) -> int:
    """Reduce confidence of concepts not reinforced recently. Returns count decayed.

    Only decays concepts whose updated_at is older than min_age_days,
    preventing aggressive decay of recently-created or recently-reinforced concepts.
    Concepts in protected_ids (from feedback signals) are skipped entirely.
    Concepts that reach the floor are tagged dormant instead of superseded.
    """
    from datetime import timedelta

    cutoff = datetime.now(UTC) - timedelta(days=min_age_days)
    protected = protected_ids or set()

    stmt = (
        select(Concept)
        .where(Concept.namespace == namespace)
        .where(~Concept.tags.any("superseded"))
        .where(Concept.confidence > decay_floor)
        .where(Concept.updated_at < cutoff)
    )
    decayed = 0
    for concept in session.scalars(stmt):
        if concept.id in active_concept_ids or concept.id in protected:
            continue
        concept.confidence = max(decay_floor, concept.confidence - decay_rate)
        if concept.confidence <= decay_floor:
            concept.tags = list(concept.tags or []) + ["dormant"]
        decayed += 1

    session.flush()
    return decayed


def resurrect_concept(session: Session, concept_id: UUID) -> bool:
    """Revive a dormant concept to confidence 0.5."""
    concept = session.get(Concept, concept_id)
    if not concept:
        return False
    if concept.confidence < 0.5 and "dormant" in (concept.tags or []):
        concept.confidence = 0.5
        concept.tags = [t for t in concept.tags if t != "dormant"]
        concept.updated_at = datetime.now(UTC)
        session.flush()
        return True
    return False
