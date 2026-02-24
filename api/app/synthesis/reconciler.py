import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import and_, func, select, text as sa_text
from sqlalchemy.orm import Session

from ..embeddings import EmbeddingProvider, EmbeddingProviderError
from ..models import Concept, ConceptScope, ConceptType

logger = logging.getLogger("agentssot.synthesis.reconciler")

_SCOPE_MAP = {"global": ConceptScope.global_, "project": ConceptScope.project, "device": ConceptScope.device}
_TYPE_MAP = {"mental_model": ConceptType.mental_model, "relationship": ConceptType.relationship, "principle": ConceptType.principle, "skill": ConceptType.skill}


def reconcile_concepts(
    session: Session,
    namespace: str,
    proposals: list[dict],
    embedding_provider: EmbeddingProvider,
    embedding_provider_kind: str,
    embedding_dim: int,
    agent_keys: set[str] | None = None,
) -> dict:
    """Reconcile synthesis proposals against existing concepts.

    Returns: {"new": int, "updated": int}
    """
    new_count = 0
    updated_count = 0

    for proposal in proposals:
        proposal_type = proposal.get("type", "")
        if proposal_type not in _TYPE_MAP:
            logger.warning("skipping proposal with unknown type: %s", proposal_type)
            continue

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
                existing.confidence = min(existing.confidence + 0.05, 1.0)
                new_evidence = [UUID(eid) for eid in proposal.get("evidence_item_ids", []) if eid]
                existing.evidence_ids = list(set(existing.evidence_ids or []) | set(new_evidence))
                existing.version += 1
                if proposal.get("content") and len(proposal["content"]) > len(existing.content):
                    existing.content = proposal["content"]
                    existing.title = proposal.get("title", existing.title)
                    if concept_embedding:
                        existing.embedding = concept_embedding
                    # Update skill fields if present in proposal
                    if proposal.get("trigger"):
                        existing.trigger = proposal["trigger"]
                    if proposal.get("action"):
                        existing.action = proposal["action"]
                    if proposal.get("success_hint"):
                        existing.success_hint = proposal["success_hint"]
                # Layer 5: Track confirming agents
                if agent_keys:
                    existing_agents = set(existing.confirming_agents or [])
                    new_agents = agent_keys - existing_agents
                    if new_agents:
                        existing.confirming_agents = list(existing_agents | new_agents)
                        existing.confidence = min(1.0, existing.confidence + len(new_agents) * 0.05)
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
                    trigger=proposal.get("trigger"),
                    action=proposal.get("action"),
                    success_hint=proposal.get("success_hint"),
                    embedding=concept_embedding,
                    confirming_agents=list(agent_keys) if agent_keys else [],
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
            trigger=proposal.get("trigger"),
            action=proposal.get("action"),
            success_hint=proposal.get("success_hint"),
            embedding=concept_embedding,
            confirming_agents=list(agent_keys) if agent_keys else [],
        )
        session.add(new_concept)
        new_count += 1

    session.commit()

    # Maintain concept graph: link concepts sharing evidence
    _update_concept_graph(session, namespace, proposals)

    return {"new": new_count, "updated": updated_count}


def _update_concept_graph(session: Session, namespace: str, proposals: list[dict]) -> None:
    """Incrementally update concept_links for newly reconciled concepts."""
    from itertools import combinations
    from ..models import ConceptLink

    # Collect all evidence_ids from proposals
    all_evidence: dict[UUID, set[UUID]] = {}
    for p in proposals:
        evidence = set()
        for eid in p.get("evidence_item_ids", []):
            try:
                evidence.add(UUID(eid))
            except (ValueError, TypeError):
                continue
        if not evidence:
            continue
        # Find which concept this proposal ended up as
        matched_id = p.get("matches_existing_id")
        if matched_id:
            try:
                all_evidence[UUID(matched_id)] = evidence
            except (ValueError, TypeError):
                pass

    if len(all_evidence) < 2:
        return

    # Find pairs with overlapping evidence
    concept_ids = list(all_evidence.keys())
    for a, b in combinations(concept_ids, 2):
        overlap = len(all_evidence[a] & all_evidence[b])
        if overlap == 0:
            continue
        # Normalize so concept_a < concept_b
        ca, cb = (min(a, b), max(a, b))
        weight = min(overlap * 0.3, 5.0)
        try:
            session.execute(
                sa_text("""
                    INSERT INTO concept_links (concept_a, concept_b, weight, co_occurrence_count, link_type)
                    VALUES (:a, :b, :w, :c, 'evidence_overlap')
                    ON CONFLICT (concept_a, concept_b)
                    DO UPDATE SET
                        weight = LEAST(concept_links.weight + :w_inc, 10.0),
                        co_occurrence_count = concept_links.co_occurrence_count + :c,
                        updated_at = NOW()
                """),
                {"a": ca, "b": cb, "w": weight, "c": overlap, "w_inc": weight * 0.3},
            )
        except Exception:
            logger.debug("Failed to update concept link %s <-> %s", ca, cb)
    session.flush()


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

    # Layer 5: Tag concepts with 3+ confirming agents as "consensus"
    consensus_candidates = session.scalars(
        select(Concept)
        .where(Concept.namespace == namespace)
        .where(~Concept.tags.any("superseded"))
        .where(func.array_length(Concept.confirming_agents, 1) >= 3)
    ).all()

    for concept in consensus_candidates:
        if "consensus" not in (concept.tags or []):
            concept.tags = list(concept.tags or []) + ["consensus"]
            logger.info("concept reached consensus", extra={
                "concept_id": str(concept.id),
                "agents": list(concept.confirming_agents or []),
            })

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
