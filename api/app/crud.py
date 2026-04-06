import logging
import math
from datetime import UTC, datetime
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import and_, case, delete as sa_delete, func, or_, select, text as sa_text, update
from sqlalchemy.orm import Session

from .chunking import chunk_text_semantic
from .embeddings import EmbeddingProvider, EmbeddingProviderError
from .llm import LLMProvider, LLMProviderError
from .reranker import RerankerProvider, RerankerProviderError
from .models import AgentProfile, ApiKey, ApiRole, Concept, ConceptFeedback, ConceptScope, ConceptType, EnrollmentToken, Entity, EntityType, Event, EventType, FeedbackSignal, KnowledgeItem, MemoryType, Namespace, RecallEvent, Requirement
from .schemas import IngestRequest, RecallRequest
from .secret_scanner import scan_ingest_payload, scan_text
from .security import generate_api_key, generate_enrollment_token, hash_api_key, verify_api_key

logger = logging.getLogger("agentssot.crud")


def _clip(text: str | None, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _safe_float(value, fallback: float = 1.0) -> float:
    """Convert scores to finite floats to avoid NaN/Inf leaking into JSON responses."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return fallback
    return f if math.isfinite(f) else fallback


def _validate_embedding_dim(embedding: list[float] | None, expected_dim: int) -> None:
    if embedding is None:
        return
    if len(embedding) != expected_dim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Embedding dimension mismatch. Expected {expected_dim}, got {len(embedding)}",
        )


def ensure_namespace_exists(session: Session, namespace: str) -> None:
    exists = session.scalar(select(Namespace.name).where(Namespace.name == namespace))
    if not exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Namespace '{namespace}' not found")


def get_entity_by_slug(session: Session, namespace: str, slug: str) -> Entity | None:
    return session.scalar(select(Entity).where(and_(Entity.namespace == namespace, Entity.slug == slug)))


def _resolve_entity_id(session: Session, namespace: str, slug: str | None, field_name: str) -> UUID | None:
    if not slug:
        return None
    entity = get_entity_by_slug(session, namespace, slug)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{field_name} '{slug}' not found in namespace '{namespace}'",
        )
    return entity.id


def _maybe_embed_text(
    provider: EmbeddingProvider,
    text: str,
    provider_kind: str,
) -> list[float] | None:
    if not text.strip():
        return None

    if provider_kind == "none":
        return None

    if not provider.is_available:
        raise EmbeddingProviderError(provider.unavailable_reason or "Embedding provider unavailable")

    return provider.embed_text(text)


def ingest_batch(session: Session, payload: IngestRequest, embedding_provider: EmbeddingProvider, settings) -> dict[str, int]:
    ensure_namespace_exists(session, payload.namespace)

    # ── Secret scanning gate ──────────────────────────────────────
    if getattr(settings, "ingest_secret_scanning", True):
        rejections = scan_ingest_payload(payload)
        if rejections:
            detail = "Ingest rejected: content contains potential secrets.\n" + "\n".join(rejections)
            logger.warning("secret scan rejected %d item(s) in namespace=%s", len(rejections), payload.namespace)
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)

    counts = {
        "entities": 0,
        "requirements": 0,
        "knowledge_items": 0,
        "events": 0,
    }

    try:
        for item in payload.entities:
            existing = get_entity_by_slug(session, payload.namespace, item.slug)
            if existing:
                existing.type = EntityType(item.type)
                existing.name = item.name
                existing.description = item.description
                existing.meta = item.metadata
            else:
                session.add(
                    Entity(
                        namespace=payload.namespace,
                        slug=item.slug,
                        type=EntityType(item.type),
                        name=item.name,
                        description=item.description,
                        meta=item.metadata,
                    )
                )
            counts["entities"] += 1

        session.flush()

        for item in payload.requirements:
            project_id = _resolve_entity_id(session, payload.namespace, item.project_slug, "project_slug")
            owner_id = _resolve_entity_id(session, payload.namespace, item.owner_entity_slug, "owner_entity_slug")

            embedding = item.embedding
            _validate_embedding_dim(embedding, settings.embedding_dim)
            if embedding is None:
                source_text = "\n".join(
                    part for part in [item.title, item.body or "", item.context_snippet or ""] if part
                )
                embedding = _maybe_embed_text(embedding_provider, source_text, settings.embedding_provider)
                _validate_embedding_dim(embedding, settings.embedding_dim)

            session.add(
                Requirement(
                    namespace=payload.namespace,
                    project_id=project_id,
                    owner_entity_id=owner_id,
                    title=item.title,
                    body=item.body,
                    priority=item.priority,
                    status=item.status,
                    context_snippet=item.context_snippet,
                    tags=item.tags,
                    embedding=embedding,
                )
            )
            counts["requirements"] += 1

        for item in payload.knowledge_items:
            project_id = _resolve_entity_id(session, payload.namespace, item.project_slug, "project_slug")
            entity_id = _resolve_entity_id(session, payload.namespace, item.entity_slug, "entity_slug")

            chunks = chunk_text_semantic(item.content, max_chars=800)
            if not chunks:
                continue

            _validate_embedding_dim(item.embedding, settings.embedding_dim)

            for chunk in chunks:
                # Pre-ingest dedup: skip if identical content already exists in this namespace
                exists = session.scalar(
                    select(func.count()).select_from(KnowledgeItem).where(
                        and_(KnowledgeItem.namespace == payload.namespace, KnowledgeItem.content == chunk)
                    )
                )
                if exists:
                    logger.debug("skipping duplicate knowledge chunk (namespace=%s)", payload.namespace)
                    continue

                chunk_embedding = item.embedding
                if chunk_embedding is None:
                    chunk_embedding = _maybe_embed_text(embedding_provider, chunk, settings.embedding_provider)
                    _validate_embedding_dim(chunk_embedding, settings.embedding_dim)

                ki_kwargs = dict(
                    namespace=payload.namespace,
                    project_id=project_id,
                    entity_id=entity_id,
                    content=chunk,
                    source=item.source,
                    source_ref=item.source_ref,
                    tags=item.tags,
                    embedding=chunk_embedding,
                )
                # Pass through typed memory fields if provided
                if item.memory_type is not None:
                    ki_kwargs["memory_type"] = item.memory_type
                if item.extraction_source is not None:
                    ki_kwargs["extraction_source"] = item.extraction_source
                if item.extraction_cursor_id is not None:
                    ki_kwargs["extraction_cursor_id"] = item.extraction_cursor_id
                session.add(KnowledgeItem(**ki_kwargs))
                counts["knowledge_items"] += 1

        for item in payload.events:
            project_id = _resolve_entity_id(session, payload.namespace, item.project_slug, "project_slug")
            agent_id = _resolve_entity_id(session, payload.namespace, item.agent_slug, "agent_slug")

            embedding = item.embedding
            _validate_embedding_dim(embedding, settings.embedding_dim)
            if embedding is None:
                source_text = "\n".join(
                    part for part in [item.title, item.body or "", item.context_snippet or ""] if part
                )
                embedding = _maybe_embed_text(embedding_provider, source_text, settings.embedding_provider)
                _validate_embedding_dim(embedding, settings.embedding_dim)

            session.add(
                Event(
                    namespace=payload.namespace,
                    project_id=project_id,
                    agent_id=agent_id,
                    type=EventType(item.type),
                    title=item.title,
                    body=item.body,
                    context_snippet=item.context_snippet,
                    session_id=item.session_id,
                    tags=item.tags,
                    embedding=embedding,
                )
            )
            counts["events"] += 1

        session.commit()
    except HTTPException:
        session.rollback()
        raise
    except EmbeddingProviderError as exc:
        session.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        session.rollback()
        logger.exception("ingest failed")
        raise

    return counts


def query_records(
    session: Session,
    namespace: str,
    q: str,
    project_slug: str | None,
    entity_slug: str | None,
    limit: int,
    max_snippet_chars: int,
) -> list[dict]:
    ensure_namespace_exists(session, namespace)

    project_id = _resolve_entity_id(session, namespace, project_slug, "project_slug") if project_slug else None
    entity_id = _resolve_entity_id(session, namespace, entity_slug, "entity_slug") if entity_slug else None

    rows: list[dict] = []
    query_limit = min(max(limit, 1), 100)
    text_filter = f"%{q}%" if q else None

    entity_stmt = select(Entity).where(Entity.namespace == namespace)
    if text_filter:
        entity_stmt = entity_stmt.where(or_(Entity.name.ilike(text_filter), Entity.description.ilike(text_filter)))
    if entity_slug:
        entity_stmt = entity_stmt.where(Entity.slug == entity_slug)
    entity_stmt = entity_stmt.order_by(Entity.updated_at.desc()).limit(query_limit)
    for entity in session.scalars(entity_stmt):
        rows.append(
            {
                "id": str(entity.id),
                "kind": "entity",
                "title": entity.name,
                "snippet": _clip(entity.description or "", max_snippet_chars),
                "tags": [entity.type.value],
                "created_at": entity.created_at,
            }
        )

    req_stmt = select(Requirement).where(Requirement.namespace == namespace)
    if text_filter:
        req_stmt = req_stmt.where(or_(Requirement.title.ilike(text_filter), Requirement.body.ilike(text_filter)))
    if project_id:
        req_stmt = req_stmt.where(Requirement.project_id == project_id)
    if entity_id:
        req_stmt = req_stmt.where(Requirement.owner_entity_id == entity_id)
    req_stmt = req_stmt.order_by(Requirement.updated_at.desc()).limit(query_limit)
    for req in session.scalars(req_stmt):
        rows.append(
            {
                "id": str(req.id),
                "kind": "requirement",
                "title": req.title,
                "snippet": _clip(req.context_snippet or req.body or "", max_snippet_chars),
                "tags": list(req.tags or []),
                "created_at": req.created_at,
            }
        )

    ki_stmt = select(KnowledgeItem).where(KnowledgeItem.namespace == namespace)
    if text_filter:
        ki_stmt = ki_stmt.where(
            or_(KnowledgeItem.content.ilike(text_filter), KnowledgeItem.source.ilike(text_filter), KnowledgeItem.source_ref.ilike(text_filter))
        )
    if project_id:
        ki_stmt = ki_stmt.where(KnowledgeItem.project_id == project_id)
    if entity_id:
        ki_stmt = ki_stmt.where(KnowledgeItem.entity_id == entity_id)
    ki_stmt = ki_stmt.order_by(KnowledgeItem.created_at.desc()).limit(query_limit)
    for item in session.scalars(ki_stmt):
        rows.append(
            {
                "id": str(item.id),
                "kind": "knowledge_item",
                "title": item.source or "knowledge_item",
                "snippet": _clip(item.content, max_snippet_chars),
                "tags": list(item.tags or []),
                "created_at": item.created_at,
            }
        )

    ev_stmt = select(Event).where(Event.namespace == namespace)
    if text_filter:
        ev_stmt = ev_stmt.where(or_(Event.title.ilike(text_filter), Event.body.ilike(text_filter)))
    if project_id:
        ev_stmt = ev_stmt.where(Event.project_id == project_id)
    if entity_id:
        ev_stmt = ev_stmt.where(Event.agent_id == entity_id)
    ev_stmt = ev_stmt.order_by(Event.created_at.desc()).limit(query_limit)
    for event in session.scalars(ev_stmt):
        rows.append(
            {
                "id": str(event.id),
                "kind": "event",
                "title": event.title,
                "snippet": _clip(event.body or event.context_snippet or "", max_snippet_chars),
                "tags": list(event.tags or []),
                "created_at": event.created_at,
            }
        )

    rows.sort(key=lambda r: (r.get("created_at") or datetime.min.replace(tzinfo=UTC)), reverse=True)
    return rows[:query_limit]


def _recall_knowledge_weighted(
    session: Session,
    namespace: str,
    query_embedding: list[float],
    candidate_k: int,
    project_id: UUID | None,
    entity_id: UUID | None,
    settings,
    memory_type: str | None = None,
    max_staleness: float | None = None,
) -> list[dict]:
    """Recall knowledge items with weighted scoring: similarity + strength + recency.

    Optional filters (only applied when typed_memory_enabled is True in settings):
    - memory_type: filter to a specific MemoryType value
    - max_staleness: exclude items with staleness_score above this threshold
    """
    similarity = (1.0 - KnowledgeItem.embedding.cosine_distance(query_embedding)).label("similarity")
    # Normalized strength: cap at 5.0, divide by 5
    norm_strength = func.least(func.coalesce(KnowledgeItem.strength, 1.0), 5.0) / 5.0
    # Recency bonus based on last recall time
    recency = case(
        (KnowledgeItem.last_recalled_at > func.now() - sa_text("INTERVAL '1 day'"), 1.0),
        (KnowledgeItem.last_recalled_at > func.now() - sa_text("INTERVAL '7 days'"), 0.7),
        (KnowledgeItem.last_recalled_at > func.now() - sa_text("INTERVAL '30 days'"), 0.4),
        else_=0.1,
    )
    weighted_score = (similarity * 0.6 + norm_strength * 0.3 + recency * 0.1).label("weighted_score")
    # Use cosine_distance for ORDER BY (pgvector operator) but weighted_score for final ranking
    cosine_dist = KnowledgeItem.embedding.cosine_distance(query_embedding).label("score")

    stmt = (
        select(KnowledgeItem, cosine_dist, weighted_score)
        .where(KnowledgeItem.namespace == namespace)
        .where(KnowledgeItem.embedding.is_not(None))
        .where(or_(KnowledgeItem.status == "active", KnowledgeItem.status.is_(None)))
        .order_by(weighted_score.desc())
        .limit(candidate_k)
    )
    if project_id:
        stmt = stmt.where(KnowledgeItem.project_id == project_id)
    if entity_id:
        stmt = stmt.where(KnowledgeItem.entity_id == entity_id)

    # Opt-in typed memory filters (gated by feature flag)
    if settings.typed_memory_enabled:
        if memory_type is not None:
            stmt = stmt.where(KnowledgeItem.memory_type == memory_type)
        if max_staleness is not None:
            # Include items with NULL staleness (not yet scored) AND items below threshold
            stmt = stmt.where(
                or_(
                    KnowledgeItem.staleness_score.is_(None),
                    KnowledgeItem.staleness_score <= max_staleness,
                )
            )

    rows = session.execute(stmt).all()
    results = []
    for item, cosine_val, _ws in rows:
        row_dict = {
            "id": str(item.id),
            "scope": "knowledge",
            "score": _safe_float(cosine_val),
            "snippet": _clip(item.content, settings.max_snippet_chars),
            "tags": list(item.tags or []),
            "created_at": item.created_at,
        }
        # Include typed memory metadata in response when feature is enabled
        if settings.typed_memory_enabled:
            row_dict["memory_type"] = item.memory_type
            row_dict["last_verified_at"] = item.last_verified_at
            row_dict["staleness_score"] = item.staleness_score
            row_dict["extraction_source"] = item.extraction_source
        results.append(row_dict)
    return results


def _apply_spreading_activation(session: Session, items: list[dict], namespace: str) -> list[dict]:
    """Boost concept recall results by spreading activation through concept graph.

    Concepts linked to highly-ranked results get a score boost.
    This helps surface associated concepts that may not match the query directly.
    """
    from .models import ConceptLink
    if not items or len(items) < 2:
        return items

    concept_ids = [i["id"] for i in items if i.get("scope") == "concepts"]
    if not concept_ids:
        return items

    from uuid import UUID as _UUID
    top_ids = set(_UUID(cid) for cid in concept_ids[:5])

    # Find linked concepts from top results
    linked = session.execute(
        select(ConceptLink.concept_a, ConceptLink.concept_b, ConceptLink.weight)
        .where(or_(
            ConceptLink.concept_a.in_(top_ids),
            ConceptLink.concept_b.in_(top_ids),
        ))
        .where(ConceptLink.weight >= 0.3)
    ).all()

    # Build boost map: concept_id -> accumulated boost from links
    boost_map: dict[str, float] = {}
    for a, b, w in linked:
        # The "other" concept in the pair gets a boost
        if a in top_ids:
            other = str(b)
        else:
            other = str(a)
        # Small boost proportional to link weight, capped
        boost_map[other] = boost_map.get(other, 0) + min(w * 0.02, 0.05)

    if not boost_map:
        return items

    # Apply boosts (lower score = better for cosine distance)
    for item in items:
        boost = boost_map.get(item["id"], 0)
        if boost > 0 and item.get("scope") == "concepts":
            if "reranker_score" in item and item["reranker_score"] is not None:
                item["reranker_score"] = item["reranker_score"] * (1 + min(boost, 0.15))
            else:
                item["score"] = item["score"] * (1 - min(boost, 0.15))

    # Re-sort
    if items and items[0].get("reranker_score") is not None:
        items.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)
    else:
        items.sort(key=lambda x: x["score"])

    return items


def _track_knowledge_recalls(session: Session, item_ids: list[str]) -> None:
    """Update recall tracking on knowledge items that were surfaced."""
    if not item_ids:
        return
    from uuid import UUID as _UUID
    uuids = [_UUID(i) for i in item_ids]
    session.execute(
        update(KnowledgeItem)
        .where(KnowledgeItem.id.in_(uuids))
        .values(
            last_recalled_at=func.now(),
            recall_count=KnowledgeItem.recall_count + 1,
        )
    )
    session.flush()


def _apply_reranker(
    query_text: str | None,
    items: list[dict],
    top_k: int,
    reranker: RerankerProvider,
) -> list[dict]:
    """Rerank recall results using the cross-encoder, then trim to top_k."""
    if not query_text or not reranker.is_available or not items:
        return items[:top_k]

    documents = [item["snippet"] for item in items]
    try:
        scores = reranker.rerank(query_text, documents)
    except RerankerProviderError:
        logger.warning("Reranker failed, falling back to vector scores", exc_info=True)
        return items[:top_k]

    for item, reranker_score in zip(items, scores):
        item["reranker_score"] = _safe_float(reranker_score, fallback=0.0)

    items.sort(key=lambda x: x["reranker_score"], reverse=True)
    return items[:top_k]


def recall(
    session: Session,
    payload: RecallRequest,
    embedding_provider: EmbeddingProvider,
    reranker_provider: RerankerProvider,
    settings,
) -> list[dict]:
    ensure_namespace_exists(session, payload.namespace)

    top_k = payload.top_k if payload.top_k is not None else settings.default_top_k
    top_k = min(max(top_k, 1), 50)

    # When reranker is available, fetch a wider candidate set for re-scoring.
    use_reranker = reranker_provider.is_available and payload.query_text
    candidate_k = top_k * settings.reranker_candidate_multiplier if use_reranker else top_k
    candidate_k = min(candidate_k, 150)

    query_embedding = payload.query_embedding
    if query_embedding is None:
        if not payload.query_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either query_embedding or query_text is required",
            )
        if settings.embedding_provider == "none":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="EMBEDDING_PROVIDER=none. Provide query_embedding explicitly.",
            )
        if not embedding_provider.is_available:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=embedding_provider.unavailable_reason or "Embedding provider unavailable",
            )
        try:
            query_embedding = embedding_provider.embed_text(payload.query_text)
        except EmbeddingProviderError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    _validate_embedding_dim(query_embedding, settings.embedding_dim)

    project_id = _resolve_entity_id(session, payload.namespace, payload.project_slug, "project_slug") if payload.project_slug else None
    entity_id = _resolve_entity_id(session, payload.namespace, payload.entity_slug, "entity_slug") if payload.entity_slug else None

    def _concept_to_recall(item, score_value):
        if item.type.value == "skill":
            parts = [f"[SKILL] {item.title}"]
            if item.trigger:
                parts.append(f"  When: {item.trigger}")
            if item.action:
                parts.append(f"  Do: {item.action}")
            if item.success_hint:
                parts.append(f"  Verify: {item.success_hint}")
            snippet = _clip("\n".join(parts), settings.max_snippet_chars)
        else:
            snippet = _clip(f"[{item.type.value}] {item.title}: {item.content}", settings.max_snippet_chars)
        return {
            "id": str(item.id),
            "scope": "concepts",
            "score": _safe_float(score_value),
            "snippet": snippet,
            "tags": list(item.tags or []),
            "created_at": item.created_at,
            "concept_type": item.type.value,
            "confidence": item.confidence,
            "trigger": item.trigger,
            "action": item.action,
            "success_hint": item.success_hint,
        }

    if payload.scope == "knowledge":
        items = _recall_knowledge_weighted(
            session, payload.namespace, query_embedding, candidate_k,
            project_id, entity_id, settings,
            memory_type=payload.memory_type,
            max_staleness=payload.max_staleness,
        )
        items = _apply_reranker(payload.query_text, items, top_k, reranker_provider)
        _track_knowledge_recalls(session, [i["id"] for i in items])
        if payload.agent_key:
            update_profile_from_recall(session, payload.agent_key, payload.namespace)
        return items

    if payload.scope == "requirements":
        score = Requirement.embedding.cosine_distance(query_embedding).label("score")
        stmt = (
            select(Requirement, score)
            .where(Requirement.namespace == payload.namespace)
            .where(Requirement.embedding.is_not(None))
            .order_by(score)
            .limit(candidate_k)
        )
        if project_id:
            stmt = stmt.where(Requirement.project_id == project_id)
        if entity_id:
            stmt = stmt.where(Requirement.owner_entity_id == entity_id)

        rows = session.execute(stmt).all()
        items = [
            {
                "id": str(item.id),
                "scope": "requirements",
                "score": _safe_float(score_value),
                "snippet": _clip(item.context_snippet or item.body or item.title, settings.max_snippet_chars),
                "tags": list(item.tags or []),
                "created_at": item.created_at,
            }
            for item, score_value in rows
        ]
        items = _apply_reranker(payload.query_text, items, top_k, reranker_provider)
        if payload.agent_key:
            update_profile_from_recall(session, payload.agent_key, payload.namespace)
        return items

    if payload.scope == "events":
        score = Event.embedding.cosine_distance(query_embedding).label("score")
        stmt = (
            select(Event, score)
            .where(Event.namespace == payload.namespace)
            .where(Event.embedding.is_not(None))
            .order_by(score)
            .limit(candidate_k)
        )
        if project_id:
            stmt = stmt.where(Event.project_id == project_id)
        if entity_id:
            stmt = stmt.where(Event.agent_id == entity_id)

        rows = session.execute(stmt).all()
        items = [
            {
                "id": str(item.id),
                "scope": "events",
                "score": _safe_float(score_value),
                "snippet": _clip(item.body or item.context_snippet or item.title, settings.max_snippet_chars),
                "tags": list(item.tags or []),
                "created_at": item.created_at,
            }
            for item, score_value in rows
        ]
        items = _apply_reranker(payload.query_text, items, top_k, reranker_provider)
        if payload.agent_key:
            update_profile_from_recall(session, payload.agent_key, payload.namespace)
        return items

    if payload.scope == "concepts":
        score = Concept.embedding.cosine_distance(query_embedding).label("score")
        stmt = (
            select(Concept, score)
            .where(Concept.namespace == payload.namespace)
            .where(Concept.embedding.is_not(None))
            .where(~Concept.tags.any("superseded"))
            .order_by(score)
            .limit(candidate_k)
        )

        rows = session.execute(stmt).all()

        items = [_concept_to_recall(item, score_value) for item, score_value in rows]
        items = _apply_reranker(payload.query_text, items, top_k, reranker_provider)
        items = _apply_spreading_activation(session, items, payload.namespace)
        if payload.agent_key:
            items = _boost_by_agent_profile(session, items, payload.agent_key, payload.namespace)
        # Log recall events for concepts that were surfaced
        if payload.session_id:
            concept_items = [item for item in items if item.get("scope") == "concepts"]
            if concept_items:
                from uuid import UUID as _UUID
                concept_ids = [_UUID(item["id"]) for item in concept_items]
                scores_map = {_UUID(item["id"]): item.get("score", 0.0) for item in concept_items}
                log_recall_events(
                    session=session,
                    namespace=payload.namespace,
                    concept_ids=concept_ids,
                    session_id=payload.session_id,
                    agent_key=payload.agent_key or "unknown",
                    query_text=payload.query_text or "",
                    scores=scores_map,
                )
        update_profile_from_recall(session, payload.agent_key or "unknown", payload.namespace)
        return items

    if payload.scope == "all":
        # --- knowledge items (weighted) ---
        items = _recall_knowledge_weighted(
            session, payload.namespace, query_embedding, candidate_k,
            project_id, entity_id, settings,
            memory_type=payload.memory_type,
            max_staleness=payload.max_staleness,
        )

        # --- concepts ---
        c_score = Concept.embedding.cosine_distance(query_embedding).label("score")
        c_stmt = (
            select(Concept, c_score)
            .where(Concept.namespace == payload.namespace)
            .where(Concept.embedding.is_not(None))
            .where(~Concept.tags.any("superseded"))
            .order_by(c_score)
            .limit(candidate_k)
        )

        c_rows = session.execute(c_stmt).all()
        items.extend([_concept_to_recall(item, score_value) for item, score_value in c_rows])

        # merge by vector score (lower = closer for cosine_distance)
        items.sort(key=lambda x: x["score"])
        items = items[:candidate_k]

        items = _apply_reranker(payload.query_text, items, top_k, reranker_provider)
        items = _apply_spreading_activation(session, items, payload.namespace)
        if payload.agent_key:
            items = _boost_by_agent_profile(session, items, payload.agent_key, payload.namespace)
        # Track knowledge item recalls
        ki_ids = [i["id"] for i in items if i.get("scope") == "knowledge"]
        if ki_ids:
            _track_knowledge_recalls(session, ki_ids)
        # Log recall events for concepts that were surfaced
        if payload.session_id:
            concept_items = [item for item in items if item.get("scope") == "concepts"]
            if concept_items:
                from uuid import UUID as _UUID
                concept_ids = [_UUID(item["id"]) for item in concept_items]
                scores_map = {_UUID(item["id"]): item.get("score", 0.0) for item in concept_items}
                log_recall_events(
                    session=session,
                    namespace=payload.namespace,
                    concept_ids=concept_ids,
                    session_id=payload.session_id,
                    agent_key=payload.agent_key or "unknown",
                    query_text=payload.query_text or "",
                    scores=scores_map,
                )
        update_profile_from_recall(session, payload.agent_key or "unknown", payload.namespace)
        return items

    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown scope '{payload.scope}'")


def create_namespace(session: Session, name: str) -> Namespace:
    existing = session.get(Namespace, name)
    if existing:
        return existing

    namespace = Namespace(name=name)
    session.add(namespace)
    session.commit()
    session.refresh(namespace)
    return namespace


def create_api_key_record(session: Session, name: str, role: str, namespaces: list[str]) -> tuple[ApiKey, str]:
    normalized_namespaces = sorted(set(ns.strip() for ns in namespaces if ns.strip()))
    if not normalized_namespaces:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one namespace is required")

    # Allow "*" as a special wildcard namespace marker without requiring it to exist in the namespaces table.
    namespaces_to_check = [ns for ns in normalized_namespaces if ns != "*"]
    existing_namespaces = set(session.scalars(select(Namespace.name).where(Namespace.name.in_(namespaces_to_check))).all())
    missing = [ns for ns in namespaces_to_check if ns not in existing_namespaces]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown namespaces: {', '.join(missing)}",
        )

    plaintext = generate_api_key()
    record = ApiKey(
        name=name,
        key_hash=hash_api_key(plaintext),
        role=ApiRole(role),
        namespaces=normalized_namespaces,
        is_active=True,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record, plaintext


def list_api_keys_masked(session: Session) -> list[dict]:
    rows = session.scalars(select(ApiKey).order_by(ApiKey.created_at.desc())).all()
    return [
        {
            "id": str(row.id),
            "name": row.name,
            "role": row.role.value if isinstance(row.role, ApiRole) else str(row.role),
            "namespaces": list(row.namespaces or []),
            "is_active": row.is_active,
            "created_at": row.created_at,
            "key_preview": f"bcrypt:{row.key_hash[:12]}...",
        }
        for row in rows
    ]


def backfill_embeddings(
    session: Session,
    namespace: str,
    scope: str,
    embedding_provider: EmbeddingProvider,
    settings,
    limit: int = 500,
    batch_size: int = 50,
    dry_run: bool = False,
) -> dict:
    ensure_namespace_exists(session, namespace)

    if settings.embedding_provider == "none":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="EMBEDDING_PROVIDER=none. Cannot backfill embeddings server-side.",
        )

    if not embedding_provider.is_available:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=embedding_provider.unavailable_reason or "Embedding provider unavailable",
        )

    limit = min(max(int(limit), 1), 50_000)
    batch_size = min(max(int(batch_size), 1), 500)

    updated = 0
    skipped = 0

    max_chars = 6000  # safe limit for nomic-embed-text 8192-token context

    def embed_source_text(text: str) -> list[float] | None:
        if not text.strip():
            return None
        emb = embedding_provider.embed_text(text[:max_chars])
        _validate_embedding_dim(emb, settings.embedding_dim)
        return emb

    remaining = limit
    while remaining > 0:
        take = min(batch_size, remaining)

        if scope == "knowledge":
            rows = session.scalars(
                select(KnowledgeItem)
                .where(KnowledgeItem.namespace == namespace)
                .where(KnowledgeItem.embedding.is_(None))
                .order_by(KnowledgeItem.created_at.desc())
                .limit(take)
            ).all()
            if not rows:
                break

            for item in rows:
                try:
                    emb = embed_source_text(item.content)
                except EmbeddingProviderError as exc:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
                if emb is None:
                    skipped += 1
                    continue
                if not dry_run:
                    item.embedding = emb
                updated += 1

        elif scope == "requirements":
            rows = session.scalars(
                select(Requirement)
                .where(Requirement.namespace == namespace)
                .where(Requirement.embedding.is_(None))
                .order_by(Requirement.updated_at.desc())
                .limit(take)
            ).all()
            if not rows:
                break

            for req in rows:
                source_text = "\n".join(part for part in [req.title, req.body or "", req.context_snippet or ""] if part)
                try:
                    emb = embed_source_text(source_text)
                except EmbeddingProviderError as exc:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
                if emb is None:
                    skipped += 1
                    continue
                if not dry_run:
                    req.embedding = emb
                updated += 1

        elif scope == "events":
            rows = session.scalars(
                select(Event)
                .where(Event.namespace == namespace)
                .where(Event.embedding.is_(None))
                .order_by(Event.created_at.desc())
                .limit(take)
            ).all()
            if not rows:
                break

            for ev in rows:
                source_text = "\n".join(part for part in [ev.title, ev.body or "", ev.context_snippet or ""] if part)
                try:
                    emb = embed_source_text(source_text)
                except EmbeddingProviderError as exc:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
                if emb is None:
                    skipped += 1
                    continue
                if not dry_run:
                    ev.embedding = emb
                updated += 1

        elif scope == "concepts":
            rows = session.scalars(
                select(Concept)
                .where(Concept.namespace == namespace)
                .where(Concept.embedding.is_(None))
                .order_by(Concept.updated_at.desc())
                .limit(take)
            ).all()
            if not rows:
                break

            for concept in rows:
                source_text = "\n".join(part for part in [concept.title, concept.content or ""] if part)
                try:
                    emb = embed_source_text(source_text)
                except EmbeddingProviderError as exc:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
                if emb is None:
                    skipped += 1
                    continue
                if not dry_run:
                    concept.embedding = emb
                updated += 1

        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown scope '{scope}'")

        if not dry_run:
            session.commit()

        remaining -= take

    return {"namespace": namespace, "scope": scope, "updated": updated, "skipped": skipped, "dry_run": dry_run}


def find_compaction_candidates(
    session: Session,
    event_threshold: int,
    char_threshold: int,
) -> list[dict]:
    char_count_expr = func.coalesce(
        func.sum(
            func.length(func.coalesce(Event.title, ""))
            + func.length(func.coalesce(Event.body, ""))
            + func.length(func.coalesce(Event.context_snippet, ""))
        ),
        0,
    )

    stmt = (
        select(
            Event.namespace,
            Event.project_id,
            Event.session_id,
            func.count(Event.id).label("event_count"),
            char_count_expr.label("char_count"),
        )
        .where(Event.is_archived.is_(False))
        .where(Event.session_id.is_not(None))
        .group_by(Event.namespace, Event.project_id, Event.session_id)
        .having(or_(func.count(Event.id) > event_threshold, char_count_expr > char_threshold))
    )

    rows = session.execute(stmt).all()
    return [
        {
            "namespace": namespace,
            "project_id": project_id,
            "session_id": session_id,
            "event_count": int(event_count),
            "char_count": int(char_count),
        }
        for namespace, project_id, session_id, event_count, char_count in rows
        if session_id
    ]


def _build_event_transcript(events: list[Event]) -> str:
    lines: list[str] = []
    for event in events:
        created = event.created_at.isoformat() if event.created_at else ""
        lines.append(f"[{created}] ({event.type.value}) {event.title}")
        if event.body:
            lines.append(event.body)
        if event.context_snippet:
            lines.append(f"Context: {event.context_snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def summarize_and_archive_session(
    session: Session,
    namespace: str,
    session_id: str,
    llm_provider: LLMProvider,
    embedding_provider: EmbeddingProvider,
    settings,
    project_slug: str | None = None,
    project_id: UUID | None = None,
    max_events: int = 500,
) -> dict:
    ensure_namespace_exists(session, namespace)

    if not llm_provider.is_available:
        raise LLMProviderError(llm_provider.unavailable_reason or "LLM provider unavailable")

    project_filter_id: UUID | None = project_id
    if project_slug:
        project_filter_id = _resolve_entity_id(session, namespace, project_slug, "project_slug")

    stmt = (
        select(Event)
        .where(Event.namespace == namespace)
        .where(Event.session_id == session_id)
        .where(Event.is_archived.is_(False))
        .order_by(Event.created_at.asc())
        .limit(min(max_events, 2000))
    )
    if project_filter_id:
        stmt = stmt.where(Event.project_id == project_filter_id)

    events = list(session.scalars(stmt).all())
    if not events:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No unarchived events found for session_id '{session_id}'",
        )

    transcript = _build_event_transcript(events)
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session transcript is empty; nothing to summarize",
        )

    summary = llm_provider.summarize(transcript)
    if not summary:
        raise LLMProviderError("Summarizer returned an empty summary")

    summary_embedding = None
    if settings.embedding_provider != "none" and embedding_provider.is_available:
        try:
            summary_embedding = embedding_provider.embed_text(summary)
            _validate_embedding_dim(summary_embedding, settings.embedding_dim)
        except Exception:
            summary_embedding = None

    summary_item = KnowledgeItem(
        namespace=namespace,
        project_id=project_filter_id or events[0].project_id,
        entity_id=None,
        content=summary,
        source="session_compaction",
        source_ref=session_id,
        tags=["summary", "compaction"],
        embedding=summary_embedding,
        memory_type="session_summary",
    )
    session.add(summary_item)

    for event in events:
        event.is_archived = True

    session.commit()
    session.refresh(summary_item)

    return {
        "namespace": namespace,
        "session_id": session_id,
        "archived_events": len(events),
        "summary_knowledge_item_id": str(summary_item.id),
    }


def delete_items(session: Session, namespace: str, ids: list[str]) -> int:
    """Delete knowledge items by ID within a namespace. Returns count deleted."""
    ensure_namespace_exists(session, namespace)
    deleted = 0
    for item_id in ids:
        try:
            uid = UUID(item_id)
        except ValueError:
            continue
        item = session.scalar(
            select(KnowledgeItem).where(
                and_(KnowledgeItem.id == uid, KnowledgeItem.namespace == namespace)
            )
        )
        if item:
            session.delete(item)
            deleted += 1
    session.commit()
    return deleted


def delete_concepts(session: Session, namespace: str, ids: list[str]) -> dict:
    """Delete concepts by ID within a namespace.

    Also cleans up associated concept_links, recall_events, and concept_feedback
    via CASCADE, but we explicitly handle it for clarity.
    Returns dict with deleted count and IDs.
    """
    ensure_namespace_exists(session, namespace)
    deleted_ids: list[str] = []
    for concept_id in ids:
        try:
            uid = UUID(concept_id)
        except ValueError:
            continue
        concept = session.scalar(
            select(Concept).where(
                and_(Concept.id == uid, Concept.namespace == namespace)
            )
        )
        if concept:
            session.delete(concept)
            deleted_ids.append(str(uid))
    session.commit()
    return {"deleted": len(deleted_ids), "deleted_ids": deleted_ids}


# ── Dedup ──────────────────────────────────────────────────────────


def dedup_knowledge_items(session: Session, namespace: str, dry_run: bool = False) -> dict:
    """Find and remove duplicate knowledge items within a namespace.

    Keeps the oldest row for each unique content string; deletes newer duplicates.
    Returns a summary with duplicate_groups count and deleted count.
    """
    ensure_namespace_exists(session, namespace)

    # Find content values that appear more than once
    dupe_query = (
        select(KnowledgeItem.content, func.count().label("cnt"), func.min(KnowledgeItem.created_at).label("oldest"))
        .where(KnowledgeItem.namespace == namespace)
        .group_by(KnowledgeItem.content)
        .having(func.count() > 1)
    )
    dupe_rows = session.execute(dupe_query).all()

    duplicate_groups = len(dupe_rows)
    total_deleted = 0

    if not dry_run:
        for content, cnt, oldest_at in dupe_rows:
            # Keep the oldest, delete the rest
            oldest_id = session.scalar(
                select(KnowledgeItem.id)
                .where(and_(KnowledgeItem.namespace == namespace, KnowledgeItem.content == content))
                .order_by(KnowledgeItem.created_at.asc())
                .limit(1)
            )
            if oldest_id is None:
                continue
            deleted = session.execute(
                sa_delete(KnowledgeItem).where(
                    and_(
                        KnowledgeItem.namespace == namespace,
                        KnowledgeItem.content == content,
                        KnowledgeItem.id != oldest_id,
                    )
                )
            )
            total_deleted += deleted.rowcount
        session.commit()
    else:
        total_deleted = sum((cnt - 1) for _, cnt, _ in dupe_rows)

    return {
        "namespace": namespace,
        "duplicate_groups": duplicate_groups,
        "deleted": total_deleted,
        "dry_run": dry_run,
    }


# ── Stats ──────────────────────────────────────────────────────────


def get_namespace_stats(session: Session, namespace: str) -> dict:
    """Return item counts and embedding coverage for a namespace."""
    ensure_namespace_exists(session, namespace)

    ki_total = session.scalar(
        select(func.count()).select_from(KnowledgeItem).where(KnowledgeItem.namespace == namespace)
    ) or 0
    ki_embedded = session.scalar(
        select(func.count()).select_from(KnowledgeItem).where(
            and_(KnowledgeItem.namespace == namespace, KnowledgeItem.embedding.is_not(None))
        )
    ) or 0

    req_total = session.scalar(
        select(func.count()).select_from(Requirement).where(Requirement.namespace == namespace)
    ) or 0
    req_embedded = session.scalar(
        select(func.count()).select_from(Requirement).where(
            and_(Requirement.namespace == namespace, Requirement.embedding.is_not(None))
        )
    ) or 0

    ev_total = session.scalar(
        select(func.count()).select_from(Event).where(Event.namespace == namespace)
    ) or 0
    ev_embedded = session.scalar(
        select(func.count()).select_from(Event).where(
            and_(Event.namespace == namespace, Event.embedding.is_not(None))
        )
    ) or 0

    entity_total = session.scalar(
        select(func.count()).select_from(Entity).where(Entity.namespace == namespace)
    ) or 0

    concept_total = session.scalar(
        select(func.count()).select_from(Concept).where(Concept.namespace == namespace)
    ) or 0
    concept_embedded = session.scalar(
        select(func.count()).select_from(Concept).where(
            and_(Concept.namespace == namespace, Concept.embedding.is_not(None))
        )
    ) or 0

    return {
        "namespace": namespace,
        "entities": entity_total,
        "knowledge_items": {"total": ki_total, "embedded": ki_embedded},
        "requirements": {"total": req_total, "embedded": req_embedded},
        "events": {"total": ev_total, "embedded": ev_embedded},
        "concepts": {"total": concept_total, "embedded": concept_embedded},
    }


# ── Concepts ──────────────────────────────────────────────────────


def list_concepts(
    session: Session,
    namespace: str,
    concept_type: str | None = None,
    scope: str | None = None,
    include_superseded: bool = False,
    limit: int = 50,
) -> list[dict]:
    """List concepts for a namespace with optional filters."""
    ensure_namespace_exists(session, namespace)

    stmt = select(Concept).where(Concept.namespace == namespace)
    if not include_superseded:
        stmt = stmt.where(~Concept.tags.any("superseded"))
    if concept_type:
        stmt = stmt.where(Concept.type == ConceptType(concept_type))
    if scope:
        stmt = stmt.where(Concept.scope == ConceptScope(scope))
    stmt = stmt.order_by(Concept.confidence.desc()).limit(min(max(limit, 1), 1000))

    rows = session.scalars(stmt).all()
    return [
        {
            "id": str(c.id),
            "namespace": c.namespace,
            "type": c.type.value,
            "scope": c.scope.value if hasattr(c.scope, "value") else str(c.scope),
            "scope_ref": c.scope_ref,
            "title": c.title,
            "content": c.content,
            "evidence_ids": [str(eid) for eid in (c.evidence_ids or [])],
            "confidence": c.confidence,
            "version": c.version,
            "parent_id": str(c.parent_id) if c.parent_id else None,
            "tags": list(c.tags or []),
            "trigger": c.trigger,
            "action": c.action,
            "success_hint": c.success_hint,
            "confirming_agents": list(c.confirming_agents or []),
            "created_at": c.created_at,
            "updated_at": c.updated_at,
        }
        for c in rows
    ]


def list_concept_links(
    session: Session,
    namespace: str,
    limit: int = 200,
) -> list[dict]:
    """List concept links for a namespace, ordered by weight descending."""
    from .models import ConceptLink, Concept

    stmt = (
        select(
            ConceptLink.concept_a,
            ConceptLink.concept_b,
            ConceptLink.weight,
            ConceptLink.link_type,
            ConceptLink.co_occurrence_count,
        )
        .join(Concept, ConceptLink.concept_a == Concept.id)
        .where(Concept.namespace == namespace)
        .order_by(ConceptLink.weight.desc())
        .limit(min(max(limit, 1), 500))
    )

    rows = session.execute(stmt).all()
    return [
        {
            "source": str(r.concept_a),
            "target": str(r.concept_b),
            "weight": r.weight,
            "link_type": r.link_type,
            "co_occurrences": r.co_occurrence_count,
        }
        for r in rows
    ]


def get_concept_with_history(session: Session, namespace: str, concept_id: str) -> dict | None:
    """Get a concept and its version history chain."""
    ensure_namespace_exists(session, namespace)
    try:
        uid = UUID(concept_id)
    except ValueError:
        return None

    concept = session.scalar(
        select(Concept).where(and_(Concept.id == uid, Concept.namespace == namespace))
    )
    if not concept:
        return None

    def _to_dict(c):
        return {
            "id": str(c.id),
            "namespace": c.namespace,
            "type": c.type.value,
            "scope": c.scope.value if hasattr(c.scope, "value") else str(c.scope),
            "scope_ref": c.scope_ref,
            "title": c.title,
            "content": c.content,
            "evidence_ids": [str(eid) for eid in (c.evidence_ids or [])],
            "confidence": c.confidence,
            "version": c.version,
            "parent_id": str(c.parent_id) if c.parent_id else None,
            "tags": list(c.tags or []),
            "trigger": c.trigger,
            "action": c.action,
            "success_hint": c.success_hint,
            "confirming_agents": list(c.confirming_agents or []),
            "created_at": c.created_at,
            "updated_at": c.updated_at,
        }

    result = _to_dict(concept)

    history = []
    current = concept
    while current.parent_id:
        parent = session.scalar(
            select(Concept).where(Concept.id == current.parent_id)
        )
        if not parent:
            break
        history.append(_to_dict(parent))
        current = parent

    result["history"] = history
    return result


# ── Enrollment tokens ──────────────────────────────────────────────


def create_enrollment_token(
    session: Session,
    role: str,
    namespaces: list[str],
    name_hint: str | None,
    max_uses: int,
    expires_at: datetime | None,
) -> tuple[EnrollmentToken, str]:
    normalized = sorted(set(ns.strip() for ns in namespaces if ns.strip()))
    if not normalized:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one namespace is required")

    # Enrollment tokens cannot grant admin role
    if role == "admin":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Enrollment tokens cannot grant admin role")

    # Validate namespaces exist
    ns_to_check = [ns for ns in normalized if ns != "*"]
    existing_ns = set(session.scalars(select(Namespace.name).where(Namespace.name.in_(ns_to_check))).all())
    missing = [ns for ns in ns_to_check if ns not in existing_ns]
    if missing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown namespaces: {', '.join(missing)}")

    plaintext = generate_enrollment_token()
    token = EnrollmentToken(
        token_hash=hash_api_key(plaintext),
        role=ApiRole(role),
        namespaces=normalized,
        name_hint=name_hint,
        max_uses=max_uses,
        expires_at=expires_at,
        is_active=True,
    )
    session.add(token)
    session.commit()
    session.refresh(token)
    return token, plaintext


def list_enrollment_tokens(session: Session) -> list[dict]:
    rows = session.scalars(select(EnrollmentToken).order_by(EnrollmentToken.created_at.desc())).all()
    return [
        {
            "id": str(t.id),
            "role": t.role.value if isinstance(t.role, ApiRole) else str(t.role),
            "namespaces": list(t.namespaces or []),
            "name_hint": t.name_hint,
            "max_uses": t.max_uses,
            "times_used": t.times_used,
            "expires_at": t.expires_at,
            "is_active": t.is_active,
            "created_at": t.created_at,
        }
        for t in rows
    ]


def redeem_enrollment_token(session: Session, plaintext_token: str, key_name: str) -> tuple[ApiKey, str]:
    """Validate an enrollment token and create a new API key."""
    active_tokens = session.scalars(
        select(EnrollmentToken).where(EnrollmentToken.is_active.is_(True))
    ).all()

    matched_token: EnrollmentToken | None = None
    for token in active_tokens:
        if verify_api_key(plaintext_token, token.token_hash):
            matched_token = token
            break

    if not matched_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired enrollment token")

    # Check expiry
    if matched_token.expires_at and matched_token.expires_at < datetime.now(UTC):
        matched_token.is_active = False
        session.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Enrollment token has expired")

    # Check usage limit
    if matched_token.times_used >= matched_token.max_uses:
        matched_token.is_active = False
        session.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Enrollment token has been fully used")

    # Create the API key
    name = key_name.strip()
    if matched_token.name_hint:
        name = f"{matched_token.name_hint}-{name}"

    record, api_key_plaintext = create_api_key_record(
        session=session,
        name=name,
        role=matched_token.role.value if isinstance(matched_token.role, ApiRole) else str(matched_token.role),
        namespaces=list(matched_token.namespaces or []),
    )

    # Increment usage
    matched_token.times_used += 1
    if matched_token.times_used >= matched_token.max_uses:
        matched_token.is_active = False

    session.commit()
    return record, api_key_plaintext


def auto_enroll(session: Session, name: str, passphrase: str, settings) -> tuple[ApiKey, str, dict]:
    """Open enrollment: create namespace + writer key for a new device. Returns (ApiKey, plaintext_key, agent_config_dict)."""
    expected = settings.enrollment_passphrase
    if expected and passphrase != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid enrollment passphrase")

    # Sanitize device name for namespace
    safe_name = name.strip().lower().replace(" ", "-")
    if not safe_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Device name is required")

    private_ns = f"device-{safe_name}-private"
    shared_ns = "claude-shared"

    # Ensure both namespaces exist
    for ns in [shared_ns, private_ns]:
        create_namespace(session, ns)

    # Create writer key scoped to shared + private
    namespaces = [shared_ns, private_ns]
    key_name = f"enroll-{safe_name}"
    record, plaintext = create_api_key_record(
        session=session,
        name=key_name,
        role="writer",
        namespaces=namespaces,
    )

    agent_config = {
        "base_url": f"http://localhost:{settings.api_port}",
        "api_key": plaintext,
        "device_name": safe_name,
        "default_namespace": shared_ns,
        "default_scope": "knowledge",
        "namespaces": namespaces,
    }

    return record, plaintext, agent_config


# ── Agent Profiles (Layer 4: Personalization) ─────────────────────


def get_or_create_profile(session: Session, agent_key: str, namespace: str) -> AgentProfile:
    """Get existing profile or create one, extracting device_name from key."""
    profile = session.get(AgentProfile, agent_key)
    if profile:
        return profile
    # Extract device name: "device-hari-writer" -> "hari"
    parts = agent_key.split("-")
    device_name = parts[1] if len(parts) >= 3 and parts[0] == "device" else agent_key
    profile = AgentProfile(
        agent_key=agent_key,
        namespace=namespace,
        device_name=device_name,
    )
    session.add(profile)
    session.flush()
    return profile


def update_profile_from_recall(session: Session, agent_key: str, namespace: str) -> None:
    """Increment recall counter on profile."""
    if not agent_key:
        return
    profile = get_or_create_profile(session, agent_key, namespace)
    profile.total_recalls = (profile.total_recalls or 0) + 1
    session.flush()


def update_profile_from_feedback(session: Session, agent_key: str, namespace: str) -> None:
    """Increment feedback counter on profile."""
    if not agent_key:
        return
    profile = get_or_create_profile(session, agent_key, namespace)
    profile.total_feedback = (profile.total_feedback or 0) + 1
    session.flush()


def get_agent_profile(session: Session, agent_key: str) -> dict | None:
    """Return profile as dict, or None if not found."""
    profile = session.get(AgentProfile, agent_key)
    if not profile:
        return None
    return {
        "agent_key": profile.agent_key,
        "namespace": profile.namespace,
        "device_name": profile.device_name,
        "model_hint": profile.model_hint,
        "strengths": list(profile.strengths or []),
        "preferences": dict(profile.preferences or {}),
        "total_recalls": profile.total_recalls,
        "total_feedback": profile.total_feedback,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }


def _boost_by_agent_profile(
    session: Session, items: list[dict], agent_key: str, namespace: str
) -> list[dict]:
    """Lightly boost recall results matching agent's strength topics.

    Handles both scoring modes:
    - If reranker_score is present: higher = better, boost increases it.
    - If only cosine distance score: lower = better, boost decreases it.
    """
    profile = get_or_create_profile(session, agent_key, namespace)
    strengths = set(s.lower() for s in (profile.strengths or []))
    if not strengths:
        return items
    for item in items:
        snippet_lower = item.get("snippet", "").lower()
        matches = sum(1 for s in strengths if s in snippet_lower)
        if matches > 0:
            boost = min(matches * 0.10, 0.30)
            if "reranker_score" in item and item["reranker_score"] is not None:
                # Reranker score: higher = better
                item["reranker_score"] = item["reranker_score"] * (1 + boost)
            else:
                # Cosine distance: lower = better
                item["score"] = item["score"] * (1 - boost)
    # Re-sort based on which scoring mode is active
    if items and items[0].get("reranker_score") is not None:
        items.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)
    else:
        items.sort(key=lambda x: x["score"])
    return items


# ── Feedback loop ──────────────────────────────────────────────────


def log_recall_events(
    session: Session,
    namespace: str,
    concept_ids: list[UUID],
    session_id: str,
    agent_key: str,
    query_text: str,
    scores: dict[UUID, float],
) -> int:
    """Log recall events for concepts that were surfaced. Returns count logged."""
    count = 0
    for cid in concept_ids:
        session.add(RecallEvent(
            concept_id=cid,
            namespace=namespace,
            session_id=session_id,
            agent_key=agent_key,
            query_text=query_text,
            score=scores.get(cid, 0.0),
        ))
        count += 1
    session.flush()
    return count


def create_concept_feedback(
    session: Session,
    namespace: str,
    signal: str,
    agent_key: str,
    embedding_provider,
    concept_id: UUID | None = None,
    query: str | None = None,
    session_id: str | None = None,
    note: str | None = None,
) -> dict:
    """Create feedback for a concept. Resolves by ID or fuzzy semantic match."""
    if not concept_id and not query:
        raise ValueError("Must provide concept_id or query")

    if concept_id:
        concept = session.get(Concept, concept_id)
        if not concept or concept.namespace != namespace:
            raise ValueError(f"Concept {concept_id} not found in namespace {namespace}")
    else:
        # Fuzzy match: embed query, find closest concept
        query_embedding = embedding_provider.embed_text(query)
        score_col = Concept.embedding.cosine_distance(query_embedding).label("score")
        stmt = (
            select(Concept, score_col)
            .where(Concept.namespace == namespace)
            .where(Concept.embedding.is_not(None))
            .where(~Concept.tags.any("superseded"))
            .order_by(score_col)
            .limit(1)
        )
        row = session.execute(stmt).first()
        if not row:
            raise ValueError("No concepts found to match query")
        concept, _match_score = row
        concept_id = concept.id

    fb = ConceptFeedback(
        concept_id=concept_id,
        namespace=namespace,
        signal=FeedbackSignal(signal),
        agent_key=agent_key,
        session_id=session_id,
        note=note,
    )
    session.add(fb)

    # If "wrong" signal with a note, also ingest the correction as knowledge
    if signal == "wrong" and note:
        session.add(KnowledgeItem(
            namespace=namespace,
            content=f"Correction: {note} (re: concept '{concept.title}')",
            tags=["correction", "operator-feedback"],
            embedding=embedding_provider.embed_text(note),
            memory_type="correction",
        ))

    # Resurrect dormant concepts on positive feedback
    if signal in ("useful", "noted") and concept.confidence < 0.5:
        from app.synthesis.reconciler import resurrect_concept
        resurrect_concept(session, concept_id)
        concept = session.get(Concept, concept_id)  # refresh after resurrection

    # Layer 5: Add agent to confirming_agents on positive feedback
    if signal in ("useful", "noted") and concept and agent_key:
        existing_agents = set(concept.confirming_agents or [])
        if agent_key not in existing_agents:
            existing_agents.add(agent_key)
            concept.confirming_agents = list(existing_agents)

    session.flush()

    update_profile_from_feedback(session, agent_key, namespace)

    return {
        "concept_id": str(concept_id),
        "concept_title": concept.title,
        "signal": signal,
        "confidence": concept.confidence,
    }


def get_feedback_summary(
    session: Session,
    namespace: str,
    since: datetime,
) -> dict[UUID, dict]:
    """Get aggregated feedback per concept since a timestamp.
    Returns {concept_id: {"useful": N, "noted": N, "wrong": N, "implicit_recalls": N, "wrong_notes": [...]}}
    """
    from collections import defaultdict

    summary: dict[UUID, dict] = defaultdict(lambda: {
        "useful": 0, "noted": 0, "wrong": 0, "implicit_recalls": 0, "wrong_notes": []
    })

    # Explicit feedback
    fb_stmt = (
        select(ConceptFeedback)
        .where(ConceptFeedback.namespace == namespace)
        .where(ConceptFeedback.created_at >= since)
    )
    for fb in session.scalars(fb_stmt):
        summary[fb.concept_id][fb.signal.value] += 1
        if fb.signal == FeedbackSignal.wrong and fb.note:
            summary[fb.concept_id]["wrong_notes"].append(fb.note)

    # Implicit recalls (session completed)
    re_stmt = (
        select(RecallEvent.concept_id, func.count(func.distinct(RecallEvent.session_id)))
        .where(RecallEvent.namespace == namespace)
        .where(RecallEvent.session_completed == True)
        .where(RecallEvent.created_at >= since)
        .group_by(RecallEvent.concept_id)
    )
    for cid, count in session.execute(re_stmt):
        summary[cid]["implicit_recalls"] = count

    return dict(summary)


KNOWLEDGE_STRENGTH_DELTAS = {
    "useful": 0.2,
    "noted": 0.1,
    "wrong": -0.5,
}


def create_knowledge_feedback(
    session: Session,
    namespace: str,
    signal: str,
    agent_key: str,
    knowledge_item_id: UUID,
    note: str | None = None,
) -> dict:
    """Apply feedback signal to a knowledge item's strength."""
    item = session.get(KnowledgeItem, knowledge_item_id)
    if not item or item.namespace != namespace:
        raise ValueError(f"Knowledge item {knowledge_item_id} not found in namespace {namespace}")

    delta = KNOWLEDGE_STRENGTH_DELTAS.get(signal, 0)
    new_strength = max(item.strength + delta, 0.1)
    item.strength = new_strength

    if delta > 0:
        item.positive_feedback = (item.positive_feedback or 0) + 1
    elif delta < 0:
        item.negative_feedback = (item.negative_feedback or 0) + 1

    if signal == "wrong":
        item.status = "flagged"

    session.flush()
    return {
        "knowledge_item_id": str(knowledge_item_id),
        "signal": signal,
        "strength": new_strength,
    }


def mark_session_completed(session: Session, session_id: str) -> int:
    """Mark all recall events for a session as completed. Returns count updated."""
    stmt = (
        update(RecallEvent)
        .where(RecallEvent.session_id == session_id)
        .where(RecallEvent.session_completed == False)
        .values(session_completed=True)
    )
    result = session.execute(stmt)
    session.flush()
    return result.rowcount
