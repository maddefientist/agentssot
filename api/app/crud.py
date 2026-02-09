import logging
from datetime import UTC, datetime
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from .chunking import chunk_text_semantic
from .embeddings import EmbeddingProvider, EmbeddingProviderError
from .llm import LLMProvider, LLMProviderError
from .models import ApiKey, ApiRole, Entity, EntityType, Event, EventType, KnowledgeItem, Namespace, Requirement
from .schemas import IngestRequest, RecallRequest
from .security import generate_api_key, hash_api_key

logger = logging.getLogger("agentssot.crud")


def _clip(text: str | None, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


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
                chunk_embedding = item.embedding
                if chunk_embedding is None:
                    chunk_embedding = _maybe_embed_text(embedding_provider, chunk, settings.embedding_provider)
                    _validate_embedding_dim(chunk_embedding, settings.embedding_dim)

                session.add(
                    KnowledgeItem(
                        namespace=payload.namespace,
                        project_id=project_id,
                        entity_id=entity_id,
                        content=chunk,
                        source=item.source,
                        source_ref=item.source_ref,
                        tags=item.tags,
                        embedding=chunk_embedding,
                    )
                )
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


def recall(
    session: Session,
    payload: RecallRequest,
    embedding_provider: EmbeddingProvider,
    settings,
) -> list[dict]:
    ensure_namespace_exists(session, payload.namespace)

    top_k = payload.top_k if payload.top_k is not None else settings.default_top_k
    top_k = min(max(top_k, 1), 50)

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

    if payload.scope == "knowledge":
        score = KnowledgeItem.embedding.cosine_distance(query_embedding).label("score")
        stmt = (
            select(KnowledgeItem, score)
            .where(KnowledgeItem.namespace == payload.namespace)
            .where(KnowledgeItem.embedding.is_not(None))
            .order_by(score)
            .limit(top_k)
        )
        if project_id:
            stmt = stmt.where(KnowledgeItem.project_id == project_id)
        if entity_id:
            stmt = stmt.where(KnowledgeItem.entity_id == entity_id)

        rows = session.execute(stmt).all()
        return [
            {
                "id": str(item.id),
                "scope": "knowledge",
                "score": float(score_value),
                "snippet": _clip(item.content, settings.max_snippet_chars),
                "tags": list(item.tags or []),
                "created_at": item.created_at,
            }
            for item, score_value in rows
        ]

    if payload.scope == "requirements":
        score = Requirement.embedding.cosine_distance(query_embedding).label("score")
        stmt = (
            select(Requirement, score)
            .where(Requirement.namespace == payload.namespace)
            .where(Requirement.embedding.is_not(None))
            .order_by(score)
            .limit(top_k)
        )
        if project_id:
            stmt = stmt.where(Requirement.project_id == project_id)
        if entity_id:
            stmt = stmt.where(Requirement.owner_entity_id == entity_id)

        rows = session.execute(stmt).all()
        return [
            {
                "id": str(item.id),
                "scope": "requirements",
                "score": float(score_value),
                "snippet": _clip(item.context_snippet or item.body or item.title, settings.max_snippet_chars),
                "tags": list(item.tags or []),
                "created_at": item.created_at,
            }
            for item, score_value in rows
        ]

    score = Event.embedding.cosine_distance(query_embedding).label("score")
    stmt = (
        select(Event, score)
        .where(Event.namespace == payload.namespace)
        .where(Event.embedding.is_not(None))
        .order_by(score)
        .limit(top_k)
    )
    if project_id:
        stmt = stmt.where(Event.project_id == project_id)
    if entity_id:
        stmt = stmt.where(Event.agent_id == entity_id)

    rows = session.execute(stmt).all()
    return [
        {
            "id": str(item.id),
            "scope": "events",
            "score": float(score_value),
            "snippet": _clip(item.body or item.context_snippet or item.title, settings.max_snippet_chars),
            "tags": list(item.tags or []),
            "created_at": item.created_at,
        }
        for item, score_value in rows
    ]


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

    def embed_source_text(text: str) -> list[float] | None:
        if not text.strip():
            return None
        emb = embedding_provider.embed_text(text)
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
