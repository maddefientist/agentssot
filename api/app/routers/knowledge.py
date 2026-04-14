from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select, and_, func
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import KnowledgeItem, MemoryCategory, ContentLayer, ApiRole
from ..settings import get_settings
from ..schemas import (
    TieredKnowledgeCreate,
    TieredKnowledgeResponse,
    TieredRecallRequest,
    TieredRecallResponse,
    TieredRecallResult,
)
from ..security import AuthContext, ensure_namespace_access, require_api_key
from ..synthesis.summary_generator import generate_tiered_summaries
from .. import wal

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _map_memory_type(category: str | None, memory_type: str | None) -> str | None:
    """Map legacy memory_type to category if category not specified."""
    if category is not None:
        return category
    if memory_type is None:
        return None
    type_to_category = {
        "preference": "user_preferences",
        "decision": "user_events",
        "skill": "agent_skills",
        "fact": "user_entities",
    }
    return type_to_category.get(memory_type)


@router.post("/ingest", response_model=TieredKnowledgeResponse)
async def ingest_tiered(
    data: TieredKnowledgeCreate,
    request: Request,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Ingest knowledge with optional tiered content."""
    namespace = data.namespace or "default"
    ensure_namespace_access(auth, namespace, {ApiRole.writer.value, ApiRole.admin.value})

    # Generate embedding for full content
    embedding = None
    embedding_provider = request.app.state.embedding_provider
    if embedding_provider and embedding_provider.is_available:
        try:
            embedding = embedding_provider.embed_text(data.content)
        except Exception:
            embedding = None

    # Map memory_type to category if category not specified
    category_value = _map_memory_type(data.category, data.memory_type)
    category_enum = MemoryCategory(category_value) if category_value else None

    # Semantic dedup: if this item is near-identical to an existing one in the
    # same namespace, return the existing record instead of inserting. Verbatim
    # items bypass — the user explicitly asked for exact-text preservation, so
    # we never collapse them.
    dedup_threshold = get_settings().semantic_dedup_threshold
    if embedding is not None and dedup_threshold > 0 and not data.verbatim:
        dist_col = KnowledgeItem.embedding.cosine_distance(embedding).label("distance")
        dup_stmt = (
            select(KnowledgeItem, dist_col)
            .where(
                and_(
                    KnowledgeItem.namespace == namespace,
                    KnowledgeItem.embedding.isnot(None),
                )
            )
            .order_by(dist_col)
            .limit(1)
        )
        dup_row = session.execute(dup_stmt).first()
        if dup_row is not None:
            existing = dup_row[0]
            similarity = max(0.0, 1.0 - dup_row.distance)
            if similarity >= dedup_threshold:
                wal.log_event(
                    "knowledge.dedup_hit",
                    namespace=namespace,
                    actor_key_id=auth.key_id,
                    payload={"content_preview": data.content[:200]},
                    result={
                        "existing_id": str(existing.id),
                        "similarity": round(similarity, 4),
                        "threshold": dedup_threshold,
                    },
                )
                return existing

    # Auto-generate abstract/summary if requested. Verbatim mode suppresses
    # all LLM-derived summaries; caller-supplied abstract/summary are also
    # dropped because they would be indistinguishable from future auto-gen
    # and defeat the truth-critical guarantee.
    if data.verbatim:
        abstract = None
        summary = None
    else:
        abstract = data.abstract
        summary = data.summary

        if data.generate_summaries and (not abstract or not summary):
            llm_provider = getattr(request.app.state, 'llm_provider', None)
            gen_abstract, gen_summary = await generate_tiered_summaries(data.content, llm_provider)
            if not abstract and gen_abstract:
                abstract = gen_abstract
            if not summary and gen_summary:
                summary = gen_summary

    # Persist layer=full because the record always stores full content;
    # abstract/summary are optional metadata (not separate layers).
    layer = ContentLayer.full

    ki = KnowledgeItem(
        namespace=namespace,
        content=data.content,
        abstract=abstract,
        summary=summary,
        category=category_enum,
        layer=layer,
        source=data.source,
        source_ref=data.source_ref,
        tags=data.tags,
        memory_type=data.memory_type,
        embedding=embedding,
        verbatim=data.verbatim,
    )

    session.add(ki)
    session.commit()
    session.refresh(ki)

    wal.log_event(
        "knowledge.ingest",
        namespace=namespace,
        actor_key_id=auth.key_id,
        payload=data.model_dump(),
        result={"id": str(ki.id), "verbatim": ki.verbatim, "layer": ki.layer.value},
    )

    return ki


@router.post("/recall", response_model=TieredRecallResponse)
async def recall_tiered(
    data: TieredRecallRequest,
    request: Request,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Semantic recall with tiered content and category filtering."""
    namespace = data.namespace or "default"
    ensure_namespace_access(auth, namespace, {ApiRole.reader.value, ApiRole.writer.value, ApiRole.admin.value})

    embedding_provider = request.app.state.embedding_provider
    if not embedding_provider or not embedding_provider.is_available:
        raise HTTPException(
            status_code=400,
            detail="Embedding provider unavailable. Set EMBEDDING_PROVIDER=ollama or openai.",
        )

    # Generate embedding for query
    try:
        query_embedding = embedding_provider.embed_text(data.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

    # Build query with category filter
    conditions = [KnowledgeItem.namespace == namespace]

    if data.categories:
        category_enums = [MemoryCategory(c) for c in data.categories]
        conditions.append(KnowledgeItem.category.in_(category_enums))

    # Exclude rows with null embeddings before ordering to avoid DB/runtime issues
    conditions.append(KnowledgeItem.embedding.isnot(None))

    # Select with explicit cosine_distance label so we can convert to a meaningful score
    dist_col = KnowledgeItem.embedding.cosine_distance(query_embedding).label("distance")
    stmt = (
        select(KnowledgeItem, dist_col)
        .where(and_(*conditions))
        .order_by(dist_col)
        .limit(data.limit)
    )

    result = session.execute(stmt)
    rows = result.all()

    # Build response based on layer preference
    results = []
    for row in rows:
        item = row[0]          # KnowledgeItem
        distance = row.distance  # float cosine distance
        score = max(0.0, 1.0 - distance)  # convert distance → similarity score

        # Determine which content to return based on preference
        content_to_return = item.content
        layer_used = ContentLayer.full
        if data.layer_preference == "abstract" and item.abstract:
            content_to_return = item.abstract
            layer_used = ContentLayer.abstract
        elif data.layer_preference == "summary" and item.summary:
            content_to_return = item.summary
            layer_used = ContentLayer.summary

        results.append(
            TieredRecallResult(
                id=item.id,
                category=item.category.value if item.category else None,
                layer=layer_used.value,
                content=content_to_return,
                abstract=item.abstract,
                summary=item.summary,
                full_content=item.content,
                score=score,
                tags=item.tags or [],
                source=item.source,
                source_ref=item.source_ref,
            )
        )

    return TieredRecallResponse(
        results=results,
        query=data.query,
        total=len(results),
        layer_used=data.layer_preference,
    )


@router.get("/categories")
async def list_categories(
    auth: AuthContext = Depends(require_api_key),
):
    """List available memory categories. Requires valid API key."""
    return {
        "categories": [c.value for c in MemoryCategory],
        "categories_by_domain": {
            "user": ["user_profile", "user_preferences", "user_entities", "user_events"],
            "agent": ["agent_patterns", "agent_tools", "agent_skills", "agent_cases"],
        },
    }
