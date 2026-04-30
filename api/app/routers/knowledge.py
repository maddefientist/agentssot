from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select, and_, func, or_
from sqlalchemy.orm import Session
from uuid import UUID

from ..db import get_session
from ..models import KnowledgeItem, MemoryCategory, ContentLayer, ApiRole, Entity
from ..settings import get_settings
from ..schemas import (
    TieredKnowledgeCreate,
    TieredKnowledgeResponse,
    TieredRecallRequest,
    TieredRecallResponse,
    TieredRecallResult,
    BucketedRecallRequest, BucketedRecallResponse, BucketedRecallItem,
    BucketedRecallDiagnostics, ExpandResponse, LoadoutRequest, LoadoutResponse,
    LoadoutItem, DEFAULT_RECALL_TIERS, DEFAULT_TOP_PER_TIER,
    ContentLayerLiteral,
)
from ..security import AuthContext, ensure_namespace_access, require_api_key
from ..synthesis.summary_generator import generate_tiered_summaries
from .. import wal
from ..reranker import build_reranker_pair, pick_reranker
from ..services.loadout import (
    resolve_cwd_entities, fetch_loadout_candidates, pack_loadout, loadout_cache_key,
)

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


@router.post("/recall")
async def recall_dispatch(
    payload: dict,
    request: Request,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Recall dispatcher.

    - bucketed=False (default, for backwards compat): delegates to the existing
      flat TieredRecallResponse path.
    - bucketed=True: returns tier-bucketed response with per-tier reranking
      and diagnostics.
    """
    if payload.get("bucketed") is True:
        req = BucketedRecallRequest(**{k: v for k, v in payload.items() if k != "bucketed"})
        return await _recall_bucketed(req, request, session, auth)
    # Legacy flat path
    legacy = TieredRecallRequest(**{k: v for k, v in payload.items() if k != "bucketed"})
    return await recall_tiered(legacy, request, session, auth)


async def recall_tiered(
    data: TieredRecallRequest,
    request: Request,
    session: Session,
    auth: AuthContext,
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


async def _recall_bucketed(
    data: BucketedRecallRequest,
    request: Request,
    session: Session,
    auth: AuthContext,
) -> BucketedRecallResponse:
    import time
    from sqlalchemy import select, and_, or_, func
    from datetime import datetime, timezone

    namespace = data.namespace or "default"
    ensure_namespace_access(
        auth, namespace,
        {ApiRole.reader.value, ApiRole.writer.value, ApiRole.admin.value},
    )

    embedding_provider = request.app.state.embedding_provider
    if not embedding_provider or not embedding_provider.is_available:
        raise HTTPException(status_code=400, detail="Embedding provider unavailable.")

    try:
        t0 = time.perf_counter()
        query_embedding = embedding_provider.embed_text(data.query)
        vec_ms = int((time.perf_counter() - t0) * 1000)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

    now = datetime.now(timezone.utc)
    base_filters = [
        KnowledgeItem.namespace == namespace,
        KnowledgeItem.embedding.isnot(None),
    ]
    if not data.include_superseded:
        base_filters.append(KnowledgeItem.superseded_by.is_(None))
    if not data.include_expired:
        base_filters.append(
            or_(KnowledgeItem.expires_at.is_(None), KnowledgeItem.expires_at > now)
        )

    buckets: dict[str, list[BucketedRecallItem]] = {}
    candidates_per_tier: dict[str, int] = {}

    fast_provider, deep_provider = build_reranker_pair(get_settings())
    reranker_name, reranker = pick_reranker(data.tiers, fast_provider, deep_provider)
    multiplier = get_settings().reranker_candidate_multiplier or 3

    rerank_total_ms = 0
    for tier in data.tiers:
        top_k = data.top_per_tier.get(tier, 5)
        candidate_pool = top_k * multiplier
        dist_col = KnowledgeItem.embedding.cosine_distance(query_embedding).label("distance")
        stmt = (
            select(KnowledgeItem, dist_col)
            .where(and_(*base_filters, KnowledgeItem.memory_type == tier))
            .order_by(dist_col)
            .limit(candidate_pool)
        )
        rows = list(session.execute(stmt))
        candidates_per_tier[tier] = len(rows)
        if not rows:
            buckets[tier] = []
            continue

        items = [r[0] for r in rows]
        scores = [1.0 - float(r[1]) for r in rows]

        # Optional rerank
        if reranker.is_available:
            t1 = time.perf_counter()
            try:
                texts = [it.summary or it.abstract or it.content[:500] for it in items]
                reranked = reranker.rerank(data.query, texts)
                # Reorder items + scores by reranked indices
                paired = sorted(zip(items, reranked), key=lambda p: -p[1])
                items = [p[0] for p in paired][:top_k]
                scores = [float(p[1]) for p in paired][:top_k]
            except Exception:
                items = items[:top_k]
                scores = scores[:top_k]
            rerank_total_ms += int((time.perf_counter() - t1) * 1000)
        else:
            items = items[:top_k]
            scores = scores[:top_k]

        buckets[tier] = [
            BucketedRecallItem(
                id=it.id,
                memory_type=str(it.memory_type) if it.memory_type else "fact",
                abstract=it.abstract,
                summary=it.summary if data.expand_layer in ("summary", "full") else None,
                content=it.content if data.expand_layer == "full" else None,
                score=float(s),
                confidence=float(getattr(it, "confidence", 1.0)),
                entity_refs=list(getattr(it, "entity_refs", []) or []),
                tags=list(it.tags or []),
            )
            for it, s in zip(items, scores)
        ]

    return BucketedRecallResponse(
        buckets=buckets,
        diagnostics=BucketedRecallDiagnostics(
            candidates_per_tier=candidates_per_tier,
            vec_ms=vec_ms,
            rerank_ms=rerank_total_ms,
            reranker_used=reranker_name,
        ),
    )


@router.get("/items/{item_id}/expand", response_model=ExpandResponse)
async def expand_item(
    item_id: UUID,
    layer: ContentLayerLiteral = "full",
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Fetch L1 summary or L2 full content for an item.

    Used when an agent saw an L0 abstract in loadout/recall and needs the
    concrete details. No side effects, idempotent.
    """
    from sqlalchemy import select
    item = session.execute(
        select(KnowledgeItem).where(KnowledgeItem.id == item_id)
    ).scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    ensure_namespace_access(
        auth, item.namespace,
        {ApiRole.reader.value, ApiRole.writer.value, ApiRole.admin.value},
    )
    return ExpandResponse(
        id=item.id,
        layer=layer,
        abstract=item.abstract,
        summary=item.summary if layer in ("summary", "full") else None,
        content=item.content if layer == "full" else None,
    )


@router.post("/loadout", response_model=LoadoutResponse)
async def compute_loadout(
    data: LoadoutRequest,
    request: Request,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Compute the cwd-aware loadout bundle for the caller.

    Used by SessionStart hook (Plan 2) and by the operator for debugging
    via the Cortex /loadout page (Plan 2). Callable mid-session by an
    agent post-compaction to restore push context.
    """
    from sqlalchemy import select
    from app.models import Entity

    namespace = data.namespace or "claude-shared"
    ensure_namespace_access(
        auth, namespace,
        {ApiRole.reader.value, ApiRole.writer.value, ApiRole.admin.value},
    )

    # 1. Resolve entities for this cwd
    ents = list(session.execute(select(Entity)).scalars())
    ent_dicts = [
        {"id": str(e.id), "slug": e.slug, "cwd_hints": (e.meta or {}).get("cwd_hints", [])}
        for e in ents
    ]
    matched = resolve_cwd_entities(data.cwd, ent_dicts)
    entity_ids = [e["id"] for e in matched]

    # 2. Fetch candidates (rules + entity-linked items)
    candidates = fetch_loadout_candidates(session, namespace, entity_ids, data.device_id)

    # 3. Convert to dicts for packer
    item_dicts = [
        {
            "id": str(c.id),
            "memory_type": str(c.memory_type) if c.memory_type else "fact",
            "abstract": c.abstract or (c.content[:120] if c.content else ""),
            "title": (c.source or (c.tags[0] if c.tags else ""))[:60],
            "priority": int(c.loadout_priority or 0),
        }
        for c in candidates
    ]

    # 4. Pack to budget
    packed, overflow, used = pack_loadout(item_dicts, data.token_budget)

    # 5. Group by tier for response
    items_by_tier: dict[str, list[LoadoutItem]] = {}
    for it in packed:
        tier = it["memory_type"]
        items_by_tier.setdefault(tier, []).append(LoadoutItem(
            id=UUID(it["id"]),
            memory_type=tier,
            abstract=it["abstract"],
            title=it["title"],
            priority=it["priority"],
        ))

    cache_key = loadout_cache_key(
        data.cwd, data.device_id, [it["id"] for it in packed]
    )
    return LoadoutResponse(
        items=items_by_tier,
        overflow_count=overflow,
        tokens_used=used,
        cache_key=cache_key,
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
