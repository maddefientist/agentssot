import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError
from sqlalchemy import select, and_, func, or_
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from uuid import UUID

from ..db import get_session
from ..llm.classifier import classify
from ..llm.layer_compute import compute_layers
from ..services.lifecycle import find_supersession_candidates, apply_supersession, soft_expire
from ..services.contradiction import detect_contradictions
from ..services.review_queue import list_pending as rq_list, resolve as rq_resolve, dismiss as rq_dismiss
from ..models import (
    KnowledgeItem, MemoryCategory, ContentLayer, ApiRole, Entity,
    ReviewQueueItem, ReviewQueueKind, ReviewQueueStatus,
)
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
    ReviewQueueItemOut, SupersedeRequest, ExpireRequest, PromoteRequest,
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

    # Plan 1 T2.3: auto-classify if caller didn't provide explicit type/abstract/summary
    settings = get_settings()
    classifier_out: dict | None = None
    needs_review = False
    if (data.abstract is None and data.summary is None and not data.verbatim):
        classifier_out = await asyncio.to_thread(classify, data.content, tags=data.tags, hint=data.memory_type)
        if classifier_out.get("confidence", 0.0) < settings.classifier_min_confidence:
            needs_review = True
        # If caller didn't pin a memory_type, accept classifier's decision
        if data.memory_type is None and classifier_out.get("confidence", 0.0) >= settings.classifier_min_confidence:
            data.memory_type = classifier_out.get("memory_type")

    layers = compute_layers(data.content, classifier_out)
    # Compose layer fields onto the persisted record
    abstract_to_store = data.abstract or layers["abstract"]
    summary_to_store = data.summary or layers["summary"]

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
        abstract_to_store = None
        summary_to_store = None
    else:
        abstract = data.abstract
        summary = data.summary
        abstract_to_store = abstract or layers["abstract"]
        summary_to_store = summary or layers["summary"]

        if data.generate_summaries and (not abstract or not summary):
            llm_provider = getattr(request.app.state, 'llm_provider', None)
            gen_abstract, gen_summary = await generate_tiered_summaries(data.content, llm_provider)
            if not abstract and gen_abstract:
                abstract = gen_abstract
                abstract_to_store = abstract or layers["abstract"]
            if not summary and gen_summary:
                summary = gen_summary
                summary_to_store = summary or layers["summary"]

    # Persist layer=full because the record always stores full content;
    # abstract/summary are optional metadata (not separate layers).
    layer = ContentLayer.full

    # Ensure every persisted item has a non-null memory_type
    if data.memory_type is None:
        data.memory_type = 'fact'

    ki = KnowledgeItem(
        namespace=namespace,
        content=data.content,
        abstract=abstract_to_store,
        summary=summary_to_store,
        category=category_enum,
        layer=layer,
        source=data.source,
        source_ref=data.source_ref,
        tags=data.tags,
        memory_type=data.memory_type,
        embedding=embedding,
        verbatim=data.verbatim,
        confidence=float(classifier_out.get("confidence", 1.0)) if classifier_out else 1.0,
        cwd_hints=(list(classifier_out.get("cwd_hints", []) or [])[:50]) if classifier_out else [],
        device_hints=(list(classifier_out.get("device_hints", []) or [])[:50]) if classifier_out else [],
        last_classified_at=datetime.now(timezone.utc) if classifier_out else None,
    )

    session.add(ki)
    session.flush()
    session.refresh(ki)

    if needs_review:
        rq = ReviewQueueItem(
            namespace=namespace,
            kind=ReviewQueueKind.low_conf,
            priority=10,
            primary_id=ki.id,
            reason=f"classifier_confidence={classifier_out.get('confidence', 0.0):.2f}; reason={classifier_out.get('_reason','low_conf')}",
            status=ReviewQueueStatus.pending,
        )
        session.add(rq)

    session.commit()

    # Plan 1 T2.5: supersession + contradiction scans
    new_entity_refs: list[str] = []
    if classifier_out:
        new_entity_refs = list(classifier_out.get("entity_mentions") or [])
    if new_entity_refs:
        ki.entity_refs = new_entity_refs
        session.commit()

    if ki.memory_type and new_entity_refs:
        cand_stmt = (
            select(KnowledgeItem)
            .where(
                KnowledgeItem.namespace == namespace,
                KnowledgeItem.memory_type == ki.memory_type,
                KnowledgeItem.id != ki.id,
                KnowledgeItem.superseded_by.is_(None),
                KnowledgeItem.entity_refs.op("?|")(new_entity_refs),
            )
            .limit(20)
        )
        candidates = list(session.execute(cand_stmt).scalars())
        superseded = find_supersession_candidates(ki, candidates)
        for old in superseded:
            apply_supersession(old, ki)
            session.add(ReviewQueueItem(
                namespace=namespace,
                kind=ReviewQueueKind.supersede,
                priority=5,
                primary_id=ki.id,
                secondary_id=old.id,
                reason="auto-supersession on entity+type match",
                status=ReviewQueueStatus.pending,
            ))
        if superseded:
            session.commit()

    if str(ki.memory_type) in ("command", "skill") and new_entity_refs:
        rule_stmt = (
            select(KnowledgeItem)
            .where(
                KnowledgeItem.namespace == namespace,
                KnowledgeItem.memory_type == "rule",
                KnowledgeItem.entity_refs.op("?|")(new_entity_refs),
                KnowledgeItem.superseded_by.is_(None),
            )
        )
        rules = list(session.execute(rule_stmt).scalars())
        contras = detect_contradictions(
            new_type=str(ki.memory_type),
            new_entity_refs=new_entity_refs,
            existing_rules=rules,
        )
        for rule in contras:
            session.add(ReviewQueueItem(
                namespace=namespace,
                kind=ReviewQueueKind.contradiction,
                priority=20,
                primary_id=ki.id,
                secondary_id=rule.id,
                reason=f"new {ki.memory_type} contradicts negation rule",
                status=ReviewQueueStatus.pending,
            ))
        if contras:
            session.commit()

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
    try:
        # Default: bucketed=True (bucketed path). Only bucketed=False goes legacy.
        if payload.get("bucketed") is not False:
            req = BucketedRecallRequest(**{k: v for k, v in payload.items()})
            return await _recall_bucketed(req, request, session, auth)
        # Legacy flat path — caller explicitly set bucketed=False
        legacy = TieredRecallRequest(**{k: v for k, v in payload.items() if k != "bucketed"})
        return await recall_tiered(legacy, request, session, auth)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())


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

    # Resolve tiers: None with exclude_episodic=True → 5 non-episodic tiers
    tiers = data.tiers
    if tiers is None and data.exclude_episodic:
        tiers = list(DEFAULT_RECALL_TIERS)
    elif tiers is None:
        tiers = list(DEFAULT_RECALL_TIERS) + ["episodic"]

    buckets: dict[str, list[BucketedRecallItem]] = {}
    candidates_per_tier: dict[str, int] = {}

    fast_provider, deep_provider = build_reranker_pair(get_settings())
    reranker_name, reranker = pick_reranker(tiers, fast_provider, deep_provider)
    multiplier = get_settings().reranker_candidate_multiplier or 3

    rerank_total_ms = 0
    for tier in tiers:
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
                entity_refs=[UUID(str(x)) for x in (getattr(it, 'entity_refs', None) or []) if x],
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
    layer: ContentLayerLiteral = "summary",
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Fetch L1 summary or L2 full content for an item.

    Used when an agent saw an L0 abstract in loadout/recall and needs the
    concrete details. No side effects, idempotent.
    """
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

    namespace = data.namespace or "claude-shared"
    ensure_namespace_access(
        auth, namespace,
        {ApiRole.reader.value, ApiRole.writer.value, ApiRole.admin.value},
    )

    # 1. Resolve entities for this cwd
    ents = list(session.execute(select(Entity).where(Entity.namespace == namespace)).scalars())
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


@router.get("/admin/review-queue")
async def get_review_queue(
    namespace: str | None = None,
    kind: str | None = None,
    limit: int = 100,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
) -> list[ReviewQueueItemOut]:
    """List pending review-queue items. Admin-only."""
    if auth.role != ApiRole.admin.value:
        raise HTTPException(status_code=403, detail="admin role required")
    items = rq_list(session, namespace, kind, limit)
    return [ReviewQueueItemOut.model_validate(i, from_attributes=True) for i in items]


@router.post("/admin/review-queue/{queue_id}/resolve")
async def review_queue_resolve(
    queue_id: UUID,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Accept/review-queue item. Admin-only."""
    if auth.role != ApiRole.admin.value:
        raise HTTPException(status_code=403, detail="admin role required")
    item = rq_resolve(session, str(queue_id), by=auth.key_name)
    if item is None:
        raise HTTPException(status_code=404, detail="review queue item not found")
    return {"status": "ok"}


@router.post("/admin/review-queue/{queue_id}/dismiss")
async def review_queue_dismiss(
    queue_id: UUID,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Dismiss/review-queue item. Admin-only."""
    if auth.role != ApiRole.admin.value:
        raise HTTPException(status_code=403, detail="admin role required")
    item = rq_dismiss(session, str(queue_id), by=auth.key_name)
    if item is None:
        raise HTTPException(status_code=404, detail="review queue item not found")
    return {"status": "ok"}


@router.post("/items/{item_id}/supersede")
async def supersede_endpoint(
    item_id: UUID,
    body: SupersedeRequest,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Manually mark an item as superseded by another."""
    old = session.get(KnowledgeItem, item_id)
    new = session.get(KnowledgeItem, body.superseded_by)
    if old is None or new is None:
        raise HTTPException(status_code=404, detail="item not found")
    ensure_namespace_access(auth, old.namespace, {ApiRole.writer.value, ApiRole.admin.value})
    apply_supersession(old, new)
    session.commit()
    return {"status": "ok", "old_id": str(item_id), "new_id": str(body.superseded_by)}


@router.post("/items/{item_id}/expire")
async def expire_endpoint(
    item_id: UUID,
    body: ExpireRequest,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Soft-expire (sets expires_at = now). Item stays in DB for audit."""
    item = session.get(KnowledgeItem, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    ensure_namespace_access(auth, item.namespace, {ApiRole.writer.value, ApiRole.admin.value})
    soft_expire(item, body.reason)
    session.commit()
    return {"status": "ok", "id": str(item_id), "expires_at": item.expires_at.isoformat()}


@router.post("/items/{item_id}/promote")
async def promote_endpoint(
    item_id: UUID,
    body: PromoteRequest,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """Bump loadout_priority."""
    item = session.get(KnowledgeItem, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    ensure_namespace_access(auth, item.namespace, {ApiRole.writer.value, ApiRole.admin.value})
    item.loadout_priority = body.priority
    session.commit()
    return {"status": "ok", "id": str(item_id), "priority": body.priority}
