import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import crud, schemas
from .background import compaction_loop
from .db import get_session
from .embeddings import build_embedding_provider
from .llm import LLMProviderError, build_llm_provider
from .reranker import build_reranker_provider
from .logging_config import configure_logging
from .models import ApiRole
from .security import AuthContext, ensure_namespace_access, require_admin, require_api_key
from .settings import get_settings
from .startup import initialize_system
from .cortex import router as cortex_router

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("agentssot.api")
BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"
PLUGIN_DIR = BASE_DIR / "plugin"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.embedding_provider = build_embedding_provider(settings)
    app.state.llm_provider = build_llm_provider(settings)
    app.state.reranker_provider = build_reranker_provider(settings)

    initialize_system(settings)

    if settings.compaction_enabled and settings.llm_provider == "none":
        logger.warning("COMPACTION_ENABLED=true but LLM_PROVIDER=none, forcing compaction disabled")

    task = None
    if settings.effective_compaction_enabled:
        task = asyncio.create_task(compaction_loop(app), name="compaction-loop")
        logger.info("background compaction loop started")
    else:
        logger.info("background compaction loop disabled")

    app.state.compaction_task = task

    from .synthesis import synthesis_loop as _synthesis_loop

    synthesis_task = None
    if settings.effective_synthesis_enabled:
        synthesis_task = asyncio.create_task(_synthesis_loop(app), name="synthesis-loop")
        logger.info("background synthesis loop started (hour=%d)", settings.synthesis_schedule_hour)
    else:
        logger.info("background synthesis loop disabled")

    app.state.synthesis_task = synthesis_task

    yield

    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    if synthesis_task:
        synthesis_task.cancel()
        try:
            await synthesis_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="AgentSSOT API", version="1.0.0", lifespan=lifespan)
if UI_DIR.exists():
    app.mount("/ui/assets", StaticFiles(directory=UI_DIR), name="ui-assets")


app.include_router(cortex_router)


@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    started = time.perf_counter()
    namespace = request.query_params.get("namespace") or "default"
    route = request.url.path

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        status_code = 500
        logger.exception(
            "request failed",
            extra={
                "route": route,
                "method": request.method,
                "namespace": namespace,
                "status_code": status_code,
            },
        )
        raise

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "request completed",
        extra={
            "route": route,
            "method": request.method,
            "namespace": namespace,
            "status_code": status_code,
            "latency_ms": latency_ms,
        },
    )
    return response


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "embedding_provider": settings.embedding_provider,
        "embedding_available": app.state.embedding_provider.is_available,
        "llm_provider": settings.llm_provider,
        "llm_available": app.state.llm_provider.is_available,
        "compaction_enabled": settings.effective_compaction_enabled,
        "reranker_provider": settings.reranker_provider,
        "reranker_available": app.state.reranker_provider.is_available,
        "synthesis_enabled": settings.effective_synthesis_enabled,
    }


@app.get("/", include_in_schema=False)
def ui_home():
    index_file = UI_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return RedirectResponse(url="/docs")


@app.get("/cortex", include_in_schema=False)
def ui_cortex():
    cortex_file = UI_DIR / "cortex.html"
    if cortex_file.exists():
        return FileResponse(cortex_file)
    return RedirectResponse(url="/")


@app.get("/cortex/data", include_in_schema=False)
def cortex_data(
    namespace: str = Query(default="claude-shared"),
    session: Session = Depends(get_session),
):
    """Public read-only endpoint for the cortex visualization. Returns concept metadata only."""
    concepts = crud.list_concepts(session, namespace, limit=1000)
    # Also grab knowledge count for the HUD
    stats = crud.get_namespace_stats(session, namespace)
    return {
        "concepts": concepts,
        "total": len(concepts),
        "knowledge_count": stats.get("knowledge_items", {}).get("total", 0),
    }


@app.get("/cortex/links", include_in_schema=False)
def cortex_links(
    namespace: str = Query(default="claude-shared"),
    limit: int = Query(default=200, le=500),
    session: Session = Depends(get_session),
):
    """Public read-only endpoint for cortex edges. Returns concept_links."""
    return {"links": crud.list_concept_links(session, namespace, limit)}


@app.get("/cortex/system-info", include_in_schema=False)
def cortex_system_info(
    namespace: str = Query(default="claude-shared"),
    session: Session = Depends(get_session),
):
    """Public read-only endpoint for cortex drawer. Returns config, health, agents."""
    settings = app.state.settings

    health = {
        "status": "ok",
        "embedding_available": app.state.embedding_provider.is_available,
        "llm_available": app.state.llm_provider.is_available,
        "reranker_available": app.state.reranker_provider.is_available,
        "synthesis_enabled": settings.effective_synthesis_enabled,
    }

    config = {
        "namespace": namespace,
        "embedding_model": settings.ollama_embed_model,
        "embedding_dim": settings.embedding_dim,
        "llm_model": settings.ollama_chat_model,
        "reranker_model": settings.ollama_reranker_model,
        "synthesis_model": settings.synthesis_model,
        "synthesis_schedule_hour": settings.synthesis_schedule_hour,
        "synthesis_similarity_threshold": settings.synthesis_similarity_threshold,
        "synthesis_min_cluster_size": settings.synthesis_min_cluster_size,
        "synthesis_confidence_decay": settings.synthesis_confidence_decay,
        "synthesis_decay_floor": settings.synthesis_decay_floor,
        "synthesis_decay_grace_days": settings.synthesis_decay_grace_days,
    }

    from .models import AgentProfile
    profiles = session.scalars(
        select(AgentProfile).where(AgentProfile.namespace == namespace)
    ).all()
    agents = [
        {
            "agent_key": p.agent_key,
            "device_name": p.device_name,
            "strengths": list(p.strengths or []),
            "total_recalls": p.total_recalls,
            "total_feedback": p.total_feedback,
            "updated_at": p.updated_at.isoformat() if p.updated_at else None,
        }
        for p in profiles
    ]

    from .models import Namespace as NsModel
    ns_names = [ns.name for ns in session.scalars(select(NsModel)).all()]

    return {
        "health": health,
        "config": config,
        "agents": agents,
        "namespaces": ns_names,
    }


@app.get("/cortex/activity", include_in_schema=False)
def cortex_activity(
    namespace: str = Query(default="claude-shared"),
    limit: int = Query(default=50, le=100),
    session: Session = Depends(get_session),
):
    """Public read-only endpoint for cortex activity ticker. Returns recent events."""
    from .models import RecallEvent, ConceptFeedback, Concept

    events = []

    recalls = session.execute(
        select(RecallEvent.agent_key, RecallEvent.query_text, RecallEvent.created_at, Concept.title)
        .join(Concept, RecallEvent.concept_id == Concept.id)
        .where(RecallEvent.namespace == namespace)
        .order_by(RecallEvent.created_at.desc())
        .limit(limit)
    ).all()
    for r in recalls:
        events.append({
            "type": "recall",
            "agent": r.agent_key,
            "detail": (r.query_text or "")[:80],
            "concept": r.title,
            "at": r.created_at.isoformat(),
        })

    feedbacks = session.execute(
        select(ConceptFeedback.agent_key, ConceptFeedback.signal, ConceptFeedback.created_at, Concept.title)
        .join(Concept, ConceptFeedback.concept_id == Concept.id)
        .where(ConceptFeedback.namespace == namespace)
        .order_by(ConceptFeedback.created_at.desc())
        .limit(limit)
    ).all()
    for f in feedbacks:
        events.append({
            "type": "feedback",
            "agent": f.agent_key,
            "detail": f.signal.value if hasattr(f.signal, 'value') else str(f.signal),
            "concept": f.title,
            "at": f.created_at.isoformat(),
        })

    events.sort(key=lambda e: e["at"], reverse=True)
    return {"events": events[:limit]}



@app.get("/dashboard/stats", include_in_schema=False)
def dashboard_stats(
    namespace: str = Query(default="claude-shared"),
    session: Session = Depends(get_session),
):
    """Lightweight stats endpoint for Homepage dashboard widget. No auth required."""
    from sqlalchemy import func, text
    from .models import Concept, KnowledgeItem, RecallEvent, ConceptFeedback

    concepts_total = session.scalar(select(func.count()).select_from(Concept).where(Concept.namespace == namespace)) or 0
    skills = session.scalar(select(func.count()).select_from(Concept).where(Concept.namespace == namespace, Concept.type == "skill")) or 0
    knowledge = session.scalar(select(func.count()).select_from(KnowledgeItem).where(KnowledgeItem.namespace == namespace)) or 0
    recalls_24h = session.scalar(
        select(func.count()).select_from(RecallEvent)
        .where(RecallEvent.namespace == namespace, RecallEvent.created_at > text("now() - interval '24 hours'"))
    ) or 0
    ingested_24h = session.scalar(
        select(func.count()).select_from(KnowledgeItem)
        .where(KnowledgeItem.namespace == namespace, KnowledgeItem.created_at > text("now() - interval '24 hours'"))
    ) or 0
    avg_conf = session.scalar(
        select(func.avg(Concept.confidence)).where(Concept.namespace == namespace, Concept.confidence.isnot(None))
    ) or 0

    return {
        "concepts": concepts_total,
        "skills": skills,
        "knowledge": knowledge,
        "recalls_24h": recalls_24h,
        "ingested_24h": ingested_24h,
        "avg_confidence": float(avg_conf),
    }

@app.post("/feedback", response_model=schemas.FeedbackResponse)
def submit_feedback(
    payload: schemas.FeedbackRequest,
    namespace: str = Query(default="claude-shared"),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    ensure_namespace_access(auth, namespace, {ApiRole.reader.value, ApiRole.writer.value, ApiRole.admin.value})
    from uuid import UUID

    # Route to knowledge item feedback if knowledge_item_id is provided
    if payload.knowledge_item_id:
        try:
            result = crud.create_knowledge_feedback(
                session=session,
                namespace=namespace,
                signal=payload.signal,
                agent_key=payload.agent_key or auth.key_name,
                knowledge_item_id=UUID(payload.knowledge_item_id),
                note=payload.note,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        session.commit()
        return schemas.FeedbackResponse(
            knowledge_item_id=result["knowledge_item_id"],
            signal=result["signal"],
            strength=result["strength"],
        )

    try:
        result = crud.create_concept_feedback(
            session=session,
            namespace=namespace,
            signal=payload.signal,
            agent_key=payload.agent_key or auth.key_name,
            embedding_provider=app.state.embedding_provider,
            concept_id=UUID(payload.concept_id) if payload.concept_id else None,
            query=payload.query,
            session_id=payload.session_id,
            note=payload.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    session.commit()
    return schemas.FeedbackResponse(**result)


@app.post("/session-complete", response_model=schemas.SessionCompleteResponse)
def session_complete(
    payload: schemas.SessionCompleteRequest,
    namespace: str = Query(default="claude-shared"),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    ensure_namespace_access(auth, namespace, {ApiRole.writer.value, ApiRole.admin.value})

    # 1. Mark recall events as session completed
    completed_count = crud.mark_session_completed(session, payload.session_id)

    # 2. Extract facts via Ollama (zero Claude tokens)
    llm = app.state.llm_provider
    extraction_prompt = (
        "Extract 3-5 key facts from this conversation summary. "
        "Return each fact on its own line. Focus on: decisions made, bugs fixed, "
        "patterns learned, architecture changes. Skip routine actions.\n\n"
        f"Summary:\n{payload.conversation_summary}"
    )
    facts: list[str] = []
    if llm.is_available:
        try:
            raw = llm.summarize(extraction_prompt)
            facts = [line.strip() for line in raw.strip().split("\n") if line.strip() and len(line.strip()) > 10]
        except Exception:
            facts = []

    # 3. Ingest extracted facts
    from . import models as _models
    agent_key = payload.agent_key or auth.key_name
    device_name = agent_key.replace("device-", "").replace("-writer", "") if agent_key else "unknown"
    for fact in facts:
        embedding = None
        if app.state.embedding_provider.is_available:
            try:
                embedding = app.state.embedding_provider.embed_text(fact)
            except Exception:
                embedding = None
        session.add(_models.KnowledgeItem(
            namespace=namespace,
            content=fact,
            source=agent_key,
            tags=["session-extract", f"device-{device_name}", "auto-extracted"],
            embedding=embedding,
        ))

    # 4. Update agent profile from session activity
    if agent_key:
        crud.update_profile_from_recall(session, agent_key, namespace)

    session.commit()

    return schemas.SessionCompleteResponse(
        session_id=payload.session_id,
        facts_extracted=len(facts),
        recall_events_completed=completed_count,
    )


@app.get("/agent-profile/{agent_key}", response_model=schemas.AgentProfileResponse)
def get_agent_profile_endpoint(
    agent_key: str,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    profile = crud.get_agent_profile(session, agent_key)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@app.get("/onboarding", response_class=PlainTextResponse, include_in_schema=False)
def onboarding_public() -> str:
    """Public onboarding page — no auth required. Tells new agents what this service is and how to enroll."""
    return "\n".join(
        [
            "AgentSSOT — Cross-LLM Shared Memory Service",
            "",
            "What is this?",
            "- AgentSSOT stores durable knowledge, events, and requirements for AI agents.",
            "- It provides keyword search (GET /query) and semantic vector recall (POST /recall).",
            "- Multiple agents share memory through namespace-based isolation.",
            "",
            "How to get an API key:",
            "",
            "  Option A — Enrollment token (self-service):",
            "    If you have been given an enrollment token (starts with ssot_enroll_),",
            "    call POST /enroll with your token to receive an API key:",
            "",
            '    curl -X POST http://<this-host>:8088/enroll \\',
            "      -H 'Content-Type: application/json' \\",
            '      -d \'{"token":"ssot_enroll_...", "name":"my-agent-name"}\'',
            "",
            "    The response contains your API key (starts with ssot_). Save it — it is shown once.",
            "",
            "  Option B — Ask an admin:",
            "    Request a key via the dashboard (/ -> Admin: API Keys) or POST /admin/api-keys.",
            "",
            "Once you have a key:",
            "- Set header: X-API-Key: <your-key>",
            "- Visit GET /onboarding/me for a personalized guide showing your role and namespaces.",
            "",
            "Endpoints (public):",
            "- GET  /health",
            "- GET  /onboarding",
            "- POST /enroll (requires enrollment token in body)",
            "",
            "Endpoints (authenticated):",
            "- GET  /onboarding/me (personalized guide)",
            "- GET  /query",
            "- POST /recall",
            "- POST /ingest (writer/admin)",
            "- POST /summarize_clear (writer/admin)",
            "",
            "Full API docs: GET /docs",
            "",
        ]
    )


@app.get("/onboarding/me", response_class=PlainTextResponse, include_in_schema=False)
def onboarding_me(auth: AuthContext = Depends(require_api_key)) -> str:
    """Personalized onboarding — requires API key. Shows role, namespaces, and operational guide."""
    s = app.state.settings
    base_url_hint = "http://<agentssot-host>:8088"
    namespaces = ", ".join(auth.namespaces) if auth.namespaces else "(none)"

    return "\n".join(
        [
            "AgentSSOT — Your Personalized Guide",
            "",
            "Auth Context (your key):",
            f"- role: {auth.role}",
            f"- namespaces: {namespaces}",
            "",
            "Connection:",
            f"- BASE_URL: {base_url_hint}",
            "- Header: X-API-Key: <your-key>",
            "",
            "Core Rules:",
            "- Always set namespace explicitly on every request.",
            "- Never read/write namespaces your key does not allow.",
            "- Prefer private namespaces; use shared namespaces only for intentional sharing.",
            "- Keep memory atomic and tagged. Do not dump full transcripts.",
            "",
            "Recommended Loop (every task):",
            "1) Start: GET /query and POST /recall in the relevant namespace(s).",
            "2) During: POST /ingest events for decisions/directives/results.",
            "3) End: POST /summarize_clear to archive events and store a summary.",
            "",
            "Recall Notes:",
            f"- EMBEDDING_PROVIDER={s.embedding_provider}. If 'none', provide query_embedding.",
            f"- DEFAULT_TOP_K={s.default_top_k}. Keep Top-K small to stay token-efficient.",
            "",
            "Ingest Notes:",
            "- Knowledge items are auto-chunked to max ~800 chars per row.",
            "- If you provide an embedding, the server will not recompute it.",
            "",
            "Full API docs: GET /docs",
            "",
        ]
    )


@app.post("/ingest", response_model=schemas.IngestResponse)
def ingest(
    payload: schemas.IngestRequest,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    namespace = payload.namespace or "default"
    payload.namespace = namespace

    ensure_namespace_access(auth, namespace, {ApiRole.writer.value, ApiRole.admin.value})

    counts = crud.ingest_batch(
        session=session,
        payload=payload,
        embedding_provider=app.state.embedding_provider,
        settings=app.state.settings,
    )
    return schemas.IngestResponse(namespace=namespace, counts=counts)


@app.get("/query", response_model=schemas.QueryResponse)
def query(
    q: str = Query(default=""),
    namespace: str = Query(default="default"),
    project_slug: str | None = Query(default=None),
    entity_slug: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    ensure_namespace_access(auth, namespace, {ApiRole.reader.value, ApiRole.writer.value, ApiRole.admin.value})

    records = crud.query_records(
        session=session,
        namespace=namespace,
        q=q,
        project_slug=project_slug,
        entity_slug=entity_slug,
        limit=limit,
        max_snippet_chars=app.state.settings.max_snippet_chars,
    )

    return schemas.QueryResponse(namespace=namespace, total=len(records), results=records)


@app.post("/recall", response_model=schemas.RecallResponse)
def recall(
    payload: schemas.RecallRequest,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    ensure_namespace_access(auth, payload.namespace, {ApiRole.reader.value, ApiRole.writer.value, ApiRole.admin.value})

    items = crud.recall(
        session=session,
        payload=payload,
        embedding_provider=app.state.embedding_provider,
        reranker_provider=app.state.reranker_provider,
        settings=app.state.settings,
    )

    # Commit tracking updates (knowledge recall counts + concept recall events)
    session.commit()

    top_k = payload.top_k if payload.top_k is not None else app.state.settings.default_top_k
    top_k = min(max(top_k, 1), 50)

    return schemas.RecallResponse(namespace=payload.namespace, scope=payload.scope, top_k=top_k, items=items)


@app.post("/summarize_clear", response_model=schemas.SummarizeClearResponse)
def summarize_clear(
    payload: schemas.SummarizeClearRequest,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    ensure_namespace_access(auth, payload.namespace, {ApiRole.writer.value, ApiRole.admin.value})

    if not app.state.llm_provider.is_available:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=app.state.llm_provider.unavailable_reason or "LLM provider unavailable",
        )

    try:
        result = crud.summarize_and_archive_session(
            session=session,
            namespace=payload.namespace,
            session_id=payload.session_id,
            project_slug=payload.project_slug,
            llm_provider=app.state.llm_provider,
            embedding_provider=app.state.embedding_provider,
            settings=app.state.settings,
            max_events=payload.max_events,
        )
    except LLMProviderError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return schemas.SummarizeClearResponse(**result)


@app.post("/admin/namespaces", response_model=schemas.NamespaceCreateResponse)
def admin_create_namespace(
    payload: schemas.NamespaceCreateRequest,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    require_admin(auth)

    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Namespace name is required")

    namespace = crud.create_namespace(session, name)
    return schemas.NamespaceCreateResponse(name=namespace.name, created_at=namespace.created_at)


@app.post("/admin/api-keys", response_model=schemas.ApiKeyCreateResponse)
def admin_create_api_key(
    payload: schemas.ApiKeyCreateRequest,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    require_admin(auth)

    record, plaintext = crud.create_api_key_record(
        session=session,
        name=payload.name,
        role=payload.role,
        namespaces=payload.namespaces,
    )

    return schemas.ApiKeyCreateResponse(
        id=str(record.id),
        name=record.name,
        role=record.role.value,
        namespaces=list(record.namespaces or []),
        is_active=record.is_active,
        created_at=record.created_at,
        api_key=plaintext,
    )


@app.get("/admin/api-keys", response_model=list[schemas.ApiKeyListItem])
def admin_list_api_keys(
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    require_admin(auth)
    rows = crud.list_api_keys_masked(session)
    return [schemas.ApiKeyListItem(**row) for row in rows]


@app.post("/admin/delete-items", response_model=schemas.DeleteItemsResponse)
def admin_delete_items(
    payload: schemas.DeleteItemsRequest,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    require_admin(auth)
    ensure_namespace_access(auth, payload.namespace, {ApiRole.admin.value})

    deleted = crud.delete_items(session, payload.namespace, payload.ids)
    return schemas.DeleteItemsResponse(namespace=payload.namespace, deleted=deleted)


@app.post("/admin/dedup", response_model=schemas.DedupResponse)
def admin_dedup(
    payload: schemas.DedupRequest,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    require_admin(auth)
    ensure_namespace_access(auth, payload.namespace, {ApiRole.admin.value})

    result = crud.dedup_knowledge_items(session, payload.namespace, dry_run=payload.dry_run)
    return schemas.DedupResponse(**result)


@app.get("/admin/stats", response_model=schemas.NamespaceStatsResponse)
def admin_stats(
    namespace: str = Query(default="default"),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    require_admin(auth)
    ensure_namespace_access(auth, namespace, {ApiRole.admin.value})

    result = crud.get_namespace_stats(session, namespace)
    return schemas.NamespaceStatsResponse(**result)


@app.post("/admin/backfill-embeddings", response_model=schemas.BackfillEmbeddingsResponse)
def admin_backfill_embeddings(
    payload: schemas.BackfillEmbeddingsRequest,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    require_admin(auth)
    ensure_namespace_access(auth, payload.namespace, {ApiRole.admin.value})

    result = crud.backfill_embeddings(
        session=session,
        namespace=payload.namespace,
        scope=payload.scope,
        embedding_provider=app.state.embedding_provider,
        settings=app.state.settings,
        limit=payload.limit,
        batch_size=payload.batch_size,
        dry_run=payload.dry_run,
    )
    return schemas.BackfillEmbeddingsResponse(**result)


# ── Concepts ───────────────────────────────────────────────────────


@app.get("/concepts", response_model=schemas.ConceptListResponse)
def list_concepts_endpoint(
    namespace: str = Query(default="default"),
    concept_type: str | None = Query(default=None),
    scope: str | None = Query(default=None),
    include_superseded: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=1000),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    ensure_namespace_access(auth, namespace, {ApiRole.reader.value, ApiRole.writer.value, ApiRole.admin.value})
    concepts = crud.list_concepts(
        session, namespace, concept_type=concept_type, scope=scope,
        include_superseded=include_superseded, limit=limit,
    )
    return schemas.ConceptListResponse(namespace=namespace, total=len(concepts), concepts=concepts)


@app.get("/concepts/{concept_id}", response_model=schemas.ConceptDetailResponse)
def get_concept(
    concept_id: str,
    namespace: str = Query(default="default"),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    ensure_namespace_access(auth, namespace, {ApiRole.reader.value, ApiRole.writer.value, ApiRole.admin.value})
    result = crud.get_concept_with_history(session, namespace, concept_id)
    if not result:
        raise HTTPException(status_code=404, detail="Concept not found")
    return schemas.ConceptDetailResponse(**result)


@app.post("/admin/synthesize", response_model=schemas.SynthesisRunResponse)
def admin_trigger_synthesis(
    namespace: str = Query(default="default"),
    full: bool = Query(default=False, description="Re-synthesize ALL knowledge, not just recent."),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Manually trigger a synthesis run. Set full=true after model upgrades."""
    require_admin(auth)
    ensure_namespace_access(auth, namespace, {ApiRole.admin.value})

    from .synthesis.loop import _run_synthesis_for_namespace

    stats = _run_synthesis_for_namespace(
        namespace=namespace,
        settings=app.state.settings,
        llm_provider=app.state.llm_provider,
        embedding_provider=app.state.embedding_provider,
        full_resynthesis=full,
        skip_decay=True,
    )
    return schemas.SynthesisRunResponse(
        namespace=namespace,
        new_concepts=stats["new"],
        updated_concepts=stats["updated"],
        decayed_concepts=stats["decayed"],
        feedback_adjustments=stats.get("feedback_adjustments", 0),
    )


# ── Enrollment ─────────────────────────────────────────────────────


@app.post("/admin/enrollment-tokens", response_model=schemas.EnrollmentTokenCreateResponse)
def admin_create_enrollment_token(
    payload: schemas.EnrollmentTokenCreateRequest,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    require_admin(auth)

    from datetime import UTC, datetime, timedelta

    expires_at = None
    if payload.expires_in_hours is not None:
        expires_at = datetime.now(UTC) + timedelta(hours=payload.expires_in_hours)

    token_record, plaintext = crud.create_enrollment_token(
        session=session,
        role=payload.role,
        namespaces=payload.namespaces,
        name_hint=payload.name_hint,
        max_uses=payload.max_uses,
        expires_at=expires_at,
    )

    return schemas.EnrollmentTokenCreateResponse(
        id=str(token_record.id),
        token=plaintext,
        role=payload.role,
        namespaces=list(token_record.namespaces or []),
        name_hint=token_record.name_hint,
        max_uses=token_record.max_uses,
        expires_at=token_record.expires_at,
    )


@app.get("/admin/enrollment-tokens", response_model=list[schemas.EnrollmentTokenListItem])
def admin_list_enrollment_tokens(
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    require_admin(auth)
    rows = crud.list_enrollment_tokens(session)
    return [schemas.EnrollmentTokenListItem(**row) for row in rows]


@app.post("/enroll", response_model=schemas.EnrollResponse)
def enroll(
    payload: schemas.EnrollRequest,
    session: Session = Depends(get_session),
):
    """Redeem an enrollment token to receive an API key. No auth required — the token IS the auth."""
    record, api_key_plaintext = crud.redeem_enrollment_token(
        session=session,
        plaintext_token=payload.token,
        key_name=payload.name,
    )

    return schemas.EnrollResponse(
        api_key=api_key_plaintext,
        name=record.name,
        role=record.role.value if hasattr(record.role, "value") else str(record.role),
        namespaces=list(record.namespaces or []),
    )



# ── Open Enrollment (LAN trust) ───────────────────────────────────


@app.post("/enroll/auto", response_model=schemas.AutoEnrollResponse)
def enroll_auto(
    payload: schemas.AutoEnrollRequest,
    request: Request,
    session: Session = Depends(get_session),
):
    """Self-service enrollment. No auth required — LAN trust + optional passphrase."""
    record, plaintext, agent_config = crud.auto_enroll(
        session=session,
        name=payload.name,
        passphrase=payload.passphrase,
        settings=app.state.settings,
    )

    return schemas.AutoEnrollResponse(
        api_key=plaintext,
        name=record.name,
        role=record.role.value if hasattr(record.role, "value") else str(record.role),
        namespaces=list(record.namespaces or []),
        agent_config=schemas.AgentConfig(**agent_config),
    )


@app.get("/enroll/portal", include_in_schema=False)
def enroll_portal():
    """Serve the enrollment portal HTML page."""
    portal_file = UI_DIR / "enroll.html"
    if not portal_file.exists():
        raise HTTPException(status_code=404, detail="Enrollment portal not found")
    return FileResponse(portal_file)


@app.get("/enroll/bootstrap.sh", response_class=PlainTextResponse, include_in_schema=False)
def enroll_bootstrap_script(request: Request):
    """Serve a curl-pipeable bootstrap script for CLI enrollment."""
    server_host = request.headers.get("host", "localhost:8088")
    base_url = f"http://{server_host}"

    return f'''#!/usr/bin/env bash
set -euo pipefail

# AgentSSOT Bootstrap Enrollment Script
# Usage: curl -s {base_url}/enroll/bootstrap.sh | bash -s -- "my-device-name"
#   or:  curl -s {base_url}/enroll/bootstrap.sh | bash -s -- "my-device-name" --passphrase "secret"

DEVICE_NAME=""
PASSPHRASE=""
FORCE=false
BASE_URL="{base_url}"
CLAUDE_DIR="$HOME/.claude"
AGENT_JSON="$CLAUDE_DIR/agentssot/local/agent.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --passphrase) PASSPHRASE="$2"; shift 2 ;;
        --force) FORCE=true; shift ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) DEVICE_NAME="$1"; shift ;;
    esac
done

if [[ -z "$DEVICE_NAME" ]]; then
    echo "Usage: curl -s $BASE_URL/enroll/bootstrap.sh | bash -s -- \"device-name\" [--passphrase \"secret\"] [--force]"
    exit 1
fi

echo "==> Enrolling device: $DEVICE_NAME"

if [[ -f "$AGENT_JSON" && "$FORCE" != "true" ]]; then
    echo "WARNING: $AGENT_JSON already exists."
    echo "Use --force to overwrite, or remove it manually."
    exit 1
fi

if [[ ! -d "$CLAUDE_DIR" ]]; then
    echo "==> ~/.claude directory not found. Creating it..."
    mkdir -p "$CLAUDE_DIR"
fi

mkdir -p "$CLAUDE_DIR/agentssot/local"

echo "==> Calling enrollment API..."
RESPONSE=$(curl -sf -X POST "$BASE_URL/enroll/auto" \
    -H "Content-Type: application/json" \
    -d "{{\\"name\\":\\"$DEVICE_NAME\\",\\"passphrase\\":\\"$PASSPHRASE\\"}}")

if [[ $? -ne 0 || -z "$RESPONSE" ]]; then
    echo "ERROR: Enrollment failed. Is $BASE_URL reachable?"
    exit 1
fi

echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
config = data['agent_config']
print(json.dumps(config, indent=2))
" > "$AGENT_JSON"

echo "==> Wrote $AGENT_JSON"

echo "==> Verifying enrollment..."
API_KEY=$(python3 -c "import sys,json; print(json.load(sys.stdin)['api_key'])" <<< "$RESPONSE")
HEALTH=$(curl -sf -H "X-API-Key: $API_KEY" "$BASE_URL/health" 2>/dev/null || echo "FAIL")

if echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])" 2>/dev/null | grep -q "ok"; then
    echo "==> Enrollment successful! Device '$DEVICE_NAME' is now connected to AgentSSOT."
    echo ""
    echo "    API Key:    $API_KEY"
    echo "    Config:     $AGENT_JSON"
    echo "    Namespaces: claude-shared, device-$DEVICE_NAME-private"
else
    echo "WARNING: Enrollment completed but health check failed. Config was still written."
    echo "    Check connectivity to $BASE_URL"
fi

# --- MCP Plugin Installation ---
echo ""
echo "==> Installing hari-hive MCP plugin..."

PLUGIN_DIR="$CLAUDE_DIR/plugins/hari-hive"

# Check if uv is available (required for MCP server)
if ! command -v uv &>/dev/null; then
    echo "WARNING: uv not found. MCP plugin requires uv (https://docs.astral.sh/uv/)."
    echo "    Install uv first, then re-run with --force to install plugin."
    echo "    Enrollment succeeded but MCP tools will not be available."
    exit 0
fi

# Download plugin bundle from API
BUNDLE=$(curl -sf "$BASE_URL/enroll/plugin-bundle" 2>/dev/null)
if [[ $? -ne 0 || -z "$BUNDLE" ]]; then
    echo "WARNING: Could not download plugin bundle from $BASE_URL/enroll/plugin-bundle"
    echo "    Enrollment succeeded but MCP plugin was not installed."
    exit 0
fi

# Create plugin directory structure
mkdir -p "$PLUGIN_DIR/hooks" "$PLUGIN_DIR/skills/hive"

# Write each plugin file from the bundle
echo "$BUNDLE" | python3 -c "
import sys, json, os
bundle = json.load(sys.stdin)
plugin_dir = os.environ.get('PLUGIN_DIR', os.path.expanduser('~/.claude/plugins/hari-hive'))
for rel_path, content in bundle.items():
    full_path = os.path.join(plugin_dir, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w') as f:
        f.write(content)
    print(f'    Wrote {{rel_path}}')
"

echo "==> Plugin installed at $PLUGIN_DIR"

# Enable plugin in settings.json if not already
SETTINGS_FILE="$CLAUDE_DIR/settings.json"
if [[ -f "$SETTINGS_FILE" ]]; then
    python3 -c "
import json, sys

settings_file = sys.argv[1]
with open(settings_file) as f:
    settings = json.load(f)

plugins = settings.setdefault('enabledPlugins', {{}})
if 'hari-hive' not in plugins:
    plugins['hari-hive'] = True
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)
        f.write('\\n')
    print('==> Enabled hari-hive plugin in settings.json')
else:
    print('==> hari-hive plugin already enabled in settings.json')
" "$SETTINGS_FILE"
else
    echo "    NOTE: No settings.json found. Plugin installed but not auto-enabled."
    echo "    Enable manually or start Claude Code to auto-detect it."
fi

# Remove old hive hooks from settings.json if present
if [[ -f "$SETTINGS_FILE" ]]; then
    python3 -c "
import json, sys

settings_file = sys.argv[1]
with open(settings_file) as f:
    settings = json.load(f)

hooks = settings.get('hooks', {{}})
changed = False

# Remove old SessionStart hive hook
start_hooks = hooks.get('SessionStart', [])
new_start = [h for h in start_hooks if 'session-recall' not in json.dumps(h) and 'hive-session-start' not in json.dumps(h)]
if len(new_start) != len(start_hooks):
    if new_start:
        hooks['SessionStart'] = new_start
    else:
        hooks.pop('SessionStart', None)
    changed = True

# Remove old SessionEnd extract_and_ingest hook
end_hooks = hooks.get('SessionEnd', [])
new_end = [h for h in end_hooks if 'extract_and_ingest' not in json.dumps(h)]
if len(new_end) != len(end_hooks):
    hooks['SessionEnd'] = new_end
    changed = True

if changed:
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)
        f.write('\\n')
    print('==> Removed old hive shell hooks from settings.json (replaced by plugin)')
" "$SETTINGS_FILE" 2>/dev/null
fi

echo ""
echo "==> Setup complete! Start a new Claude Code session to activate."
echo "    MCP tools available: hive_recall, hive_query, hive_ingest, hive_stats, + 6 more"
echo "    Slash command: /hive [query]"
'''


@app.get("/enroll/install-plugin.sh", response_class=PlainTextResponse, include_in_schema=False)
def enroll_install_plugin_script(request: Request):
    """Plugin-only install script for already-enrolled agents."""
    server_host = request.headers.get("host", "localhost:8088")
    base_url = f"http://{server_host}"

    return f'''#!/usr/bin/env bash
set -euo pipefail

# hari-hive MCP Plugin Installer (plugin-only, no re-enrollment)
# For agents that already have ~/.claude/agentssot/local/agent.json
# Usage: curl -s {base_url}/enroll/install-plugin.sh | bash

CLAUDE_DIR="$HOME/.claude"
AGENT_JSON="$CLAUDE_DIR/agentssot/local/agent.json"
PLUGIN_DIR="$CLAUDE_DIR/plugins/hari-hive"
BASE_URL="{base_url}"

# Verify agent.json exists (must be enrolled first)
if [[ ! -f "$AGENT_JSON" ]]; then
    echo "ERROR: $AGENT_JSON not found."
    echo "    You must enroll first:"
    echo "    curl -s $BASE_URL/enroll/bootstrap.sh | bash -s -- \\"my-device-name\\""
    exit 1
fi

echo "==> Found agent config at $AGENT_JSON"

# Check uv
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it first: https://docs.astral.sh/uv/"
    exit 1
fi

# Download plugin bundle
echo "==> Downloading plugin bundle..."
BUNDLE=$(curl -sf "$BASE_URL/enroll/plugin-bundle" 2>/dev/null)
if [[ $? -ne 0 || -z "$BUNDLE" ]]; then
    echo "ERROR: Could not download plugin bundle from $BASE_URL/enroll/plugin-bundle"
    exit 1
fi

# Create plugin directory structure
mkdir -p "$PLUGIN_DIR/hooks" "$PLUGIN_DIR/skills/hive"

# Write plugin files
echo "$BUNDLE" | python3 -c "
import sys, json, os
bundle = json.load(sys.stdin)
plugin_dir = os.environ.get('PLUGIN_DIR', os.path.expanduser('~/.claude/plugins/hari-hive'))
for rel_path, content in bundle.items():
    full_path = os.path.join(plugin_dir, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w') as f:
        f.write(content)
    print(f'    Wrote {{rel_path}}')
"

echo "==> Plugin installed at $PLUGIN_DIR"

# Enable plugin in settings.json
SETTINGS_FILE="$CLAUDE_DIR/settings.json"
if [[ -f "$SETTINGS_FILE" ]]; then
    python3 -c "
import json, sys
settings_file = sys.argv[1]
with open(settings_file) as f:
    settings = json.load(f)

changed = False
plugins = settings.setdefault('enabledPlugins', {{}})
if 'hari-hive' not in plugins:
    plugins['hari-hive'] = True
    changed = True

# Remove old hive hooks
hooks = settings.get('hooks', {{}})
start_hooks = hooks.get('SessionStart', [])
new_start = [h for h in start_hooks if 'session-recall' not in json.dumps(h) and 'hive-session-start' not in json.dumps(h)]
if len(new_start) != len(start_hooks):
    if new_start:
        hooks['SessionStart'] = new_start
    else:
        hooks.pop('SessionStart', None)
    changed = True

end_hooks = hooks.get('SessionEnd', [])
new_end = [h for h in end_hooks if 'extract_and_ingest' not in json.dumps(h)]
if len(new_end) != len(end_hooks):
    hooks['SessionEnd'] = new_end
    changed = True

if changed:
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)
        f.write('\\n')
    print('==> Updated settings.json (enabled plugin, removed old hooks)')
else:
    print('==> settings.json already up to date')
" "$SETTINGS_FILE"
else
    echo "    NOTE: No settings.json found. Plugin installed but not auto-enabled."
fi

echo ""
echo "==> Done! Restart Claude Code to activate MCP tools."
echo "    Tools: hive_recall, hive_query, hive_ingest, hive_stats, + 6 more"
echo "    Slash command: /hive [query]"
'''


@app.get("/enroll/plugin-bundle", include_in_schema=False)
def enroll_plugin_bundle():
    """Serve MCP plugin files as a JSON bundle for bootstrap installation."""
    if not PLUGIN_DIR.exists():
        raise HTTPException(status_code=404, detail="Plugin bundle not found")

    bundle: dict[str, str] = {}
    for rel_path in [
        "mcp_server.py",
        "plugin.json",
        ".mcp.json",
        "hooks/SessionStart.md",
        "hooks/SessionEnd.md",
        "skills/hive/SKILL.md",
    ]:
        fp = PLUGIN_DIR / rel_path
        if fp.exists():
            bundle[rel_path] = fp.read_text()

    return bundle
