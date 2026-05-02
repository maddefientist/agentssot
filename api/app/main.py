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

from . import crud, schemas, wal
from .background import compaction_loop, lifecycle_sweep_loop
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
from .sync import router as sync_router
from .routers import knowledge_router
from .routers.agent_guide import router as agent_guide_router

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

    # Lifecycle sweep loop — runs at 03:00 UTC daily
    lifecycle_task = asyncio.create_task(lifecycle_sweep_loop(app), name="lifecycle-sweep-loop")
    logger.info("background lifecycle sweep loop started (03:00 UTC)")
    app.state.lifecycle_task = lifecycle_task

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

    if lifecycle_task:
        lifecycle_task.cancel()
        try:
            await lifecycle_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="AgentSSOT API", version="1.0.0", lifespan=lifespan)
if UI_DIR.exists():
    app.mount("/ui/assets", StaticFiles(directory=UI_DIR), name="ui-assets")


app.include_router(cortex_router)
app.include_router(sync_router)
app.include_router(knowledge_router, prefix="/api/v1")
app.include_router(agent_guide_router)


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
    cortex_file = UI_DIR / "cortex-v2.html"
    if cortex_file.exists():
        return FileResponse(cortex_file)
    # fallback to original
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
    min_weight: float = Query(default=0.3, ge=0.0, le=1.0),
    session: Session = Depends(get_session),
):
    """Public read-only endpoint for cortex edges. Returns concept_links."""
    return {"links": crud.list_concept_links(session, namespace, limit, min_weight)}


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
    """Enhanced stats endpoint for dashboard. No auth required.

    Includes: basic counts, memory type distribution (M3), staleness distribution (M3),
    secret scanning stats (M8), sync checkpoint status (M10).
    """
    from sqlalchemy import func, text as sa_text, case, literal_column
    from .models import Concept, KnowledgeItem, RecallEvent, ConceptFeedback

    # ── Basic counts (existing) ────────────────────────────────────
    concepts_total = session.scalar(select(func.count()).select_from(Concept).where(Concept.namespace == namespace)) or 0
    skills = session.scalar(select(func.count()).select_from(Concept).where(Concept.namespace == namespace, Concept.type == "skill")) or 0
    knowledge = session.scalar(select(func.count()).select_from(KnowledgeItem).where(KnowledgeItem.namespace == namespace)) or 0
    recalls_24h = session.scalar(
        select(func.count()).select_from(RecallEvent)
        .where(RecallEvent.namespace == namespace, RecallEvent.created_at > sa_text("now() - interval '24 hours'"))
    ) or 0
    ingested_24h = session.scalar(
        select(func.count()).select_from(KnowledgeItem)
        .where(KnowledgeItem.namespace == namespace, KnowledgeItem.created_at > sa_text("now() - interval '24 hours'"))
    ) or 0
    avg_conf = session.scalar(
        select(func.avg(Concept.confidence)).where(Concept.namespace == namespace, Concept.confidence.isnot(None))
    ) or 0

    # ── Memory type distribution (M3 data) ─────────────────────────
    memory_type_rows = session.execute(
        select(
            func.coalesce(KnowledgeItem.memory_type, "untyped").label("mtype"),
            func.count().label("cnt"),
        )
        .where(KnowledgeItem.namespace == namespace)
        .group_by("mtype")
    ).all()
    memory_type_distribution = {row.mtype: row.cnt for row in memory_type_rows}

    # ── Staleness distribution (M3 data) ───────────────────────────
    # Buckets: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0], null
    staleness_buckets = {"fresh": 0, "aging": 0, "stale": 0, "critical": 0, "unscored": 0}
    staleness_rows = session.execute(
        select(
            case(
                (KnowledgeItem.staleness_score.is_(None), "unscored"),
                (KnowledgeItem.staleness_score < 0.25, "fresh"),
                (KnowledgeItem.staleness_score < 0.5, "aging"),
                (KnowledgeItem.staleness_score < 0.75, "stale"),
                else_="critical",
            ).label("bucket"),
            func.count().label("cnt"),
        )
        .where(KnowledgeItem.namespace == namespace)
        .group_by("bucket")
    ).all()
    for row in staleness_rows:
        staleness_buckets[row.bucket] = row.cnt

    # ── Secret scanning stats (M8 data) ────────────────────────────
    # We track rejected items via the access log, but we can report config status
    # and count items with the 'secret-rejected' tag if any got through historically
    secret_scanning_enabled = settings.ingest_secret_scanning

    # ── Sync checkpoint status (M10 data) ──────────────────────────
    sync_status: list[dict] = []
    if settings.sync_tracking_enabled:
        try:
            sync_rows = session.execute(sa_text("""
                SELECT device_id, namespace, last_synced_at
                FROM sync_checkpoints
                WHERE namespace = :namespace
                ORDER BY last_synced_at DESC
            """), {"namespace": namespace}).all()
            sync_status = [
                {
                    "device_id": row.device_id,
                    "last_synced_at": row.last_synced_at.isoformat() if row.last_synced_at else None,
                }
                for row in sync_rows
            ]
        except Exception:
            # Table might not exist yet
            pass

    return {
        # Original fields (backward compatible)
        "concepts": concepts_total,
        "skills": skills,
        "knowledge": knowledge,
        "recalls_24h": recalls_24h,
        "ingested_24h": ingested_24h,
        "avg_confidence": float(avg_conf),
        # M3: Memory type distribution
        "memory_type_distribution": memory_type_distribution,
        # M3: Staleness distribution
        "staleness_distribution": staleness_buckets,
        # M8: Secret scanning
        "secret_scanning": {
            "enabled": secret_scanning_enabled,
        },
        # M10: Sync checkpoints
        "sync": {
            "enabled": settings.sync_tracking_enabled,
            "devices": sync_status,
        },
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
    wal.log_event(
        "ingest.batch",
        namespace=namespace,
        actor_key_id=auth.key_id,
        payload=payload.model_dump(),
        result={"counts": counts},
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


@app.get("/admin/feedback")
def list_feedback(
    namespace: str = Query(default="claude-shared"),
    concept_id: str | None = Query(default=None),
    signal: str | None = Query(default=None),  # useful/noted/wrong
    limit: int = Query(default=50),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """List recent feedback with optional filters. Returns feedback records with concept titles."""
    from .models import ConceptFeedback, Concept
    from uuid import UUID

    q = (
        select(
            ConceptFeedback.id,
            ConceptFeedback.concept_id,
            ConceptFeedback.signal,
            ConceptFeedback.agent_key,
            ConceptFeedback.session_id,
            ConceptFeedback.note,
            ConceptFeedback.created_at,
            Concept.title.label("concept_title"),
        )
        .join(Concept, ConceptFeedback.concept_id == Concept.id)
        .where(ConceptFeedback.namespace == namespace)
    )

    if concept_id:
        try:
            q = q.where(ConceptFeedback.concept_id == UUID(concept_id))
        except ValueError:
            pass

    if signal:
        q = q.where(ConceptFeedback.signal == signal)

    q = q.order_by(ConceptFeedback.created_at.desc()).limit(min(limit, 200))

    rows = session.execute(q).all()
    return [
        {
            "id": str(r.id),
            "concept_id": str(r.concept_id),
            "concept_title": r.concept_title,
            "signal": r.signal.value if hasattr(r.signal, "value") else str(r.signal),
            "agent_key": r.agent_key,
            "session_id": r.session_id,
            "note": r.note,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


@app.get("/admin/feedback/contested")
def list_contested_concepts(
    namespace: str = Query(default="claude-shared"),
    limit: int = Query(default=30),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Return concepts that have received both useful AND wrong signals (contested)."""
    from .models import ConceptFeedback, Concept
    from sqlalchemy import func, and_

    subq = (
        select(
            ConceptFeedback.concept_id,
            func.count().filter(ConceptFeedback.signal == "useful").label("useful_count"),
            func.count().filter(ConceptFeedback.signal == "wrong").label("wrong_count"),
            func.count().filter(ConceptFeedback.signal == "noted").label("noted_count"),
            func.count().label("total_count"),
        )
        .where(ConceptFeedback.namespace == namespace)
        .group_by(ConceptFeedback.concept_id)
        .subquery()
    )

    rows = session.execute(
        select(
            subq.c.concept_id,
            subq.c.useful_count,
            subq.c.wrong_count,
            subq.c.noted_count,
            subq.c.total_count,
            Concept.title,
            Concept.type,
            Concept.confidence,
        )
        .join(Concept, subq.c.concept_id == Concept.id)
        .where(
            and_(subq.c.useful_count > 0, subq.c.wrong_count > 0)
        )
        .order_by(subq.c.total_count.desc())
        .limit(min(limit, 100))
    ).all()

    return [
        {
            "concept_id": str(r.concept_id),
            "concept_title": r.title,
            "concept_type": r.type.value if hasattr(r.type, "value") else str(r.type),
            "confidence": r.confidence,
            "useful_count": r.useful_count,
            "wrong_count": r.wrong_count,
            "noted_count": r.noted_count,
            "total_count": r.total_count,
        }
        for r in rows
    ]


@app.post("/admin/delete-items", response_model=schemas.DeleteItemsResponse)
def admin_delete_items(
    payload: schemas.DeleteItemsRequest,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    require_admin(auth)
    ensure_namespace_access(auth, payload.namespace, {ApiRole.admin.value})

    deleted = crud.delete_items(session, payload.namespace, payload.ids)
    wal.log_event(
        "knowledge.delete",
        namespace=payload.namespace,
        actor_key_id=auth.key_id,
        payload={"ids": payload.ids},
        result={"deleted": deleted},
    )
    return schemas.DeleteItemsResponse(namespace=payload.namespace, deleted=deleted)



@app.post("/admin/delete-concepts", response_model=schemas.DeleteConceptsResponse)
def admin_delete_concepts(
    payload: schemas.DeleteConceptsRequest,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Delete concepts by ID. Also removes associated links, recall events, and feedback via CASCADE."""
    require_admin(auth)
    ensure_namespace_access(auth, payload.namespace, {ApiRole.admin.value})

    result = crud.delete_concepts(session, payload.namespace, payload.ids)
    wal.log_event(
        "concept.delete",
        namespace=payload.namespace,
        actor_key_id=auth.key_id,
        payload={"ids": payload.ids},
        result=result,
    )
    return schemas.DeleteConceptsResponse(namespace=payload.namespace, **result)

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


# ── Settings API ──────────────────────────────────────────────────

# Fields treated as sensitive — values are masked in GET responses.
_SENSITIVE_FIELDS = frozenset({
    "openai_api_key",
    "enrollment_passphrase",
    "database_url",
})

# Settings that can be updated at runtime without a restart.
_RUNTIME_CONFIGURABLE = frozenset({
    "synthesis_enabled",
    "synthesis_similarity_threshold",
    "synthesis_min_cluster_size",
    "synthesis_confidence_decay",
    "synthesis_decay_floor",
    "synthesis_decay_grace_days",
    "compaction_enabled",
    "compaction_event_threshold",
    "compaction_char_threshold",
    "typed_memory_enabled",
    "ingest_secret_scanning",
    "semantic_dedup_threshold",
    "wal_enabled",
    "wal_retention_days",
    "sync_tracking_enabled",
    "sync_conflict_window_hours",
})

# Human-readable descriptions for each setting.
_SETTING_DESCRIPTIONS: dict[str, str] = {
    "database_url": "PostgreSQL connection string (restart required)",
    "api_port": "HTTP port the API listens on (restart required)",
    "log_level": "Log verbosity: debug/info/warning/error (restart required)",
    "default_top_k": "Default number of recall results returned",
    "max_snippet_chars": "Maximum characters per knowledge snippet",
    "embedding_provider": "Embedding backend: none/openai/ollama (restart required)",
    "embedding_dim": "Embedding vector dimension (restart required)",
    "openai_api_key": "OpenAI API key — masked (restart required)",
    "openai_embed_model": "OpenAI embedding model name (restart required)",
    "ollama_base_url": "Ollama service base URL (restart required)",
    "ollama_embed_model": "Ollama embedding model name (restart required)",
    "llm_provider": "LLM backend: none/openai/ollama (restart required)",
    "openai_chat_model": "OpenAI chat model name (restart required)",
    "ollama_chat_model": "Ollama chat model name (restart required)",
    "reranker_provider": "Reranker backend: none/ollama (restart required)",
    "ollama_reranker_model": "Ollama reranker model name (restart required)",
    "reranker_candidate_multiplier": "Candidate pool multiplier for reranking",
    "compaction_enabled": "Enable/disable background event compaction",
    "compaction_event_threshold": "Number of events that triggers compaction",
    "compaction_char_threshold": "Character count that triggers compaction",
    "compaction_interval_seconds": "How often the compaction loop checks (restart required)",
    "synthesis_enabled": "Enable/disable background concept synthesis",
    "synthesis_model": "LLM model used for synthesis (restart required)",
    "synthesis_fallback_model": "Fallback LLM for synthesis (restart required)",
    "synthesis_schedule_hour": "UTC hour when synthesis runs (restart required)",
    "synthesis_similarity_threshold": "Cosine similarity threshold for clustering items into concepts",
    "synthesis_min_cluster_size": "Minimum items per cluster for synthesis",
    "synthesis_confidence_decay": "Per-day confidence decay applied to concepts",
    "synthesis_decay_floor": "Minimum confidence floor (concepts never decay below this)",
    "synthesis_decay_grace_days": "Days before decay begins on a concept",
    "synthesis_feedback_protection_days": "Days after last feedback before decay resumes (restart required)",
    "enable_hnsw_index": "Use HNSW index for vector recall (restart required)",
    "typed_memory_enabled": "Enable memory_type and staleness filters on recall",
    "ingest_secret_scanning": "Reject knowledge items containing likely secrets on ingest",
    "sync_tracking_enabled": "Enable per-device sync checkpoints and conflict detection",
    "sync_conflict_window_hours": "Hours window used for sync conflict detection",
    "wal_enabled": "Enable write-ahead log for ingest/delete audit",
    "wal_dir": "Directory for WAL files (restart required)",
    "wal_retention_days": "Days before WAL files are pruned",
    "semantic_dedup_threshold": "Cosine similarity threshold for ingest dedup (0.0 = disabled)",
    "enrollment_passphrase": "Required passphrase for /enroll/auto — masked (restart required)",
    "expose_db_port": "Expose Postgres port externally (restart required)",
    "bootstrap_admin_namespaces": "Comma-separated namespaces given to the bootstrap admin key (restart required)",
}

# Expected Python types for each configurable field (used for coercion and validation).
_SETTING_TYPES: dict[str, type] = {
    "synthesis_enabled": bool,
    "synthesis_similarity_threshold": float,
    "synthesis_min_cluster_size": int,
    "synthesis_confidence_decay": float,
    "synthesis_decay_floor": float,
    "synthesis_decay_grace_days": int,
    "compaction_enabled": bool,
    "compaction_event_threshold": int,
    "compaction_char_threshold": int,
    "typed_memory_enabled": bool,
    "ingest_secret_scanning": bool,
    "semantic_dedup_threshold": float,
    "wal_enabled": bool,
    "wal_retention_days": int,
    "sync_tracking_enabled": bool,
    "sync_conflict_window_hours": int,
}

# Acceptable value ranges for numeric settings (inclusive).
_SETTING_RANGES: dict[str, tuple[float, float]] = {
    "synthesis_similarity_threshold": (0.0, 1.0),
    "synthesis_confidence_decay": (0.0, 1.0),
    "synthesis_decay_floor": (0.0, 1.0),
    "semantic_dedup_threshold": (0.0, 1.0),
    "synthesis_min_cluster_size": (1, 1000),
    "synthesis_decay_grace_days": (0, 3650),
    "compaction_event_threshold": (1, 100_000),
    "compaction_char_threshold": (1, 10_000_000),
    "wal_retention_days": (1, 3650),
    "sync_conflict_window_hours": (1, 8760),
}


def _mask_value(key: str, value: object) -> object:
    if key in _SENSITIVE_FIELDS:
        raw = str(value)
        return "****" if not raw else f"****({len(raw)} chars)"
    return value


def _type_name(val: object) -> str:
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    return "str"


def _coerce_setting(key: str, raw: object) -> object:
    """Coerce a raw JSON value to the expected Python type. Raises ValueError on mismatch."""
    expected = _SETTING_TYPES.get(key)
    if expected is None:
        raise ValueError(f"Unknown configurable setting: {key!r}")

    if expected is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            if raw.lower() in ("true", "1", "yes"):
                return True
            if raw.lower() in ("false", "0", "no"):
                return False
        raise ValueError(f"{key}: expected bool, got {type(raw).__name__} {raw!r}")

    if expected is int:
        if isinstance(raw, bool):
            raise ValueError(f"{key}: expected int, not bool")
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float) and raw == int(raw):
            return int(raw)
        if isinstance(raw, str):
            try:
                return int(raw)
            except ValueError:
                pass
        raise ValueError(f"{key}: expected int, got {type(raw).__name__} {raw!r}")

    if expected is float:
        if isinstance(raw, bool):
            raise ValueError(f"{key}: expected float, not bool")
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw)
            except ValueError:
                pass
        raise ValueError(f"{key}: expected float, got {type(raw).__name__} {raw!r}")

    return str(raw)


@app.get("/admin/settings", response_model=schemas.SettingsGetResponse)
def get_settings_endpoint(
    auth: AuthContext = Depends(require_api_key),
):
    """Return all current settings with metadata (admin only). Sensitive values are masked."""
    require_admin(auth)

    s = app.state.settings
    items = []
    for field_name in s.model_fields:
        value = getattr(s, field_name, None)
        items.append(
            schemas.SettingMeta(
                key=field_name,
                value=_mask_value(field_name, value),
                type=_type_name(value),
                runtime_configurable=field_name in _RUNTIME_CONFIGURABLE,
                description=_SETTING_DESCRIPTIONS.get(field_name, ""),
            )
        )

    return schemas.SettingsGetResponse(settings=items)


@app.post("/admin/settings", response_model=schemas.SettingsUpdateResponse)
def update_settings_endpoint(
    payload: schemas.SettingsUpdateRequest,
    auth: AuthContext = Depends(require_api_key),
):
    """Update runtime-configurable settings (admin only). Changes are session-only — not persisted to .env."""
    require_admin(auth)

    s = app.state.settings
    applied: dict[str, object] = {}
    skipped: dict[str, str] = {}
    errors: list[str] = []

    for key, raw_value in payload.updates.items():
        # Reject unknown fields
        if not hasattr(s, key):
            skipped[key] = "unknown setting"
            continue

        # Reject non-runtime-configurable settings
        if key not in _RUNTIME_CONFIGURABLE:
            skipped[key] = "requires restart — update .env and redeploy to change this setting"
            continue

        # Coerce and validate type
        try:
            value = _coerce_setting(key, raw_value)
        except ValueError as exc:
            errors.append(str(exc))
            skipped[key] = f"type error: {exc}"
            continue

        # Range check for numerics
        if key in _SETTING_RANGES:
            lo, hi = _SETTING_RANGES[key]
            if not (lo <= value <= hi):  # type: ignore[operator]
                msg = f"{key}: value {value} out of range [{lo}, {hi}]"
                errors.append(msg)
                skipped[key] = f"range error: {msg}"
                continue

        # Apply to live settings object
        object.__setattr__(s, key, value)
        applied[key] = _mask_value(key, value)
        logger.info("settings.update key=%s value=%r actor=%s", key, value, auth.key_name)

    if errors and not applied:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="; ".join(errors),
        )

    # Log to WAL (best-effort)
    wal.log_event(
        "settings.update",
        namespace=None,
        actor_key_id=auth.key_id,
        payload={"updates": {k: _mask_value(k, v) for k, v in payload.updates.items()}},
        result={"applied": applied, "skipped": skipped},
    )

    return schemas.SettingsUpdateResponse(applied=applied, skipped=skipped)


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
