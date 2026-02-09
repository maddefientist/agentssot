import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from . import crud, schemas
from .background import compaction_loop
from .db import get_session
from .embeddings import build_embedding_provider
from .llm import LLMProviderError, build_llm_provider
from .logging_config import configure_logging
from .models import ApiRole
from .security import AuthContext, ensure_namespace_access, require_admin, require_api_key
from .settings import get_settings
from .startup import initialize_system

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("agentssot.api")
BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.embedding_provider = build_embedding_provider(settings)
    app.state.llm_provider = build_llm_provider(settings)

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

    yield

    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="AgentSSOT (hari-hive) API", version="1.0.0", lifespan=lifespan)
if UI_DIR.exists():
    app.mount("/ui/assets", StaticFiles(directory=UI_DIR), name="ui-assets")


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
    }


@app.get("/", include_in_schema=False)
def ui_home():
    index_file = UI_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return RedirectResponse(url="/docs")


@app.get("/onboarding", response_class=PlainTextResponse, include_in_schema=False)
def onboarding(auth: AuthContext = Depends(require_api_key)) -> str:
    s = app.state.settings
    base_url_hint = "http://<agentssot-host>:8088"

    namespaces = ", ".join(auth.namespaces) if auth.namespaces else "(none)"
    role = auth.role

    # Intentionally plaintext and LLM-friendly.
    return "\n".join(
        [
            "hari-hive Onboarding (LLM-Friendly)",
            "",
            "Purpose:",
            "- hari-hive is our cross-LLM shared memory service, backed by AgentSSOT.",
            "- Use it to reduce prompt bloat by retrieving only the top relevant facts/events, then writing back durable memory.",
            "",
            "Getting an API key:",
            "- This page requires an API key; it cannot be used to obtain one.",
            "- Ask an operator (admin) to issue you a key via the dashboard (/ -> Admin: API Keys) or POST /admin/api-keys.",
            "- Keys should be distributed out-of-band (secret manager / config sync secrets), never committed to git.",
            "",
            "Auth Context (your key):",
            f"- role: {role}",
            f"- namespaces: {namespaces}",
            "",
            "Connection:",
            f"- BASE_URL: {base_url_hint}",
            "- Header: X-API-Key: <your-key>",
            "",
            "Core Rules:",
            "- Always set namespace explicitly on every request (default is 'default' only if omitted).",
            "- Never read/write namespaces your key does not allow.",
            "- Prefer private namespaces by default; use shared namespaces only for intentional sharing.",
            "- Scoped additional silos can be created on demand (new namespaces), but require an admin to provision access.",
            "- Keep memory atomic and tagged. Do not dump full transcripts unless explicitly needed.",
            "",
            "Recommended Loop (every task):",
            "1) Start: GET /query and POST /recall in the relevant namespace(s).",
            "2) During: POST /ingest events for decisions/directives/results; knowledge_items for durable facts.",
            "3) End: POST /summarize_clear for the session_id (if enabled) to store a summary knowledge_item and archive events.",
            "",
            "Endpoints:",
            "- GET  /health (no auth)",
            "- GET  /query (auth)",
            "- POST /recall (auth)",
            "- POST /ingest (writer/admin)",
            "- POST /summarize_clear (writer/admin)",
            "- POST /admin/namespaces (admin)",
            "- POST /admin/api-keys (admin)",
            "- GET  /admin/api-keys (admin)",
            "",
            "Recall Notes:",
            f"- EMBEDDING_PROVIDER={s.embedding_provider}. If 'none', clients must provide query_embedding.",
            f"- DEFAULT_TOP_K={s.default_top_k}. Keep Top-K small to stay token-efficient.",
            "",
            "Ingest Notes:",
            "- Knowledge items are auto-chunked server-side to max ~800 chars per row.",
            "- If you provide an embedding in the request, the server will not recompute it.",
            "",
            "Migration Guidance (high level):",
            "- Migrate durable facts first (knowledge_items).",
            "- Migrate project context files next (PROJECT.md/CONTEXT.md/etc) as knowledge_items tagged 'project-context'.",
            "- Migrate sessions by storing decisions/results as events and letting compaction create summaries.",
            "",
            "Web UI:",
            "- Dashboard: GET /",
            "- Swagger docs: GET /docs",
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
        settings=app.state.settings,
    )

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
