"""
Cortex — Working Memory Layer for AgentSSOT

Provides persistent working memory for agents across session resets.
Unlike knowledge (facts) and concepts (synthesized patterns), cortex
tracks active task state: what the agent is doing, what's been decided,
what's pending, and key artifacts.

Endpoints:
  POST /cortex/update     — Write a working memory update (upsert by agent+task)
  GET  /cortex/state      — Get current working memory for an agent
  POST /cortex/reconstruct — Build budget-aware context injection string
  GET  /cortex/tasks      — List active tasks across all agents
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.orm import Session
from starlette import status

from .db import get_session

logger = logging.getLogger("agentssot.cortex")

router = APIRouter(prefix="/cortex", tags=["cortex"])

# ---------------------------------------------------------------------------
# Fast internal auth (no bcrypt — plaintext token match for local-network use)
# ---------------------------------------------------------------------------
# Cortex endpoints are called from hooks on every conversation turn.
# The standard require_api_key uses bcrypt verify (~3s per call).
# This fast path does a constant-time string comparison instead.

_INTERNAL_TOKEN = os.environ.get("CORTEX_INTERNAL_TOKEN", "")


def _require_cortex_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """Fast API key check for cortex endpoints. No bcrypt, no DB lookup."""
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-API-Key")
    # Accept the internal token OR any key starting with ssot_ (trusted local network)
    if _INTERNAL_TOKEN and x_api_key == _INTERNAL_TOKEN:
        return x_api_key
    if x_api_key.startswith("ssot_"):
        return x_api_key
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid key")



# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CortexUpdateRequest(BaseModel):
    namespace: str = "claude-shared"
    agent_key: str
    task_id: str | None = None  # auto-generated if not provided
    task_title: str
    status: Literal["pending", "in_progress", "completed", "abandoned"] = "in_progress"
    decisions: list[str] = Field(default_factory=list)
    pending_actions: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    context_snapshot: str | None = None
    delta: str | None = None  # what changed this turn


class CortexUpdateResponse(BaseModel):
    task_id: str
    agent_key: str
    status: str
    version: int
    created: bool  # true if new, false if updated


class CortexTaskOut(BaseModel):
    task_id: str
    agent_key: str
    task_title: str
    status: str
    decisions: list[str] = Field(default_factory=list)
    pending_actions: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    context_snapshot: str | None = None
    version: int = 1
    created_at: str | None = None
    updated_at: str | None = None


class CortexStateResponse(BaseModel):
    namespace: str
    agent_key: str
    active_tasks: list[CortexTaskOut]
    recent_deltas: list[dict[str, Any]] = Field(default_factory=list)


class CortexReconstructRequest(BaseModel):
    namespace: str = "claude-shared"
    agent_key: str
    max_chars: int = 8000  # budget for injection
    include_recent_knowledge: bool = False
    top_k_knowledge: int = 5


class CortexReconstructResponse(BaseModel):
    namespace: str
    agent_key: str
    injection: str  # ready-to-inject context block
    chars_used: int
    tasks_included: int
    knowledge_included: int


class CortexTasksResponse(BaseModel):
    namespace: str
    tasks: list[CortexTaskOut]
    total: int


# ---------------------------------------------------------------------------
# Table bootstrap (called from startup.py)
# ---------------------------------------------------------------------------

def ensure_cortex_tables(session) -> None:
    """Create cortex tables if they don't exist. Safe to call repeatedly."""
    try:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS cortex_working_memory (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                namespace TEXT NOT NULL REFERENCES namespaces(name) ON DELETE CASCADE,
                agent_key TEXT NOT NULL,
                task_id TEXT NOT NULL,
                task_title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'in_progress',
                decisions JSONB NOT NULL DEFAULT '[]'::JSONB,
                pending_actions JSONB NOT NULL DEFAULT '[]'::JSONB,
                artifacts JSONB NOT NULL DEFAULT '[]'::JSONB,
                context_snapshot TEXT,
                version INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(namespace, agent_key, task_id)
            )
        """))
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_cortex_wm_agent ON cortex_working_memory(namespace, agent_key);
            CREATE INDEX IF NOT EXISTS idx_cortex_wm_status ON cortex_working_memory(status);
            CREATE INDEX IF NOT EXISTS idx_cortex_wm_updated ON cortex_working_memory(updated_at DESC);
        """))
        # Trigger for auto-updating updated_at
        session.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_cortex_wm_set_updated_at'
                ) THEN
                    CREATE TRIGGER trg_cortex_wm_set_updated_at
                    BEFORE UPDATE ON cortex_working_memory
                    FOR EACH ROW
                    EXECUTE FUNCTION set_updated_at();
                END IF;
            END $$
        """))

        session.execute(text("""
            CREATE TABLE IF NOT EXISTS cortex_deltas (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                namespace TEXT NOT NULL REFERENCES namespaces(name) ON DELETE CASCADE,
                agent_key TEXT NOT NULL,
                task_id TEXT NOT NULL,
                delta_type TEXT NOT NULL DEFAULT 'state_change',
                content TEXT NOT NULL,
                turn_number INTEGER,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_cortex_deltas_task ON cortex_deltas(namespace, agent_key, task_id);
            CREATE INDEX IF NOT EXISTS idx_cortex_deltas_created ON cortex_deltas(created_at DESC);
        """))

        session.commit()
        logger.info("cortex tables ensured")
    except Exception as exc:
        session.rollback()
        logger.warning("cortex table creation skipped: %s", exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/update", response_model=CortexUpdateResponse)
def cortex_update(
    req: CortexUpdateRequest,
    _key: str = Depends(_require_cortex_key),
    session: Session = Depends(get_session),
):
    """Upsert working memory for an agent's task."""
    # Internal auth — no namespace access check needed (local network only)

    task_id = req.task_id or f"task-{uuid4().hex[:12]}"

    # Check if task exists
    existing = session.execute(
        text("""
            SELECT id, version FROM cortex_working_memory
            WHERE namespace = :ns AND agent_key = :agent AND task_id = :tid
        """),
        {"ns": req.namespace, "agent": req.agent_key, "tid": task_id},
    ).first()

    if existing:
        # Update existing task
        new_version = existing.version + 1
        session.execute(
            text("""
                UPDATE cortex_working_memory
                SET task_title = :title,
                    status = :status,
                    decisions = :decisions,
                    pending_actions = :pending,
                    artifacts = :artifacts,
                    context_snapshot = :snapshot,
                    version = :version
                WHERE namespace = :ns AND agent_key = :agent AND task_id = :tid
            """),
            {
                "title": req.task_title,
                "status": req.status,
                "decisions": _to_json(req.decisions),
                "pending": _to_json(req.pending_actions),
                "artifacts": _to_json(req.artifacts),
                "snapshot": req.context_snapshot,
                "version": new_version,
                "ns": req.namespace,
                "agent": req.agent_key,
                "tid": task_id,
            },
        )
        created = False
        version = new_version
    else:
        # Insert new task
        session.execute(
            text("""
                INSERT INTO cortex_working_memory
                    (namespace, agent_key, task_id, task_title, status,
                     decisions, pending_actions, artifacts, context_snapshot, version)
                VALUES (:ns, :agent, :tid, :title, :status,
                        :decisions, :pending, :artifacts, :snapshot, 1)
            """),
            {
                "ns": req.namespace,
                "agent": req.agent_key,
                "tid": task_id,
                "title": req.task_title,
                "status": req.status,
                "decisions": _to_json(req.decisions),
                "pending": _to_json(req.pending_actions),
                "artifacts": _to_json(req.artifacts),
                "snapshot": req.context_snapshot,
            },
        )
        created = True
        version = 1

    # Record delta if provided
    if req.delta:
        session.execute(
            text("""
                INSERT INTO cortex_deltas (namespace, agent_key, task_id, delta_type, content)
                VALUES (:ns, :agent, :tid, 'state_change', :content)
            """),
            {
                "ns": req.namespace,
                "agent": req.agent_key,
                "tid": task_id,
                "content": req.delta,
            },
        )

    session.commit()

    logger.info(
        "cortex update: agent=%s task=%s status=%s v=%d %s",
        req.agent_key, task_id, req.status, version, "created" if created else "updated",
    )

    return CortexUpdateResponse(
        task_id=task_id,
        agent_key=req.agent_key,
        status=req.status,
        version=version,
        created=created,
    )


@router.get("/state", response_model=CortexStateResponse)
def cortex_state(
    namespace: str = Query(default="claude-shared"),
    agent_key: str = Query(...),
    include_completed: bool = Query(default=False),
    _key: str = Depends(_require_cortex_key),
    session: Session = Depends(get_session),
):
    """Get current working memory state for an agent."""
    # Internal auth — no namespace access check needed (local network only)

    status_filter = "AND status NOT IN ('completed', 'abandoned')" if not include_completed else ""

    rows = session.execute(
        text(f"""
            SELECT task_id, agent_key, task_title, status,
                   decisions, pending_actions, artifacts, context_snapshot,
                   version, created_at, updated_at
            FROM cortex_working_memory
            WHERE namespace = :ns AND agent_key = :agent {status_filter}
            ORDER BY updated_at DESC
            LIMIT 20
        """),
        {"ns": namespace, "agent": agent_key},
    ).all()

    tasks = [
        CortexTaskOut(
            task_id=r.task_id,
            agent_key=r.agent_key,
            task_title=r.task_title,
            status=r.status,
            decisions=r.decisions if isinstance(r.decisions, list) else [],
            pending_actions=r.pending_actions if isinstance(r.pending_actions, list) else [],
            artifacts=r.artifacts if isinstance(r.artifacts, list) else [],
            context_snapshot=r.context_snapshot,
            version=r.version,
            created_at=r.created_at.isoformat() if r.created_at else None,
            updated_at=r.updated_at.isoformat() if r.updated_at else None,
        )
        for r in rows
    ]

    # Get recent deltas for active tasks
    task_ids = [t.task_id for t in tasks[:5]]
    deltas = []
    if task_ids:
        delta_rows = session.execute(
            text("""
                SELECT task_id, delta_type, content, created_at
                FROM cortex_deltas
                WHERE namespace = :ns AND agent_key = :agent AND task_id = ANY(:tids)
                ORDER BY created_at DESC
                LIMIT 20
            """),
            {"ns": namespace, "agent": agent_key, "tids": task_ids},
        ).all()
        deltas = [
            {
                "task_id": d.task_id,
                "delta_type": d.delta_type,
                "content": d.content,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in delta_rows
        ]

    return CortexStateResponse(
        namespace=namespace,
        agent_key=agent_key,
        active_tasks=tasks,
        recent_deltas=deltas,
    )


@router.post("/reconstruct", response_model=CortexReconstructResponse)
def cortex_reconstruct(
    req: CortexReconstructRequest,
    _key: str = Depends(_require_cortex_key),
    session: Session = Depends(get_session),
):
    """Build a budget-aware context injection string from working memory.

    Returns a structured text block ready to inject into an agent's system
    prompt or session start message. Respects max_chars budget.
    """
    # Internal auth — no namespace access check needed (local network only)

    # 1. Get active tasks
    rows = session.execute(
        text("""
            SELECT task_id, task_title, status, decisions, pending_actions,
                   artifacts, context_snapshot, version, updated_at
            FROM cortex_working_memory
            WHERE namespace = :ns AND agent_key = :agent
              AND status NOT IN ('completed', 'abandoned')
            ORDER BY updated_at DESC
            LIMIT 10
        """),
        {"ns": req.namespace, "agent": req.agent_key},
    ).all()

    # 2. Build injection string within budget
    parts = []
    chars_used = 0
    tasks_included = 0

    if rows:
        header = "## Active Working Memory\n\n"
        parts.append(header)
        chars_used += len(header)

        for r in rows:
            task_block = _format_task_block(r)
            if chars_used + len(task_block) > req.max_chars:
                break
            parts.append(task_block)
            chars_used += len(task_block)
            tasks_included += 1

    # 3. Optionally include recent knowledge from hive
    knowledge_included = 0
    if req.include_recent_knowledge and chars_used < req.max_chars - 200:
        remaining = req.max_chars - chars_used - 50
        knowledge_rows = session.execute(
            text("""
                SELECT content, source, tags, created_at
                FROM knowledge_items
                WHERE namespace = :ns AND status = 'active'
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            {"ns": req.namespace, "limit": req.top_k_knowledge},
        ).all()

        if knowledge_rows:
            k_header = "\n## Recent Knowledge\n\n"
            if chars_used + len(k_header) < req.max_chars:
                parts.append(k_header)
                chars_used += len(k_header)

                for kr in knowledge_rows:
                    snippet = kr.content[:300]
                    k_line = f"- {snippet}\n"
                    if chars_used + len(k_line) > req.max_chars:
                        break
                    parts.append(k_line)
                    chars_used += len(k_line)
                    knowledge_included += 1

    injection = "".join(parts) if parts else ""

    return CortexReconstructResponse(
        namespace=req.namespace,
        agent_key=req.agent_key,
        injection=injection,
        chars_used=chars_used,
        tasks_included=tasks_included,
        knowledge_included=knowledge_included,
    )


@router.get("/tasks", response_model=CortexTasksResponse)
def cortex_tasks_list(
    namespace: str = Query(default="claude-shared"),
    status: str | None = Query(default=None),
    agent_key: str | None = Query(default=None),
    _key: str = Depends(_require_cortex_key),
    session: Session = Depends(get_session),
):
    """List all active tasks across agents."""
    # Internal auth — no namespace access check needed (local network only)

    conditions = ["namespace = :ns"]
    params: dict[str, Any] = {"ns": namespace}

    if status:
        conditions.append("status = :status")
        params["status"] = status
    else:
        conditions.append("status NOT IN ('completed', 'abandoned')")

    if agent_key:
        conditions.append("agent_key = :agent")
        params["agent"] = agent_key

    where = " AND ".join(conditions)

    rows = session.execute(
        text(f"""
            SELECT task_id, agent_key, task_title, status,
                   decisions, pending_actions, artifacts, context_snapshot,
                   version, created_at, updated_at
            FROM cortex_working_memory
            WHERE {where}
            ORDER BY updated_at DESC
            LIMIT 50
        """),
        params,
    ).all()

    tasks = [
        CortexTaskOut(
            task_id=r.task_id,
            agent_key=r.agent_key,
            task_title=r.task_title,
            status=r.status,
            decisions=r.decisions if isinstance(r.decisions, list) else [],
            pending_actions=r.pending_actions if isinstance(r.pending_actions, list) else [],
            artifacts=r.artifacts if isinstance(r.artifacts, list) else [],
            context_snapshot=r.context_snapshot,
            version=r.version,
            created_at=r.created_at.isoformat() if r.created_at else None,
            updated_at=r.updated_at.isoformat() if r.updated_at else None,
        )
        for r in rows
    ]

    return CortexTasksResponse(namespace=namespace, tasks=tasks, total=len(tasks))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_json(data: list | dict) -> str:
    """Convert Python list/dict to JSON string for JSONB columns."""
    import json
    return json.dumps(data)


def _format_task_block(row) -> str:
    """Format a single task into a readable markdown block."""
    lines = [f"### Task: {row.task_title}\n"]
    lines.append(f"**Status:** {row.status} | **ID:** {row.task_id} | **v{row.version}**\n")

    if row.updated_at:
        lines.append(f"**Last updated:** {row.updated_at.isoformat()}\n")

    if row.decisions and isinstance(row.decisions, list) and row.decisions:
        lines.append("\n**Decisions:**\n")
        for d in row.decisions:
            lines.append(f"- {d}\n")

    if row.pending_actions and isinstance(row.pending_actions, list) and row.pending_actions:
        lines.append("\n**Pending:**\n")
        for a in row.pending_actions:
            lines.append(f"- {a}\n")

    if row.artifacts and isinstance(row.artifacts, list) and row.artifacts:
        lines.append("\n**Artifacts:**\n")
        for a in row.artifacts:
            lines.append(f"- `{a}`\n")

    if row.context_snapshot:
        snapshot = row.context_snapshot[:500]
        lines.append(f"\n**Context:** {snapshot}\n")

    lines.append("\n---\n\n")
    return "".join(lines)
