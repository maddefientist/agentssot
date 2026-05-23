from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import and_, func, select, text, update
from sqlalchemy.orm import Session

from ..db import get_session
from ..security import AuthContext, require_api_key
from .models import SynapseEvent, SynapseSession
from .schemas import CollisionOut, EventCreate, EventOut, SessionHeartbeat, SessionOut, SessionRegister
from . import listener as _listener

logger = logging.getLogger("agentssot.synapse")

router = APIRouter(prefix="/synapse", tags=["synapse"])

# Kinds that update current_file/op on parent session
_FILE_OP_KINDS = {"edit", "write", "bash"}


def _session_or_404(session: Session, session_id: str) -> SynapseSession:
    row = session.get(SynapseSession, session_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return row


@router.post("/session", response_model=SessionOut)
def register_session(
    payload: SessionRegister,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
) -> SessionOut:
    """Upsert a session by session_id."""
    existing = session.get(SynapseSession, payload.session_id)
    if existing:
        existing.host = payload.host
        existing.cwd = payload.cwd
        existing.repo = payload.repo
        existing.agent = payload.agent
        existing.last_seen = datetime.now(timezone.utc)
        if payload.current_file is not None:
            existing.current_file = payload.current_file
        if payload.current_op is not None:
            existing.current_op = payload.current_op
        session.commit()
        session.refresh(existing)
        return SessionOut.model_validate(existing)

    row = SynapseSession(
        session_id=payload.session_id,
        host=payload.host,
        cwd=payload.cwd,
        repo=payload.repo,
        agent=payload.agent,
        current_file=payload.current_file,
        current_op=payload.current_op,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return SessionOut.model_validate(row)


@router.post("/heartbeat", response_model=SessionOut)
def heartbeat(
    payload: SessionHeartbeat,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
) -> SessionOut:
    """Update last_seen and optionally current_file/op."""
    row = _session_or_404(session, payload.session_id)
    row.last_seen = datetime.now(timezone.utc)
    if payload.current_file is not None:
        row.current_file = payload.current_file
    if payload.current_op is not None:
        row.current_op = payload.current_op
    session.commit()
    session.refresh(row)
    return SessionOut.model_validate(row)


@router.post("/event", response_model=EventOut)
def create_event(
    payload: EventCreate,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
) -> EventOut:
    """Insert an event; update parent session's last_seen and current_file/op for file ops."""
    row = _session_or_404(session, payload.session_id)

    event = SynapseEvent(
        session_id=payload.session_id,
        kind=payload.kind,
        file=payload.file,
        line_start=payload.line_start,
        line_end=payload.line_end,
        payload=payload.payload,
    )
    session.add(event)

    # Touch parent session
    row.last_seen = datetime.now(timezone.utc)
    if payload.kind in _FILE_OP_KINDS:
        if payload.file is not None:
            row.current_file = payload.file
        row.current_op = payload.kind

    session.commit()
    session.refresh(event)
    return EventOut.model_validate(event)


@router.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_session(
    session_id: str,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
) -> Response:
    """Delete a session and cascade its events."""
    row = _session_or_404(session, session_id)
    session.delete(row)
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/active", response_model=list[SessionOut])
def list_active(
    repo: str | None = Query(default=None),
    host: str | None = Query(default=None),
    since_seconds: int = Query(default=600, ge=1),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
) -> list[SessionOut]:
    """List sessions active within since_seconds."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=since_seconds)
    q = select(SynapseSession).where(SynapseSession.last_seen > cutoff)
    if repo is not None:
        q = q.where(SynapseSession.repo == repo)
    if host is not None:
        q = q.where(SynapseSession.host == host)
    q = q.order_by(SynapseSession.last_seen.desc())
    rows = session.scalars(q).all()
    return [SessionOut.model_validate(r) for r in rows]


@router.get("/collisions", response_model=list[CollisionOut])
def list_collisions(
    file: str = Query(..., description="File path to check for concurrent access"),
    exclude_session: str | None = Query(default=None),
    window_seconds: int = Query(default=300, ge=1),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
) -> list[CollisionOut]:
    """List sessions (other than exclude_session) that have touched file in window_seconds.

    Uses DISTINCT ON (session_id) to guarantee exactly one row per session even
    when multiple events share the same millisecond timestamp.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

    # DISTINCT ON (session_id) ordered by ts DESC guarantees one row per session —
    # the most-recent event — even on same-ms ties.
    # NOTE: exclude_clause must reference the bare column name (no alias) because
    # the alias `e` only exists in the outer query, not inside the subquery.
    exclude_clause = ""
    if exclude_session:
        exclude_clause = "AND session_id != :exclude_session"

    raw_sql = text(f"""
        SELECT
            s.session_id,
            s.host,
            s.cwd,
            e.ts AS last_event_ts,
            e.kind
        FROM (
            SELECT DISTINCT ON (session_id)
                session_id,
                ts,
                kind
            FROM synapse_event
            WHERE file = :file
              AND ts > :cutoff
              {exclude_clause}
            ORDER BY session_id, ts DESC
        ) e
        JOIN synapse_session s ON s.session_id = e.session_id
        ORDER BY e.ts DESC
    """)

    params: dict = {"file": file, "cutoff": cutoff}
    if exclude_session:
        params["exclude_session"] = exclude_session

    rows = session.execute(raw_sql, params).all()
    return [
        CollisionOut(
            session_id=r.session_id,
            host=r.host,
            cwd=r.cwd,
            last_event_ts=r.last_event_ts,
            kind=r.kind,
        )
        for r in rows
    ]


_HEARTBEAT_INTERVAL = 15  # seconds between SSE pings


@router.get("/stream")
async def synapse_stream(
    repo: str | None = Query(default=None),
    host: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    file: str | None = Query(default=None),
    exclude_session: str | None = Query(default=None),
    auth: AuthContext = Depends(require_api_key),
) -> StreamingResponse:
    """Server-Sent Events stream of synapse_event INSERTs.

    Filters: repo, host, session_id, file — all optional and ANDed.
    exclude_session: skip events from this session_id (so a session does not
    receive its own echoes).

    Framing:
    - First message: ``event: ready\\ndata: {"subscriber_id": "..."}\\n\\n``
    - Heartbeat every 15s: ``: ping\\n\\n``
    - Events: ``data: <json>\\n\\n``
    - Overflow marker: ``data: {"kind": "overflow"}\\n\\n``
    """
    filters: dict = {}
    if repo is not None:
        filters["repo"] = repo
    if host is not None:
        filters["host"] = host
    if session_id is not None:
        filters["session_id"] = session_id
    if file is not None:
        filters["file"] = file

    queue: asyncio.Queue = asyncio.Queue(maxsize=_listener._QUEUE_MAXSIZE)
    sub = _listener.Subscriber(
        queue=queue,
        filters=filters,
        exclude_session_id=exclude_session,
    )
    sub_id = await _listener.register(sub)

    async def event_generator():
        try:
            # Ready message
            yield f"event: ready\ndata: {json.dumps({'subscriber_id': sub_id})}\n\n"

            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_INTERVAL)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat comment to defeat idle proxies
                    yield ": ping\n\n"
                except asyncio.CancelledError:
                    break

        except asyncio.CancelledError:
            pass
        finally:
            await _listener.unregister(sub_id)
            logger.debug("SSE stream closed subscriber_id=%s", sub_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
