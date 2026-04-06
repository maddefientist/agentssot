"""Milestone 10: Sync state tracking and conflict detection.

Provides per-device sync cursors and duplicate detection for multi-device
memory synchronization. All endpoints are gated behind SYNC_TRACKING_ENABLED.

Conflict detection flags potential duplicates when two devices ingest items
with the same content hash within a configurable time window. Resolution is
left to the client.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select, text
from sqlalchemy.orm import Session

from .db import get_session
from .models import ApiRole, KnowledgeItem
from .security import AuthContext, ensure_namespace_access, require_api_key
from .settings import get_settings

logger = logging.getLogger("agentssot.sync")
settings = get_settings()

router = APIRouter(prefix="/sync", tags=["sync"])

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SyncCheckpointRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=200)
    namespace: str = "claude-shared"
    last_synced_item_id: str = Field(min_length=1)


class SyncCheckpointResponse(BaseModel):
    device_id: str
    namespace: str
    last_synced_item_id: str
    last_synced_at: datetime


class PendingConflict(BaseModel):
    """Two items ingested by different devices with the same content hash."""
    content_hash: str
    item_ids: list[str]
    device_ids: list[str]
    ingested_at: list[datetime]


class SyncPendingResponse(BaseModel):
    device_id: str
    namespace: str
    pending_count: int
    pending_items: list[dict]
    conflicts: list[PendingConflict] = Field(default_factory=list)


class SyncStatusDevice(BaseModel):
    namespace: str
    last_synced_item_id: str | None
    last_synced_at: datetime | None
    pending_count: int


class SyncStatusResponse(BaseModel):
    device_id: str
    namespaces: list[SyncStatusDevice]


# ---------------------------------------------------------------------------
# Feature gate helper
# ---------------------------------------------------------------------------


def _require_sync_enabled():
    if not settings.sync_tracking_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sync tracking is not enabled (SYNC_TRACKING_ENABLED=false)",
        )


# ---------------------------------------------------------------------------
# Table bootstrap (called from startup.py)
# ---------------------------------------------------------------------------


def ensure_sync_tables(session: Session) -> None:
    """Create sync_checkpoints table if it does not exist. Safe to call repeatedly."""
    try:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS sync_checkpoints (
                id SERIAL PRIMARY KEY,
                device_id TEXT NOT NULL,
                namespace TEXT NOT NULL REFERENCES namespaces(name) ON DELETE CASCADE,
                last_synced_item_id UUID NOT NULL,
                last_synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(device_id, namespace)
            )
        """))
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_sync_checkpoints_device
            ON sync_checkpoints(device_id, namespace)
        """))
        session.commit()
        logger.info("ensured sync_checkpoints table")
    except Exception as exc:
        session.rollback()
        logger.warning("sync_checkpoints table creation skipped: %s", exc)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------


def _upsert_checkpoint(
    session: Session,
    device_id: str,
    namespace: str,
    last_synced_item_id: str,
) -> dict:
    """Insert or update a sync checkpoint for a device+namespace."""
    now = datetime.now(timezone.utc)
    session.execute(text("""
        INSERT INTO sync_checkpoints (device_id, namespace, last_synced_item_id, last_synced_at)
        VALUES (:device_id, :namespace, :item_id, :now)
        ON CONFLICT (device_id, namespace)
        DO UPDATE SET
            last_synced_item_id = EXCLUDED.last_synced_item_id,
            last_synced_at = EXCLUDED.last_synced_at
    """), {
        "device_id": device_id,
        "namespace": namespace,
        "item_id": last_synced_item_id,
        "now": now,
    })
    session.commit()
    return {
        "device_id": device_id,
        "namespace": namespace,
        "last_synced_item_id": last_synced_item_id,
        "last_synced_at": now,
    }


def _get_checkpoint(session: Session, device_id: str, namespace: str) -> dict | None:
    """Get the current checkpoint for a device+namespace."""
    row = session.execute(text("""
        SELECT device_id, namespace, last_synced_item_id, last_synced_at
        FROM sync_checkpoints
        WHERE device_id = :device_id AND namespace = :namespace
    """), {"device_id": device_id, "namespace": namespace}).first()
    if not row:
        return None
    return {
        "device_id": row.device_id,
        "namespace": row.namespace,
        "last_synced_item_id": str(row.last_synced_item_id),
        "last_synced_at": row.last_synced_at,
    }


def _get_pending_items(
    session: Session,
    device_id: str,
    namespace: str,
    limit: int = 100,
) -> tuple[list[dict], int]:
    """Return knowledge items created after the device's last checkpoint.

    Returns (items, total_pending_count).
    """
    checkpoint = _get_checkpoint(session, device_id, namespace)

    if checkpoint:
        # Get the created_at of the checkpoint item to use as a time boundary
        checkpoint_item = session.execute(
            select(KnowledgeItem.created_at).where(
                KnowledgeItem.id == UUID(checkpoint["last_synced_item_id"])
            )
        ).scalar()
        if checkpoint_item:
            filter_cond = and_(
                KnowledgeItem.namespace == namespace,
                KnowledgeItem.created_at > checkpoint_item,
            )
        else:
            # Checkpoint item was deleted; fall back to checkpoint timestamp
            filter_cond = and_(
                KnowledgeItem.namespace == namespace,
                KnowledgeItem.created_at > checkpoint["last_synced_at"],
            )
    else:
        # No checkpoint — everything is pending
        filter_cond = KnowledgeItem.namespace == namespace

    total = session.scalar(
        select(func.count()).select_from(KnowledgeItem).where(filter_cond)
    ) or 0

    rows = session.execute(
        select(
            KnowledgeItem.id,
            KnowledgeItem.content,
            KnowledgeItem.source,
            KnowledgeItem.tags,
            KnowledgeItem.memory_type,
            KnowledgeItem.created_at,
        )
        .where(filter_cond)
        .order_by(KnowledgeItem.created_at.asc())
        .limit(limit)
    ).all()

    items = [
        {
            "id": str(r.id),
            "content": r.content[:200],  # snippet only
            "source": r.source,
            "tags": list(r.tags or []),
            "memory_type": r.memory_type,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
    return items, total


def _detect_conflicts(
    session: Session,
    namespace: str,
    window_hours: int = 24,
) -> list[PendingConflict]:
    """Detect potential duplicate items ingested by different devices within a time window.

    Uses content hash (first 200 chars) + source to detect when two devices
    ingested semantically identical content.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)

    # Find items from different sources with identical content in the window
    rows = session.execute(text("""
        WITH hashed AS (
            SELECT
                id,
                md5(left(content, 200)) AS content_hash,
                source,
                created_at
            FROM knowledge_items
            WHERE namespace = :namespace
              AND created_at > :cutoff
              AND source IS NOT NULL
        )
        SELECT content_hash, array_agg(id::text) AS item_ids,
               array_agg(DISTINCT source) AS sources,
               array_agg(created_at) AS created_ats
        FROM hashed
        GROUP BY content_hash
        HAVING count(DISTINCT source) > 1
        LIMIT 50
    """), {"namespace": namespace, "cutoff": cutoff}).all()

    conflicts = []
    for row in rows:
        conflicts.append(PendingConflict(
            content_hash=row.content_hash,
            item_ids=list(row.item_ids),
            device_ids=list(row.sources),
            ingested_at=list(row.created_ats),
        ))
    return conflicts


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@router.post("/checkpoint", response_model=SyncCheckpointResponse)
def record_checkpoint(
    payload: SyncCheckpointRequest,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Record a sync checkpoint for a device after successful sync."""
    _require_sync_enabled()
    ensure_namespace_access(
        auth, payload.namespace,
        {ApiRole.writer.value, ApiRole.admin.value},
    )
    result = _upsert_checkpoint(
        session, payload.device_id, payload.namespace, payload.last_synced_item_id,
    )
    return SyncCheckpointResponse(**result)


@router.get("/pending", response_model=SyncPendingResponse)
def get_pending(
    device_id: str = Query(..., min_length=1),
    namespace: str = Query(default="claude-shared"),
    limit: int = Query(default=100, ge=1, le=500),
    include_conflicts: bool = Query(default=False),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Return items newer than a device's last checkpoint, with optional conflict detection."""
    _require_sync_enabled()
    ensure_namespace_access(
        auth, namespace,
        {ApiRole.reader.value, ApiRole.writer.value, ApiRole.admin.value},
    )
    items, total = _get_pending_items(session, device_id, namespace, limit)

    conflicts = []
    if include_conflicts:
        conflicts = _detect_conflicts(session, namespace)

    return SyncPendingResponse(
        device_id=device_id,
        namespace=namespace,
        pending_count=total,
        pending_items=items,
        conflicts=conflicts,
    )


@router.get("/status", response_model=SyncStatusResponse)
def get_status(
    device_id: str = Query(..., min_length=1),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Return sync status for a device across all accessible namespaces."""
    _require_sync_enabled()

    # Get all checkpoints for this device
    rows = session.execute(text("""
        SELECT device_id, namespace, last_synced_item_id, last_synced_at
        FROM sync_checkpoints
        WHERE device_id = :device_id
    """), {"device_id": device_id}).all()

    namespaces = []
    for row in rows:
        # Check access
        ns = row.namespace
        if ns not in auth.namespaces and "*" not in auth.namespaces:
            continue

        # Count pending for each namespace
        _, pending = _get_pending_items(session, device_id, ns, limit=0)
        # Re-count without limit=0 trick: just get the count
        checkpoint = _get_checkpoint(session, device_id, ns)

        namespaces.append(SyncStatusDevice(
            namespace=ns,
            last_synced_item_id=str(row.last_synced_item_id) if row.last_synced_item_id else None,
            last_synced_at=row.last_synced_at,
            pending_count=pending,
        ))

    # Also include namespaces without checkpoints that the device has access to
    accessible = set(auth.namespaces) if "*" not in auth.namespaces else set()
    checkpointed = {ns.namespace for ns in namespaces}

    # For wildcard access, we don't enumerate all namespaces — too expensive
    if "*" not in auth.namespaces:
        for ns in accessible - checkpointed:
            total = session.scalar(
                select(func.count()).select_from(KnowledgeItem).where(
                    KnowledgeItem.namespace == ns
                )
            ) or 0
            namespaces.append(SyncStatusDevice(
                namespace=ns,
                last_synced_item_id=None,
                last_synced_at=None,
                pending_count=total,
            ))

    return SyncStatusResponse(device_id=device_id, namespaces=namespaces)
