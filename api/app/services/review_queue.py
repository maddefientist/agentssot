"""Review queue helpers."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import ReviewQueueItem, ReviewQueueKind, ReviewQueueStatus


def list_pending(session: Session, namespace: str | None, kind: str | None,
                 limit: int = 100) -> list[ReviewQueueItem]:
    stmt = select(ReviewQueueItem).where(ReviewQueueItem.status == ReviewQueueStatus.pending)
    if namespace:
        stmt = stmt.where(ReviewQueueItem.namespace == namespace)
    if kind:
        stmt = stmt.where(ReviewQueueItem.kind == ReviewQueueKind(kind))
    stmt = stmt.order_by(ReviewQueueItem.priority.desc(), ReviewQueueItem.created_at.desc()).limit(limit)
    return list(session.execute(stmt).scalars())


def resolve(session: Session, queue_id: str, by: str | None = None) -> ReviewQueueItem | None:
    from datetime import datetime, timezone
    item = session.get(ReviewQueueItem, queue_id)
    if item is None:
        return None
    item.status = ReviewQueueStatus.resolved
    item.resolved_at = datetime.now(timezone.utc)
    item.resolved_by = by
    session.commit()
    return item


def dismiss(session: Session, queue_id: str, by: str | None = None) -> ReviewQueueItem | None:
    item = session.get(ReviewQueueItem, queue_id)
    if item is None:
        return None
    item.status = ReviewQueueStatus.dismissed
    item.resolved_at = datetime.now(timezone.utc)
    item.resolved_by = by
    session.commit()
    return item


def drain_duplicates(session: Session, namespace: str | None = None,
                     by: str | None = "auto-drainer") -> dict[str, int]:
    """Collapse duplicate *pending* queue rows.

    The backfill contradiction sweep queued the same (kind, primary, secondary)
    pair many times (e.g. one primary queued 5×). For each identical signature we
    keep the oldest, highest-priority row and dismiss the rest as duplicates. This
    drains the queue's self-inflicted noise WITHOUT touching any knowledge item —
    dismissing a queue entry only marks the review task done, never deletes data.

    Returns counts: {scanned, groups, dismissed}.
    """
    stmt = select(ReviewQueueItem).where(ReviewQueueItem.status == ReviewQueueStatus.pending)
    if namespace:
        stmt = stmt.where(ReviewQueueItem.namespace == namespace)
    # Stable order so "the survivor" is deterministic: oldest first, then highest priority.
    stmt = stmt.order_by(ReviewQueueItem.created_at.asc(), ReviewQueueItem.priority.desc())
    rows = list(session.execute(stmt).scalars())

    seen: set[tuple] = set()
    dismissed = 0
    now = datetime.now(timezone.utc)
    for row in rows:
        sig = (
            row.namespace,
            row.kind.value if hasattr(row.kind, "value") else str(row.kind),
            str(row.primary_id),
            str(row.secondary_id) if row.secondary_id else None,
        )
        if sig in seen:
            row.status = ReviewQueueStatus.dismissed
            row.resolved_at = now
            row.resolved_by = by
            row.reason = (row.reason or "") + " [auto-dismissed: duplicate queue entry]"
            dismissed += 1
        else:
            seen.add(sig)
    if dismissed:
        session.commit()
    return {"scanned": len(rows), "groups": len(seen), "dismissed": dismissed}


def drain_stale(session: Session, older_than_days: int, kind: str | None = None,
                namespace: str | None = None, by: str | None = "auto-drainer") -> dict[str, int]:
    """Dismiss pending rows older than ``older_than_days``.

    EXPLICIT operator action — never call this automatically. Dismissing a stale
    contradiction means "we accept these items coexist"; that is a human judgment,
    so this is only invoked when the operator asks (e.g. to clear a one-off backfill
    sweep). Touches only queue rows, never knowledge items. Returns {dismissed}.
    """
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    stmt = select(ReviewQueueItem).where(
        ReviewQueueItem.status == ReviewQueueStatus.pending,
        ReviewQueueItem.created_at < cutoff,
    )
    if kind:
        stmt = stmt.where(ReviewQueueItem.kind == ReviewQueueKind(kind))
    if namespace:
        stmt = stmt.where(ReviewQueueItem.namespace == namespace)
    rows = list(session.execute(stmt).scalars())
    now = datetime.now(timezone.utc)
    for row in rows:
        row.status = ReviewQueueStatus.dismissed
        row.resolved_at = now
        row.resolved_by = by
        row.reason = (row.reason or "") + f" [auto-dismissed: stale >{older_than_days}d]"
    if rows:
        session.commit()
    return {"dismissed": len(rows)}


def queue_counts(session: Session, namespace: str | None = None) -> dict[str, int]:
    """Pending-row counts per kind — feeds the GUI badge and drain preview."""
    stmt = (
        select(ReviewQueueItem.kind, func.count())
        .where(ReviewQueueItem.status == ReviewQueueStatus.pending)
        .group_by(ReviewQueueItem.kind)
    )
    if namespace:
        stmt = stmt.where(ReviewQueueItem.namespace == namespace)
    out: dict[str, int] = {}
    for kind, count in session.execute(stmt):
        out[kind.value if hasattr(kind, "value") else str(kind)] = int(count)
    return out
