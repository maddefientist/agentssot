"""Review queue helpers."""
from __future__ import annotations

from sqlalchemy import select
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
    from datetime import datetime, timezone
    item = session.get(ReviewQueueItem, queue_id)
    if item is None:
        return None
    item.status = ReviewQueueStatus.dismissed
    item.resolved_at = datetime.now(timezone.utc)
    item.resolved_by = by
    session.commit()
    return item
