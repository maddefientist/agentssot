"""Review queue helpers."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select, text
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


def audit_supersede(session: Session, namespace: str | None = None,
                    threshold: float = 0.80, dry_run: bool = True,
                    by: str | None = "auto-audit") -> dict:
    """Reverse false-positive supersessions; resolve genuine ones.

    The supersession detector fires on (same memory_type AND shared entity_ref),
    so two unrelated notes about the same project get falsely paired — and
    apply_supersession already HID the old item (superseded_by), decayed its
    confidence x0.3, and set a 30d expiry. We distinguish genuine "v1->v2"
    updates from entity-only false matches by embedding similarity between the
    pair: calibration shows genuine updates sit >= 0.80, false matches below.

    For sim < threshold: reverse on the old item (restore superseded_by=NULL,
    expires_at=NULL, undo the x0.3 confidence decay) and dismiss the queue row.
    For sim >= threshold: leave the supersession applied and resolve the row.
    Touches only the wrongly-hidden old items; never deletes anything.
    """
    rows = session.execute(text(
        """
        SELECT rq.id AS qid, rq.secondary_id AS sid,
               1 - (p.embedding <=> s.embedding) AS sim
        FROM review_queue rq
        JOIN knowledge_items p ON p.id = rq.primary_id
        JOIN knowledge_items s ON s.id = rq.secondary_id
        WHERE rq.status = 'pending' AND rq.kind = 'supersede'
          AND p.embedding IS NOT NULL AND s.embedding IS NOT NULL
          AND (CAST(:ns AS text) IS NULL OR rq.namespace = :ns)
        """
    ), {"ns": namespace}).mappings().all()

    genuine = 0
    reversed_ = 0
    restored_sample: list[str] = []
    now = datetime.now(timezone.utc)
    for r in rows:
        if r["sim"] is not None and r["sim"] >= threshold:
            genuine += 1
            if not dry_run:
                resolve(session, str(r["qid"]), by)
        else:
            reversed_ += 1
            if len(restored_sample) < 8:
                restored_sample.append(str(r["sid"]))
            if not dry_run:
                # Undo apply_supersession on the wrongly-hidden old item.
                session.execute(text(
                    """
                    UPDATE knowledge_items
                    SET superseded_by = NULL,
                        expires_at = NULL,
                        confidence = LEAST(1.0, confidence / 0.3)
                    WHERE id = :sid AND superseded_by IS NOT NULL
                    """
                ), {"sid": r["sid"]})
                item = session.get(ReviewQueueItem, r["qid"])
                if item:
                    item.status = ReviewQueueStatus.dismissed
                    item.resolved_at = now
                    item.resolved_by = by
                    item.reason = (item.reason or "") + f" [reversed: false supersession, sim={r['sim']:.3f}]"
    if not dry_run:
        session.commit()
    return {
        "scanned": len(rows),
        "threshold": threshold,
        "genuine_resolved": genuine,
        "false_reversed": reversed_,
        "restored_sample": restored_sample,
        "dry_run": dry_run,
    }


def reclassify_low_conf(session: Session, namespace: str | None = None,
                        limit: int = 100, min_confidence: float = 0.6,
                        dry_run: bool = True, by: str | None = "auto-reclassify") -> dict:
    """Re-run the (now-healthy) classifier over pending low_conf items.

    The 2026-05-01/02 backfill left hundreds of items with no/low-confidence
    memory_type because the classifier was unreachable. With a confident result
    we set the type and resolve the row; otherwise the row stays pending.
    """
    from app.llm.classifier import classify
    from app.models import KnowledgeItem

    rows = session.execute(text(
        """
        SELECT rq.id AS qid, rq.primary_id AS kid
        FROM review_queue rq
        WHERE rq.status = 'pending' AND rq.kind = 'low_conf'
          AND (CAST(:ns AS text) IS NULL OR rq.namespace = :ns)
        ORDER BY rq.created_at
        LIMIT :lim
        """
    ), {"ns": namespace, "lim": limit}).mappings().all()

    typed = 0
    still_low = 0
    now = datetime.now(timezone.utc)
    for r in rows:
        ki = session.get(KnowledgeItem, r["kid"])
        if ki is None:
            continue
        out = classify(ki.content, tags=list(ki.tags or []), hint=ki.memory_type)
        conf = float(out.get("confidence", 0.0) or 0.0)
        new_type = out.get("memory_type")
        if conf >= min_confidence and new_type:
            typed += 1
            if not dry_run:
                ki.memory_type = new_type
                ki.last_classified_at = now
                resolve(session, str(r["qid"]), by)
        else:
            still_low += 1
    if not dry_run:
        session.commit()
    return {
        "scanned": len(rows),
        "min_confidence": min_confidence,
        "typed_and_resolved": typed,
        "still_low_conf": still_low,
        "dry_run": dry_run,
    }


def reclassify_untyped(session: Session, namespace: str | None = None,
                       limit: int = 200, min_confidence: float = 0.6,
                       dry_run: bool = True) -> dict:
    """Type knowledge items that have no memory_type and were never classified.

    Corpus-wide counterpart to reclassify_low_conf: the backfill window left
    items with NULL memory_type that were never flagged into the review queue.
    Every item this touches gets last_classified_at stamped (even when the
    classifier isn't confident), so it is attempted exactly once — the candidate
    pool strictly shrinks and batched callers terminate naturally (no spin on
    un-typable junk). Confident result -> set memory_type. Returns counts.
    """
    from app.llm.classifier import classify
    from app.models import KnowledgeItem

    rows = session.execute(text(
        """
        SELECT id FROM knowledge_items
        WHERE (memory_type IS NULL OR memory_type = '')
          AND last_classified_at IS NULL
          AND (CAST(:ns AS text) IS NULL OR namespace = :ns)
        ORDER BY created_at
        LIMIT :lim
        """
    ), {"ns": namespace, "lim": limit}).mappings().all()

    typed = 0
    left_untyped = 0
    now = datetime.now(timezone.utc)
    for r in rows:
        ki = session.get(KnowledgeItem, r["id"])
        if ki is None:
            continue
        out = classify(ki.content, tags=list(ki.tags or []), hint=None)
        conf = float(out.get("confidence", 0.0) or 0.0)
        new_type = out.get("memory_type")
        if not dry_run:
            ki.last_classified_at = now  # mark attempted so it never re-enters the pool
        if conf >= min_confidence and new_type:
            typed += 1
            if not dry_run:
                ki.memory_type = new_type
        else:
            left_untyped += 1
    if not dry_run:
        session.commit()
    return {
        "scanned": len(rows),
        "min_confidence": min_confidence,
        "typed": typed,
        "left_untyped": left_untyped,
        "dry_run": dry_run,
    }


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
