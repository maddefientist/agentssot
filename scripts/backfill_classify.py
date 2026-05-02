#!/usr/bin/env python3
"""Backfill classifier-driven memory_type, abstract, summary, cwd_hints,
device_hints, entity_refs on all KnowledgeItems.

Idempotent + resumable: tracks last_classified_at; only processes rows
where last_classified_at IS NULL OR < schema_version.

Rate-limited via --rps (requests per second to Ollama). Default 5 rps
with 4 parallel workers.

Pre-run: takes a pg_dump if --snapshot is set (default true).

Usage:
    python -m scripts.backfill_classify
    python -m scripts.backfill_classify --batch 200 --rps 5 --no-snapshot
    python -m scripts.backfill_classify --namespace claude-shared --resume
    python -m scripts.backfill_classify --dry-run

Plan: docs/plans/2026-04-24-hive-tiered-memory-plan-1-foundation.md T3.1–T3.3
"""
import argparse
import asyncio
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from app.db import SessionLocal
from app.models import KnowledgeItem, MemoryType, ReviewQueueItem, ReviewQueueKind, ReviewQueueStatus
from app.llm.classifier import classify
from app.llm.layer_compute import compute_layers
from app.settings import get_settings
from sqlalchemy import select


SCHEMA_VERSION_CUTOFF = datetime.fromisoformat("2026-04-24T00:00:00+00:00")


def take_snapshot(out_path: Path) -> None:
    """pg_dump the DB before backfill begins. Retained 30 days."""
    print(f"[backfill] taking pg_dump snapshot → {out_path}")
    subprocess.run(
        ["docker", "compose", "exec", "-T", "api",
         "pg_dump", "-U", "ssot", "-d", "ssot", "-Fc", "-f", "/tmp/backfill_snapshot.dump"],
        check=True,
    )
    subprocess.run(
        ["docker", "compose", "cp", "api:/tmp/backfill_snapshot.dump", str(out_path)],
        check=True,
    )
    print(f"[backfill] snapshot done: {out_path.stat().st_size // 1024 // 1024}MB")


def fetch_batch(session, namespace: str | None, batch_size: int) -> list[KnowledgeItem]:
    stmt = select(KnowledgeItem).where(
        (KnowledgeItem.last_classified_at.is_(None)) |
        (KnowledgeItem.last_classified_at < SCHEMA_VERSION_CUTOFF)
    )
    if namespace:
        stmt = stmt.where(KnowledgeItem.namespace == namespace)
    stmt = stmt.order_by(KnowledgeItem.created_at).limit(batch_size)
    return list(session.execute(stmt).scalars())


def classify_one(item: KnowledgeItem) -> dict:
    """Classify a single item; bring layer fields with it."""
    hint = None
    if item.memory_type:
        hint = item.memory_type.value if hasattr(item.memory_type, "value") else str(item.memory_type)
    return classify(item.content, tags=list(item.tags or []), hint=hint)


def apply_classification(session, item: KnowledgeItem, c: dict, settings) -> str:
    """Persist classifier output. Returns 'updated' / 'kept_low_conf' / 'unchanged'."""
    now = datetime.now(timezone.utc)
    layers = compute_layers(item.content, c)
    item.abstract = layers["abstract"]
    item.summary = layers["summary"]
    item.last_classified_at = now

    conf = float(c.get("confidence", 0.0))
    if conf >= settings.classifier_min_confidence:
        new_type = c.get("memory_type")
        existing = item.memory_type.value if hasattr(item.memory_type, "value") and item.memory_type else (
            str(item.memory_type) if item.memory_type else None
        )
        if existing in (None, "fact"):
            try:
                item.memory_type = MemoryType(new_type)
            except (ValueError, TypeError):
                pass
        elif new_type in ("command", "rule", "entity") and existing in ("preference", "reference"):
            try:
                item.memory_type = MemoryType(new_type)
            except (ValueError, TypeError):
                pass
        item.cwd_hints = list(c.get("cwd_hints") or [])[:50]
        item.device_hints = list(c.get("device_hints") or [])[:50]
        item.entity_refs = list(c.get("entity_mentions") or [])[:50]
        item.confidence = conf
        return "updated"

    rq = ReviewQueueItem(
        namespace=item.namespace,
        kind=ReviewQueueKind.low_conf,
        priority=10,
        primary_id=item.id,
        reason=f"backfill_low_conf={conf:.2f}",
        status=ReviewQueueStatus.pending,
    )
    session.add(rq)
    return "kept_low_conf"


async def worker(name: str, q: asyncio.Queue, results: Counter, lock: asyncio.Lock,
                 rps: float, settings) -> None:
    """Pulls items off queue, calls classifier, persists."""
    delay = 1.0 / rps if rps > 0 else 0.0
    while True:
        item_id = await q.get()
        if item_id is None:
            q.task_done()
            return
        session = None
        try:
            session = SessionLocal()
            item = session.get(KnowledgeItem, item_id)
            if item is None:
                async with lock:
                    results["missing"] += 1
                continue
            c = await asyncio.to_thread(classify_one, item)
            outcome = apply_classification(session, item, c, settings)
            session.commit()
            async with lock:
                results[outcome] += 1
                if c.get("memory_type"):
                    results[f"type:{c['memory_type']}"] += 1
        except Exception as e:
            async with lock:
                results["error"] += 1
            print(f"[{name}] error on {item_id}: {e}", file=sys.stderr)
        finally:
            if session is not None:
                session.close()
            q.task_done()
            if delay:
                await asyncio.sleep(delay)


async def run(args) -> Counter:
    settings = get_settings()
    results: Counter = Counter()
    q: asyncio.Queue = asyncio.Queue(maxsize=args.batch * 2)
    lock = asyncio.Lock()

    workers = [
        asyncio.create_task(worker(f"w{i}", q, results, lock, args.rps, settings))
        for i in range(args.workers)
    ]

    total_processed = 0
    while True:
        with SessionLocal() as session:
            batch = fetch_batch(session, args.namespace, args.batch)
        if not batch:
            break
        for item in batch:
            await q.put(item.id)
        total_processed += len(batch)
        await q.join()
        print(f"[backfill] processed {total_processed}, results so far: {dict(results)}", flush=True)
        if args.dry_run:
            print("[backfill] --dry-run set; stopping after first batch.")
            break

    for _ in workers:
        await q.put(None)
    await asyncio.gather(*workers)

    return results


def sweep_entities(session, default_namespace: str = "claude-shared") -> int:
    """Resolve entity_mentions strings to Entity ids; insert missing entities.

    After classify_one stores raw mention strings in entity_refs (e.g.
    ['unraid','hari']), this sweep replaces them with canonical Entity UUIDs,
    inserting Entity rows for unknown slugs.
    """
    from app.models import Entity, EntityType

    ents = list(session.execute(select(Entity)).scalars())
    by_slug = {e.slug: str(e.id) for e in ents}

    items_needing = list(session.execute(
        select(KnowledgeItem).where(KnowledgeItem.entity_refs != [])
    ).scalars())

    promoted = 0
    for item in items_needing:
        new_refs: list[str] = []
        for raw in item.entity_refs or []:
            if len(str(raw)) == 36 and "-" in str(raw):
                new_refs.append(str(raw))
                continue
            slug = str(raw).lower().strip()
            if slug in by_slug:
                new_refs.append(by_slug[slug])
            else:
                ent = Entity(namespace=default_namespace, slug=slug, type=EntityType.other, name=slug)
                session.add(ent)
                session.flush()
                by_slug[slug] = str(ent.id)
                new_refs.append(str(ent.id))
                promoted += 1
        if new_refs != list(item.entity_refs or []):
            item.entity_refs = new_refs
    session.commit()
    return promoted


def sweep_supersession(session, namespace: str | None) -> int:
    """Pairwise scan within (namespace, memory_type, entity_ref) groups.

    Newer item supersedes older when entity_refs overlap and memory_type
    matches. Excludes already-superseded items.
    """
    from app.services.lifecycle import find_supersession_candidates, apply_supersession

    stmt = select(KnowledgeItem).where(
        KnowledgeItem.superseded_by.is_(None),
        KnowledgeItem.memory_type.in_([
            MemoryType.command, MemoryType.rule, MemoryType.entity,
            MemoryType.decision,
        ]),
    )
    if namespace:
        stmt = stmt.where(KnowledgeItem.namespace == namespace)
    items = list(session.execute(stmt).scalars())
    items.sort(key=lambda x: x.created_at, reverse=True)

    seen_ids: set = set()
    superseded_count = 0
    for new in items:
        if new.id in seen_ids:
            continue
        candidates = find_supersession_candidates(new, items)
        for old in candidates:
            apply_supersession(old, new)
            seen_ids.add(old.id)
            session.add(ReviewQueueItem(
                namespace=new.namespace,
                kind=ReviewQueueKind.supersede,
                priority=5,
                primary_id=new.id,
                secondary_id=old.id,
                reason="backfill supersession sweep",
                status=ReviewQueueStatus.pending,
            ))
            superseded_count += 1
    session.commit()
    return superseded_count


def sweep_contradictions(session, namespace: str | None) -> int:
    """For every active command/skill, scan rules in same namespace for
    negation against the same entities. Enqueue HIGH-priority review entries.
    """
    from app.services.contradiction import detect_contradictions

    stmt = select(KnowledgeItem).where(
        KnowledgeItem.superseded_by.is_(None),
        KnowledgeItem.memory_type.in_([MemoryType.command, MemoryType.skill]),
    )
    if namespace:
        stmt = stmt.where(KnowledgeItem.namespace == namespace)
    items = list(session.execute(stmt).scalars())

    rules_by_ns: dict[str, list] = {}
    for item in items:
        if item.namespace not in rules_by_ns:
            rules_by_ns[item.namespace] = list(session.execute(
                select(KnowledgeItem).where(
                    KnowledgeItem.namespace == item.namespace,
                    KnowledgeItem.memory_type == MemoryType.rule,
                    KnowledgeItem.superseded_by.is_(None),
                )
            ).scalars())

    contradictions_found = 0
    for item in items:
        contras = detect_contradictions(
            new_type=item.memory_type.value if hasattr(item.memory_type, "value") else str(item.memory_type),
            new_entity_refs=list(item.entity_refs or []),
            existing_rules=rules_by_ns[item.namespace],
        )
        for rule in contras:
            session.add(ReviewQueueItem(
                namespace=item.namespace,
                kind=ReviewQueueKind.contradiction,
                priority=20,
                primary_id=item.id,
                secondary_id=rule.id,
                reason="backfill contradiction sweep",
                status=ReviewQueueStatus.pending,
            ))
            contradictions_found += 1
    session.commit()
    return contradictions_found


def distribution_report(session, namespace: str | None) -> dict:
    """Generate the gate report shown to the operator before Phase 4 ships."""
    from sqlalchemy import func

    where = []
    if namespace:
        where.append(KnowledgeItem.namespace == namespace)

    base = select(func.count(KnowledgeItem.id))
    if where:
        base = base.where(*where)
    total = session.execute(base).scalar_one()

    types: dict[str, int] = {}
    type_stmt = select(KnowledgeItem.memory_type, func.count(KnowledgeItem.id)) \
        .group_by(KnowledgeItem.memory_type)
    if where:
        type_stmt = type_stmt.where(*where)
    for t, n in session.execute(type_stmt):
        key = t.value if hasattr(t, "value") and t else (str(t) if t else "null")
        types[key] = n

    rq_stmt = select(ReviewQueueItem.kind, func.count(ReviewQueueItem.id)) \
        .where(ReviewQueueItem.status == ReviewQueueStatus.pending) \
        .group_by(ReviewQueueItem.kind)
    if namespace:
        rq_stmt = rq_stmt.where(ReviewQueueItem.namespace == namespace)
    rq_counts = {(k.value if hasattr(k, "value") else str(k)): n for k, n in session.execute(rq_stmt)}

    superseded = session.execute(
        select(func.count(KnowledgeItem.id))
        .where(KnowledgeItem.superseded_by.isnot(None))
    ).scalar_one()

    high_conf = session.execute(
        select(func.count(KnowledgeItem.id))
        .where(KnowledgeItem.last_classified_at.isnot(None),
               KnowledgeItem.confidence >= 0.6)
    ).scalar_one()

    low_conf = session.execute(
        select(func.count(KnowledgeItem.id))
        .where(KnowledgeItem.last_classified_at.isnot(None),
               KnowledgeItem.confidence < 0.6)
    ).scalar_one()

    return {
        "total_items": total,
        "classified_high_conf": high_conf,
        "classified_low_conf": low_conf,
        "type_distribution": types,
        "superseded_count": superseded,
        "review_queue_counts": rq_counts,
    }


def print_report(report: dict) -> None:
    print()
    print("=" * 60)
    print("BACKFILL DISTRIBUTION REPORT")
    print("=" * 60)
    print(f"  total items:                {report['total_items']}")
    if report['total_items']:
        print(f"  classified high-confidence: {report['classified_high_conf']} ({report['classified_high_conf']/report['total_items']:.1%})")
        print(f"  classified low-confidence:  {report['classified_low_conf']} ({report['classified_low_conf']/report['total_items']:.1%})")
    else:
        print("  (no items)")
    print()
    print("  type distribution:")
    for t, n in sorted(report["type_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {t:20s}  {n}")
    print()
    print(f"  supersession marked: {report['superseded_count']}")
    print(f"  review queue (pending):")
    for k, n in report["review_queue_counts"].items():
        print(f"    {k:20s}  {n}")
    print("=" * 60)
    print()
    print("OPERATOR GATE: review the type distribution above.")
    print("If it looks wrong (e.g. >80% episodic, <5 commands total), tune")
    print("the SYSTEM_PROMPT in api/app/llm/classifier.py and re-run with")
    print("--resume. Otherwise: proceed to Plan 2 Phase 4 (loadout hook).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default=None,
                        help="Only process this namespace (default: all)")
    parser.add_argument("--batch", type=int, default=200, help="Rows per batch")
    parser.add_argument("--rps", type=float, default=5.0, help="Classifier requests/sec per worker")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--snapshot", action="store_true", default=True,
                        help="Take pg_dump before backfill")
    parser.add_argument("--no-snapshot", action="store_false", dest="snapshot")
    parser.add_argument("--resume", action="store_true",
                        help="Skip snapshot if resuming partial run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process one batch then stop")
    args = parser.parse_args()

    if args.snapshot and not args.resume:
        snap_path = Path(f"./backups/backfill-{datetime.now().strftime('%Y%m%dT%H%M%S')}.dump")
        snap_path.parent.mkdir(exist_ok=True)
        take_snapshot(snap_path)

    t0 = time.time()
    results = asyncio.run(run(args))
    elapsed = time.time() - t0

    print("\n[backfill] post-classification sweeps")
    with SessionLocal() as session:
        promoted = sweep_entities(session, args.namespace or "claude-shared")
        print(f"  entities promoted: {promoted}")
        sup = sweep_supersession(session, args.namespace)
        print(f"  supersession candidates: {sup}")
        contras = sweep_contradictions(session, args.namespace)
        print(f"  contradictions flagged: {contras}")

    import json as _json
    with SessionLocal() as session:
        report = distribution_report(session, args.namespace)
    report_path = Path(f"./backups/backfill-report-{datetime.now().strftime('%Y%m%dT%H%M%S')}.json")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(_json.dumps(report, indent=2))
    print_report(report)
    print(f"  report: {report_path}")

    print()
    print("=" * 50)
    print("Backfill complete")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  results: {dict(results)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
