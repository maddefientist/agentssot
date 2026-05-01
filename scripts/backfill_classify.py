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

    print()
    print("=" * 50)
    print("Backfill complete")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  results: {dict(results)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
