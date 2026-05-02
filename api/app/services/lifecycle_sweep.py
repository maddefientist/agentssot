"""Nightly lifecycle sweep — runs at 03:00 UTC.

Steps (idempotent):
1. Decay: items with last_recalled_at older than 90d lose 10% confidence.
2. Expire: episodic items older than 180d get expires_at = now if unset.
3. (Stub) Contradiction recheck.
4. (Stub) Supersession recheck.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models import KnowledgeItem, MemoryType


DECAY_AGE = timedelta(days=90)
DECAY_FACTOR = 0.9
EPISODIC_TTL = timedelta(days=180)


@dataclass
class SweepResult:
    decayed: int
    expired: int
    contradictions_flagged: int
    supersessions_applied: int

    def as_dict(self) -> dict:
        return self.__dict__


def run_sweep(session: Session, namespace: str = "claude-shared",
              dry_run: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    decay_cutoff = now - DECAY_AGE
    expire_cutoff = now - EPISODIC_TTL

    # --- Decay ---
    decay_q = select(KnowledgeItem).where(
        KnowledgeItem.namespace == namespace,
        KnowledgeItem.last_recalled_at < decay_cutoff,
        KnowledgeItem.confidence > 0.1,
    ).limit(5000)

    decayed = 0
    for item in session.execute(decay_q).scalars():
        new_conf = max(0.1, float(item.confidence or 1.0) * DECAY_FACTOR)
        if not dry_run:
            item.confidence = new_conf
        decayed += 1

    # --- Expire episodic ---
    expire_q = select(KnowledgeItem).where(
        KnowledgeItem.namespace == namespace,
        KnowledgeItem.memory_type == MemoryType.episodic,
        KnowledgeItem.created_at < expire_cutoff,
        KnowledgeItem.expires_at.is_(None),
    ).limit(5000)

    expired = 0
    for item in session.execute(expire_q).scalars():
        if not dry_run:
            item.expires_at = now
        expired += 1

    if not dry_run:
        session.commit()

    return SweepResult(
        decayed=decayed, expired=expired,
        contradictions_flagged=0, supersessions_applied=0,
    ).as_dict()
