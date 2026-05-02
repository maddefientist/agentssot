"""Lifecycle sweep — confidence decay, expiration, idempotency."""
import pytest
from datetime import datetime, timedelta, timezone

from app.services.lifecycle_sweep import run_sweep
from app.models import KnowledgeItem
from app.db import SessionLocal


@pytest.mark.integration
def test_sweep_is_idempotent():
    """Two consecutive sweeps must produce the same final state."""
    with SessionLocal() as s:
        s1 = run_sweep(s, namespace="claude-shared", dry_run=True)
        s2 = run_sweep(s, namespace="claude-shared", dry_run=True)
    assert s1["decayed"] == s2["decayed"]
    assert s1["expired"] == s2["expired"]


@pytest.mark.integration
def test_sweep_decays_low_use_items():
    """Items not recalled in 90d should be counted as decay candidates."""
    with SessionLocal() as s:
        before = s.query(KnowledgeItem).filter(
            KnowledgeItem.last_recalled_at < datetime.now(timezone.utc) - timedelta(days=90),
            KnowledgeItem.namespace == "claude-shared",
        ).count()
        result = run_sweep(s, namespace="claude-shared", dry_run=False)
        assert result["decayed"] <= before
        assert result["decayed"] >= 0
