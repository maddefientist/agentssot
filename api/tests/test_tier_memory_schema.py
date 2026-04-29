"""Verify Plan 1 Phase 0 schema additions are live in the dev DB.

Runs against the live DB via docker compose exec. Requires docker compose
to be running (the agentssot-db container must be healthy).

NOTE: memory_type is stored as TEXT in this schema, not as a Postgres enum
type. The MemoryType Python enum in models.py is the source of truth for
valid values. The enum-extension test therefore checks the Python class.
"""
import subprocess

import pytest


def _psql(sql: str) -> str:
    """Run a SQL query in the agentssot-db container and return trimmed output."""
    out = subprocess.run(
        [
            "docker", "compose", "exec", "-T", "db",
            "psql", "-U", "ssot", "-d", "ssot", "-tAc", sql,
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd="/opt/agentssot",
    )
    return out.stdout.strip()


@pytest.mark.integration
def test_new_knowledge_columns_present():
    """All 9 lifecycle columns added in Phase 0 must exist on knowledge_items."""
    cols = _psql(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='knowledge_items' "
        "AND column_name IN ('expires_at','superseded_by','confidence',"
        "'entity_refs','rule_refs','cwd_hints','device_hints',"
        "'loadout_priority','last_classified_at') "
        "ORDER BY column_name;"
    ).split("\n")
    expected = sorted([
        "confidence", "cwd_hints", "device_hints", "entity_refs",
        "expires_at", "last_classified_at", "loadout_priority",
        "rule_refs", "superseded_by",
    ])
    assert sorted(cols) == expected, f"Missing columns: {set(expected) - set(cols)}"


@pytest.mark.integration
def test_memory_type_enum_extended():
    """MemoryType Python enum must include all 4 new tier values from Phase 0.

    memory_type is stored as TEXT in Postgres, so we verify the Python enum
    inside the API container (which has pgvector installed) rather than a
    Postgres enum_range() call or a host-side import (host lacks pgvector).
    """
    out = subprocess.run(
        [
            "docker", "compose", "exec", "-T", "api",
            "python", "-c",
            "from app.models import MemoryType; "
            "print(','.join(m.value for m in MemoryType))",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd="/opt/agentssot",
    )
    values = out.stdout.strip()
    for new_value in ("command", "rule", "entity", "episodic"):
        assert new_value in values, f"MemoryType enum missing: {new_value!r}. Got: {values}"


@pytest.mark.integration
def test_deletion_log_and_review_queue_present():
    """deletion_log and review_queue tables must exist in the live DB."""
    tables = _psql(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_name IN ('deletion_log','review_queue') ORDER BY table_name;"
    )
    assert "deletion_log" in tables, "deletion_log table not found"
    assert "review_queue" in tables, "review_queue table not found"
