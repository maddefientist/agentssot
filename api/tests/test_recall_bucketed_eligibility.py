"""Offline regressions for verified S0 defects in the bucketed recall path
(POST /api/v1/knowledge/recall, bucketed=true -> routers.knowledge._recall_bucketed).

2026-09-18 audit findings pinned here:
  1. The bucketed path built its candidate-eligibility filter without the
     status clause the legacy /recall path (crud._recall_knowledge_weighted)
     already applies, so an item flagged via 'wrong' feedback
     (crud.create_knowledge_feedback sets status='flagged') stayed eligible
     and kept surfacing in bucketed results.
  2. Items that were never classified/tiered (abstract IS NULL) rendered as
     a blank line instead of falling back to bounded source text.
  3. The request had no session_id/agent_key field, so a bucketed recall
     left no retrieval receipt for later correction/audit attribution.

These tests call the real `_recall_bucketed` function (not a reimplementation)
against a fake session whose row-filtering is driven by the *actual* compiled
SQL text the function builds. If a future change drops the status clause from
`base_filters`, the fake stops filtering and `test_flagged_item_excluded_*`
fails -- this is a real regression guard on production code, not a mock of it.

No live database, network, or model calls. Run:
    DATABASE_URL=postgresql+psycopg://test:test@127.0.0.1/test \
        python -m pytest api/tests/test_recall_bucketed_eligibility.py -q
"""
import asyncio
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@127.0.0.1/test")

from app.routers import knowledge as knowledge_router
from app.routers.knowledge import _recall_bucketed
from app.schemas import BucketedRecallRequest
from app.security import AuthContext


def _item(memory_type="rule", status="active", abstract=None, summary=None,
          content="fallback content text ", confidence=1.0):
    return SimpleNamespace(
        id=uuid4(),
        memory_type=memory_type,
        status=status,
        abstract=abstract,
        summary=summary,
        content=content,
        source_ref=None,
        confidence=confidence,
        entity_refs=[],
        tags=[],
    )


class _FakeSession:
    """Mimics Postgres row-level filtering by inspecting the compiled SQL
    the function under test actually builds, instead of reimplementing
    eligibility logic in the test. Only the status predicate is modeled;
    everything else in `dataset` is returned as-is."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.update_calls = 0
        self.committed = 0

    def execute(self, stmt):
        compiled = str(stmt)
        if compiled.strip().lower().startswith("select"):
            # `select(KnowledgeItem, ...)` always lists every column
            # (including "knowledge_items.status") in the SELECT clause, so a
            # status check only means something in the WHERE clause -- check
            # there specifically, not anywhere in the compiled statement.
            where_clause = compiled.split("WHERE", 1)[1] if "WHERE" in compiled else ""
            rows = list(self.dataset)
            if "status" in where_clause:
                rows = [(it, d) for it, d in rows if it.status == "active" or it.status is None]
            return rows
        self.update_calls += 1
        return None

    def commit(self):
        self.committed += 1


class _FakeEmbedder:
    is_available = True

    def embed_text(self, _text):
        return [0.1] * 8


def _auth():
    return AuthContext(key_id="k1", key_name="tester", role="writer", namespaces=["default"])


def _request():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(embedding_provider=_FakeEmbedder())))


@pytest.fixture(autouse=True)
def _no_wal_writes(monkeypatch):
    monkeypatch.setattr(knowledge_router.wal, "log_event", lambda *a, **k: None)


def _run(data, session):
    return asyncio.run(_recall_bucketed(data, _request(), session, _auth()))


def test_flagged_item_excluded_from_bucketed_recall():
    active = _item(status="active", abstract="an eligible rule")
    flagged = _item(status="flagged", abstract="a disputed rule")
    session = _FakeSession([(active, 0.1), (flagged, 0.1)])
    data = BucketedRecallRequest(query="", tiers=["rule"], top_per_tier={"rule": 5})

    resp = _run(data, session)

    ids = {str(it.id) for it in resp.buckets["rule"]}
    assert str(active.id) in ids
    assert str(flagged.id) not in ids


def test_untyped_legacy_status_none_remains_eligible():
    """status IS NULL is legacy data, not a dispute -- must stay eligible."""
    legacy = _item(status=None, abstract="legacy row, never given a status")
    session = _FakeSession([(legacy, 0.1)])
    data = BucketedRecallRequest(query="", tiers=["rule"], top_per_tier={"rule": 5})

    resp = _run(data, session)

    ids = {str(it.id) for it in resp.buckets["rule"]}
    assert str(legacy.id) in ids


def test_blank_abstract_falls_back_to_bounded_source_text():
    never_classified = _item(abstract=None, summary=None, content="x" * 500)
    session = _FakeSession([(never_classified, 0.1)])
    data = BucketedRecallRequest(query="", tiers=["rule"], top_per_tier={"rule": 5})

    resp = _run(data, session)

    (it,) = resp.buckets["rule"]
    assert it.abstract
    assert len(it.abstract) < len(never_classified.content)


def test_recall_with_session_id_writes_attribution_receipt(monkeypatch):
    received = {}
    monkeypatch.setattr(
        knowledge_router.wal, "log_event",
        lambda op, **kw: received.update({"op": op, **kw}),
    )
    item = _item(abstract="rule text")
    session = _FakeSession([(item, 0.1)])
    data = BucketedRecallRequest(
        query="synthetic private query MUST_NOT_BE_LOGGED", tiers=["rule"], top_per_tier={"rule": 5},
        session_id="session-abc", agent_key="device-x-writer",
    )

    _run(data, session)

    assert received["op"] == "knowledge.recall_bucketed"
    assert "query" not in received["payload"]
    assert received["payload"]["query_chars"] == len(data.query)
    assert data.query not in repr(received)
    assert received["payload"]["session_id"] == "session-abc"
    assert received["payload"]["agent_key"] == "device-x-writer"
    assert str(item.id) in received["result"]["item_ids"]


def test_recall_without_attribution_fields_writes_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(knowledge_router.wal, "log_event", lambda *a, **k: calls.append((a, k)))
    item = _item(abstract="rule text")
    session = _FakeSession([(item, 0.1)])
    data = BucketedRecallRequest(query="", tiers=["rule"], top_per_tier={"rule": 5})

    _run(data, session)

    assert calls == []


def test_bucketed_request_backward_compatible_without_attribution_fields():
    """Existing callers (test_recall_bucketed.py) that never set session_id/
    agent_key must still validate with the same defaults as before."""
    data = BucketedRecallRequest(query="ssh storage-node", tiers=["command", "rule"])
    assert data.session_id is None
    assert data.agent_key is None
    assert data.bucketed is True
