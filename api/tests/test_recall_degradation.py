"""Recall degradation under GPU/synthesis contention.

Exercises app.routers.knowledge._recall_bucketed end-to-end against a fake
SQLAlchemy session (statement construction only, no real DB) to pin three
behaviors:

  1. A reranker exception on a tier falls back to vector-only results for
     that tier and surfaces a non-None degraded_reason (existing safety net,
     newly covered).
  2. While synthesis is active (app.synthesis.loop._synthesis_active),
     reranking is skipped entirely -- the reranker is never called -- and
     diagnostics report degraded_reason="synthesis_active: ...".
  3. The tier loop stays sequential (no asyncio.gather across tiers) --
     the shared SQLAlchemy Session must never be used from multiple
     concurrently-running coroutines/threads at once.

Also covers the slow-recall alert cooldown (app.routers.knowledge.
_maybe_alert_slow_recall) in isolation.
"""
import asyncio
import os
import uuid
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@127.0.0.1/test")

import pytest

from app.routers import knowledge as knowledge_router
from app.reranker.base import RerankerProviderError
from app.schemas import BucketedRecallRequest, RecallRequest
from app.security import AuthContext, ApiRole
from app.synthesis import loop as synthesis_loop_module


def test_recall_schema_defaults_to_fast_non_reranked_path():
    assert RecallRequest().rerank is False


def _fake_item(content="hello world"):
    return SimpleNamespace(
        id=uuid.uuid4(),
        memory_type="command",
        source_ref=None,
        abstract="an abstract",
        summary="a summary",
        content=content,
        confidence=1.0,
        entity_refs=[],
        tags=[],
    )


class _FakeSession:
    """session.execute(stmt) returns an iterable of (item, distance) tuples,
    unpacking the same way a real SQLAlchemy Result does -- the statement's
    actual contents are never inspected, so no DB/connection is required."""

    def __init__(self, rows):
        self._rows = rows
        self.execute_calls = 0

    def execute(self, _stmt):
        self.execute_calls += 1
        return list(self._rows)

    def commit(self):
        pass


class _FakeEmbeddingProvider:
    is_available = True

    def embed_text(self, _text):
        return [0.0] * 8


class _FakeApp:
    def __init__(self, embedding_provider):
        self.state = SimpleNamespace(embedding_provider=embedding_provider)


class _FakeRequest:
    def __init__(self, embedding_provider):
        self.app = _FakeApp(embedding_provider)


def _auth():
    return AuthContext(key_id="k1", key_name="tester", role=ApiRole.writer.value, namespaces=["default"])


def _request_data(tiers=("command",), top_per_tier=None):
    return BucketedRecallRequest(
        query="find the thing",
        namespace="default",
        tiers=list(tiers),
        top_per_tier=top_per_tier or {t: 5 for t in tiers},
    )


class _RaisingReranker:
    is_available = True
    model = "raising-model"
    called = False

    def rerank(self, query, texts):
        type(self).called = True
        raise RerankerProviderError("simulated Ollama timeout")


class _NeverCallReranker:
    is_available = True
    model = "should-not-be-called"

    def rerank(self, query, texts):
        raise AssertionError("reranker must not be called while synthesis is active")


@pytest.fixture(autouse=True)
def _reset_synthesis_flag():
    synthesis_loop_module._synthesis_active = False
    yield
    synthesis_loop_module._synthesis_active = False


def test_reranker_exception_falls_back_to_vector_only_with_degraded_reason(monkeypatch):
    rows = [(_fake_item(f"item-{i}"), 0.1 * i) for i in range(3)]
    session = _FakeSession(rows)
    request = _FakeRequest(_FakeEmbeddingProvider())

    _RaisingReranker.called = False
    monkeypatch.setattr(
        knowledge_router, "build_reranker_pair",
        lambda settings: (_RaisingReranker(), _RaisingReranker()),
    )
    monkeypatch.setattr(
        knowledge_router, "pick_reranker",
        lambda tiers, fast, deep: ("raising-model", _RaisingReranker()),
    )

    result = asyncio.run(
        knowledge_router._recall_bucketed(_request_data(), request, session, _auth())
    )

    assert _RaisingReranker.called is True
    assert len(result.buckets["command"]) == 3  # vector-only fallback still returns items
    assert result.diagnostics.degraded_reason is not None
    assert "reranker_error" in result.diagnostics.degraded_reason


def test_synthesis_active_skips_reranking_entirely(monkeypatch):
    rows = [(_fake_item(f"item-{i}"), 0.1 * i) for i in range(3)]
    session = _FakeSession(rows)
    request = _FakeRequest(_FakeEmbeddingProvider())

    monkeypatch.setattr(
        knowledge_router, "build_reranker_pair",
        lambda settings: (_NeverCallReranker(), _NeverCallReranker()),
    )
    monkeypatch.setattr(
        knowledge_router, "pick_reranker",
        lambda tiers, fast, deep: ("should-not-be-called", _NeverCallReranker()),
    )
    synthesis_loop_module._synthesis_active = True

    result = asyncio.run(
        knowledge_router._recall_bucketed(_request_data(), request, session, _auth())
    )

    assert len(result.buckets["command"]) == 3
    assert result.diagnostics.degraded_reason is not None
    assert result.diagnostics.degraded_reason.startswith("synthesis_active")
    assert result.diagnostics.rerank_ms == 0  # rerank path never entered


def test_synthesis_inactive_reranking_runs_normally(monkeypatch):
    """Control case: with the flag clear and a healthy reranker, no
    degraded_reason is set."""
    rows = [(_fake_item(f"item-{i}"), 0.1 * i) for i in range(2)]
    session = _FakeSession(rows)
    request = _FakeRequest(_FakeEmbeddingProvider())

    class _HealthyReranker:
        is_available = True
        model = "healthy-model"

        def rerank(self, query, texts):
            return [1.0] * len(texts)

    monkeypatch.setattr(
        knowledge_router, "build_reranker_pair",
        lambda settings: (_HealthyReranker(), _HealthyReranker()),
    )
    monkeypatch.setattr(
        knowledge_router, "pick_reranker",
        lambda tiers, fast, deep: ("healthy-model", _HealthyReranker()),
    )

    result = asyncio.run(
        knowledge_router._recall_bucketed(_request_data(), request, session, _auth())
    )

    assert result.diagnostics.degraded_reason is None


def test_tier_loop_is_sequential_not_gathered():
    """Static guard: the tier loop over the shared SQLAlchemy Session must
    stay a plain `for` loop -- no asyncio.gather/create_task fan-out across
    tiers, which would let multiple coroutines touch one Session concurrently."""
    import inspect

    src = inspect.getsource(knowledge_router._recall_bucketed)
    body_after_loop_start = src.split("for tier in tiers:", 1)
    assert len(body_after_loop_start) == 2, "expected a single sequential 'for tier in tiers:' loop"
    tier_body = body_after_loop_start[1]
    # Stop scanning at the next top-level statement after the loop (the
    # output-sanitization block that follows it in source order).
    tier_body = tier_body.split("if getattr(get_settings(), \"recall_output_sanitization\"", 1)[0]
    assert "asyncio.gather(" not in tier_body
    assert "create_task(" not in tier_body


# ── slow-recall alert cooldown ─────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_alert_cooldown():
    knowledge_router._last_slow_recall_alert_at = 0.0
    yield
    knowledge_router._last_slow_recall_alert_at = 0.0


def test_slow_recall_below_threshold_does_not_alert(monkeypatch):
    calls = []
    monkeypatch.setattr(knowledge_router, "get_settings", lambda: SimpleNamespace(
        recall_slow_alert_threshold_ms=10000, recall_slow_alert_cooldown_seconds=900,
    ))
    with pytest.MonkeyPatch.context() as mp:
        import app.alerting as alerting_mod
        mp.setattr(alerting_mod, "send_alert", lambda *a, **k: calls.append((a, k)))
        knowledge_router._maybe_alert_slow_recall(vec_ms=100, rerank_ms=200, namespace="ns")
    assert calls == []


def test_slow_recall_above_threshold_alerts_once_then_cools_down(monkeypatch):
    calls = []
    monkeypatch.setattr(knowledge_router, "get_settings", lambda: SimpleNamespace(
        recall_slow_alert_threshold_ms=1000, recall_slow_alert_cooldown_seconds=900,
    ))
    import app.alerting as alerting_mod
    monkeypatch.setattr(alerting_mod, "send_alert", lambda *a, **k: calls.append((a, k)) or True)

    knowledge_router._maybe_alert_slow_recall(vec_ms=800, rerank_ms=800, namespace="ns")
    assert len(calls) == 1
    assert calls[0][0][0] == "recall.slow"

    # Immediately again: still inside the cooldown window -- must not alert again.
    knowledge_router._maybe_alert_slow_recall(vec_ms=900, rerank_ms=900, namespace="ns")
    assert len(calls) == 1


def test_slow_recall_alerts_again_after_cooldown_elapses(monkeypatch):
    calls = []
    monkeypatch.setattr(knowledge_router, "get_settings", lambda: SimpleNamespace(
        recall_slow_alert_threshold_ms=1000, recall_slow_alert_cooldown_seconds=0,
    ))
    import app.alerting as alerting_mod
    monkeypatch.setattr(alerting_mod, "send_alert", lambda *a, **k: calls.append((a, k)) or True)

    knowledge_router._maybe_alert_slow_recall(vec_ms=800, rerank_ms=800, namespace="ns")
    knowledge_router._maybe_alert_slow_recall(vec_ms=800, rerank_ms=800, namespace="ns")
    assert len(calls) == 2  # cooldown_seconds=0 -> never suppresses
