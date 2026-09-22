"""Route-level secret-boundary repair tests.

Proves that tiered ingest and session-complete reject synthetic secrets
before provider calls, DB writes, or raw-content audit logging. Never uses
real credentials: fake tokens are assembled at runtime.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@127.0.0.1/test")

from app import main
from app.routers import knowledge as knowledge_router
from app.schemas import SessionCompleteRequest, TieredKnowledgeCreate
from app.security import AuthContext, ApiRole


def _fake_openai_key() -> str:
    return "sk-" + "abc123def456ghi789jkl012mno345pqr678stu901"


def _auth(namespaces=("default",), role=ApiRole.writer.value):
    return AuthContext(key_id="k1", key_name="tester", role=role, namespaces=list(namespaces))


class _CountingEmbedder:
    def __init__(self, available=True):
        self.is_available = available
        self.calls: list[str] = []

    def embed_text(self, text):
        self.calls.append(text)
        return [0.0] * 8


class _CountingLLM:
    def __init__(self, available=True, raw=""):
        self.is_available = available
        self.raw = raw
        self.summarize_calls: list[str] = []

    def summarize(self, prompt: str) -> str:
        self.summarize_calls.append(prompt)
        return self.raw


class _FakeSession:
    def __init__(self):
        self.added: list = []
        self.committed = 0
        self.flushed = 0
        self.executed = 0
        self.rolled_back = 0

    def add(self, obj):
        self.added.append(obj)
        if getattr(obj, "id", None) is None:
            obj.id = uuid4()

    def flush(self):
        self.flushed += 1

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = uuid4()
        if getattr(obj, "created_at", None) is None:
            obj.created_at = datetime.now(timezone.utc)
        if getattr(obj, "verbatim", None) is None:
            obj.verbatim = False

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1

    def execute(self, _stmt):
        self.executed += 1

        class _Empty:
            def first(self):
                return None

            def scalars(self):
                return self

            def scalar_one_or_none(self):
                return None

            def all(self):
                return []

            def __iter__(self):
                return iter([])

        return _Empty()

    def begin_nested(self):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            yield self

        return _cm()


def _settings(*, scanning=True):
    return SimpleNamespace(
        ingest_secret_scanning=scanning,
        classifier_min_confidence=0.6,
        semantic_dedup_threshold=0.0,
        supersession_similarity_threshold=0.0,
    )


def _request(embedder, llm=None):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                embedding_provider=embedder,
                llm_provider=llm or _CountingLLM(available=False),
            )
        )
    )


@pytest.fixture(autouse=True)
def _isolate_settings_and_wal(monkeypatch):
    wal_calls: list[tuple] = []

    def _capture(op, **kwargs):
        wal_calls.append((op, kwargs))

    monkeypatch.setattr(knowledge_router.wal, "log_event", _capture)
    monkeypatch.setattr(main.wal, "log_event", _capture)
    monkeypatch.setattr(knowledge_router, "classify", lambda *a, **k: {"confidence": 1.0, "memory_type": "fact"})
    return wal_calls


def _run_ingest(data, request, session, auth):
    return asyncio.run(knowledge_router.ingest_tiered(data, request, session, auth))


# ── tiered ingest ──────────────────────────────────────────────────


def test_tiered_ingest_rejects_secret_before_provider_or_db(monkeypatch):
    secret = _fake_openai_key()
    embedder = _CountingEmbedder()
    session = _FakeSession()
    monkeypatch.setattr(knowledge_router, "get_settings", lambda: _settings(scanning=True))

    data = TieredKnowledgeCreate(
        content=f"Use key {secret}",
        namespace="default",
        abstract="short",
        summary="longer summary text",
        memory_type="fact",
    )
    with pytest.raises(HTTPException) as exc:
        _run_ingest(data, _request(embedder), session, _auth())

    assert exc.value.status_code == 422
    detail = str(exc.value.detail)
    assert "potential secrets" in detail
    assert "openai_api_key" in detail
    assert secret not in detail
    assert embedder.calls == []
    assert session.added == []
    assert session.committed == 0
    assert session.executed == 0


def test_tiered_ingest_error_body_and_wal_omit_secret_substring(monkeypatch):
    secret = _fake_openai_key()
    wal_calls: list = []
    monkeypatch.setattr(knowledge_router.wal, "log_event", lambda *a, **k: wal_calls.append((a, k)))
    monkeypatch.setattr(knowledge_router, "get_settings", lambda: _settings(scanning=True))
    embedder = _CountingEmbedder()
    session = _FakeSession()

    data = TieredKnowledgeCreate(content=f"Key: {secret}", namespace="default")
    with pytest.raises(HTTPException) as exc:
        _run_ingest(data, _request(embedder), session, _auth())

    assert secret not in str(exc.value.detail)
    assert wal_calls == []
    assert embedder.calls == []


def test_tiered_ingest_benign_content_persists_without_raw_wal(monkeypatch):
    embedder = _CountingEmbedder(available=False)
    session = _FakeSession()
    wal_calls: list = []
    monkeypatch.setattr(knowledge_router.wal, "log_event", lambda op, **kw: wal_calls.append((op, kw)))
    monkeypatch.setattr(knowledge_router, "get_settings", lambda: _settings(scanning=True))

    content = "We decided to use PostgreSQL with pgvector for embeddings"
    data = TieredKnowledgeCreate(
        content=content,
        namespace="default",
        abstract="Use pgvector",
        summary="PostgreSQL plus pgvector stores embeddings",
        memory_type="fact",
        source="operator",
        source_ref="note-1",
        tags=["architecture"],
    )
    result = _run_ingest(data, _request(embedder), session, _auth())

    assert result is not None
    assert session.committed == 1
    assert len(session.added) == 1
    assert session.added[0].content == content
    assert session.added[0].source == "operator"
    assert session.added[0].source_ref == "note-1"
    assert session.added[0].tags == ["architecture"]
    assert wal_calls, "successful ingest must write an audit receipt"
    op, payload = wal_calls[0]
    assert op == "knowledge.ingest"
    dumped = repr(payload)
    assert content not in dumped
    assert "note-1" not in dumped
    assert "operator" not in dumped
    assert "architecture" not in dumped
    assert "content_chars" in payload["payload"]
    assert payload["payload"]["has_source"] is True
    assert payload["payload"]["has_source_ref"] is True
    assert payload["payload"]["tag_count"] == 1
    assert "source_ref" not in payload["payload"]
    assert "tags" not in payload["payload"]
    assert payload["result"]["id"]


def test_tiered_ingest_unauthorized_namespace_denied_before_scan(monkeypatch):
    called = {"scan": 0}
    monkeypatch.setattr(
        knowledge_router,
        "collect_field_rejections",
        lambda *a, **k: called.__setitem__("scan", called["scan"] + 1) or [],
    )
    embedder = _CountingEmbedder()
    session = _FakeSession()
    data = TieredKnowledgeCreate(content="benign fact about caching", namespace="other-ns")

    with pytest.raises(HTTPException) as exc:
        _run_ingest(data, _request(embedder), session, _auth(namespaces=["default"]))

    assert exc.value.status_code == 403
    assert called["scan"] == 0
    assert embedder.calls == []
    assert session.committed == 0


def test_tiered_ingest_scanner_exception_does_not_fall_through(monkeypatch):
    def _boom(_fields):
        raise RuntimeError("scanner backend failed")

    monkeypatch.setattr(knowledge_router, "collect_field_rejections", _boom)
    monkeypatch.setattr(knowledge_router, "get_settings", lambda: _settings(scanning=True))
    embedder = _CountingEmbedder()
    session = _FakeSession()
    data = TieredKnowledgeCreate(content="benign architecture note", namespace="default")

    with pytest.raises(RuntimeError, match="scanner backend failed"):
        _run_ingest(data, _request(embedder), session, _auth())

    assert embedder.calls == []
    assert session.added == []
    assert session.committed == 0


def test_tiered_ingest_derived_model_output_scanned_before_persist(monkeypatch):
    secret = _fake_openai_key()
    embedder = _CountingEmbedder(available=False)
    session = _FakeSession()
    monkeypatch.setattr(knowledge_router, "get_settings", lambda: _settings(scanning=True))

    async def _poison(_content, _llm):
        return (f"keep {secret}", None)

    monkeypatch.setattr(knowledge_router, "generate_tiered_summaries", _poison)

    data = TieredKnowledgeCreate(
        content="We chose local embeddings for recall quality",
        namespace="default",
        memory_type="fact",
        generate_summaries=True,
    )
    with pytest.raises(HTTPException) as exc:
        _run_ingest(data, _request(embedder), session, _auth())

    assert exc.value.status_code == 422
    assert secret not in str(exc.value.detail)
    assert session.added == []
    assert session.committed == 0


def test_tiered_ingest_scanning_disabled_is_the_only_bypass(monkeypatch):
    """Bypass policy: only INGEST_SECRET_SCANNING=false skips the gate."""
    secret = _fake_openai_key()
    embedder = _CountingEmbedder(available=False)
    session = _FakeSession()
    monkeypatch.setattr(knowledge_router, "get_settings", lambda: _settings(scanning=False))

    data = TieredKnowledgeCreate(
        content=f"Use key {secret}",
        namespace="default",
        abstract="key note",
        summary="operator supplied a token in content",
        memory_type="fact",
    )
    result = _run_ingest(data, _request(embedder), session, _auth())
    assert result is not None
    assert session.committed == 1


@pytest.mark.parametrize(
    "field",
    ["tags", "source", "source_ref", "memory_type", "cwd_hints", "entity_refs"],
)
def test_tiered_ingest_rejects_secret_in_metadata_before_classify_or_persist(
    monkeypatch, caplog, field,
):
    secret = _fake_openai_key()
    classify_calls: list = []
    monkeypatch.setattr(
        knowledge_router,
        "classify",
        lambda *a, **k: classify_calls.append((a, k)) or {"confidence": 1.0, "memory_type": "fact"},
    )
    wal_calls: list = []
    monkeypatch.setattr(knowledge_router.wal, "log_event", lambda *a, **k: wal_calls.append((a, k)))
    monkeypatch.setattr(knowledge_router, "get_settings", lambda: _settings(scanning=True))
    embedder = _CountingEmbedder()
    session = _FakeSession()
    extras = {
        "tags": {"tags": [f"topic-{secret}"]},
        "source": {"source": secret},
        "source_ref": {"source_ref": secret},
        "memory_type": {"memory_type": secret},
        "cwd_hints": {"cwd_hints": [f"/tmp/{secret}"]},
        "entity_refs": {"entity_refs": [secret]},
    }[field]
    data = TieredKnowledgeCreate(
        content="We decided to use PostgreSQL with pgvector for embeddings",
        namespace="default",
        **extras,
    )

    with caplog.at_level("WARNING"):
        with pytest.raises(HTTPException) as exc:
            _run_ingest(data, _request(embedder), session, _auth())

    assert exc.value.status_code == 422
    detail = str(exc.value.detail)
    assert "potential secrets" in detail
    assert "openai_api_key" in detail
    assert field in detail
    assert secret not in detail
    assert secret not in caplog.text
    assert classify_calls == []
    assert embedder.calls == []
    assert session.added == []
    assert session.committed == 0
    assert wal_calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", "operator"),
        ("source_ref", "note-1"),
        ("tags", ["architecture"]),
        ("memory_type", "fact"),
        ("cwd_hints", ["/srv/app"]),
        ("entity_refs", ["11111111-1111-1111-1111-111111111111"]),
    ],
)
def test_tiered_ingest_benign_metadata_persists_and_wal_omits_raw_strings(
    monkeypatch, field, value,
):
    embedder = _CountingEmbedder(available=False)
    session = _FakeSession()
    wal_calls: list = []
    monkeypatch.setattr(knowledge_router.wal, "log_event", lambda op, **kw: wal_calls.append((op, kw)))
    monkeypatch.setattr(knowledge_router, "get_settings", lambda: _settings(scanning=True))
    kwargs = {
        "content": "We decided to use PostgreSQL with pgvector for embeddings",
        "namespace": "default",
        "abstract": "Use pgvector",
        "summary": "PostgreSQL plus pgvector stores embeddings",
        "memory_type": "fact",
        field: value,
    }
    data = TieredKnowledgeCreate(**kwargs)
    result = _run_ingest(data, _request(embedder), session, _auth())

    assert result is not None
    assert session.committed == 1
    stored = session.added[0]
    if field != "entity_refs":
        assert getattr(stored, field) == value
    dumped = repr(wal_calls)
    if isinstance(value, list):
        for item in value:
            assert item not in dumped
    else:
        assert value not in dumped
    payload = wal_calls[0][1]["payload"]
    assert field not in payload
    if field == "source":
        assert payload["has_source"] is True
    elif field == "source_ref":
        assert payload["has_source_ref"] is True
    elif field == "tags":
        assert payload["tag_count"] == 1
    elif field == "memory_type":
        assert payload["has_memory_type"] is True
    elif field == "cwd_hints":
        assert payload["cwd_hint_count"] == 1
    elif field == "entity_refs":
        assert payload["entity_ref_count"] == 1


def test_tiered_ingest_derived_classifier_metadata_scanned_before_persist(monkeypatch):
    secret = _fake_openai_key()
    embedder = _CountingEmbedder(available=False)
    session = _FakeSession()
    wal_calls: list = []
    monkeypatch.setattr(knowledge_router.wal, "log_event", lambda *a, **k: wal_calls.append((a, k)))
    monkeypatch.setattr(knowledge_router, "get_settings", lambda: _settings(scanning=True))
    monkeypatch.setattr(
        knowledge_router,
        "classify",
        lambda *a, **k: {
            "confidence": 1.0,
            "memory_type": "fact",
            "abstract": "Local embeddings stay on-box",
            "summary": "Chose local embeddings for recall quality",
            "cwd_hints": [f"/opt/{secret}"],
            "device_hints": [secret],
            "entity_mentions": [secret],
        },
    )
    data = TieredKnowledgeCreate(
        content="We chose local embeddings for recall quality",
        namespace="default",
    )
    with pytest.raises(HTTPException) as exc:
        _run_ingest(data, _request(embedder), session, _auth())
    assert exc.value.status_code == 422
    assert secret not in str(exc.value.detail)
    assert session.added == []
    assert session.committed == 0
    assert wal_calls == []


# ── session-complete ───────────────────────────────────────────────


def _prepare_main_state(monkeypatch, *, scanning=True, llm=None, embedder=None):
    llm = llm or _CountingLLM(available=False)
    embedder = embedder or _CountingEmbedder(available=False)
    main.app.state.llm_provider = llm
    main.app.state.embedding_provider = embedder
    main.app.state.settings = _settings(scanning=scanning)
    monkeypatch.setattr(main.crud, "mark_session_completed", lambda *a, **k: (_ for _ in ()).throw(AssertionError("mark_session_completed must not run before scan")))
    monkeypatch.setattr(main.crud, "update_profile_from_recall", lambda *a, **k: None)
    return llm, embedder


def test_session_complete_rejects_secret_before_llm_or_write(monkeypatch):
    secret = _fake_openai_key()
    llm, embedder = _prepare_main_state(monkeypatch, scanning=True)
    session = _FakeSession()
    payload = SessionCompleteRequest(
        session_id="s-1",
        conversation_summary=f"Rotated the key to {secret}",
        agent_key="device-test-writer",
    )

    with pytest.raises(HTTPException) as exc:
        main.session_complete(payload, namespace="default", auth=_auth(), session=session)

    assert exc.value.status_code == 422
    assert secret not in str(exc.value.detail)
    assert llm.summarize_calls == []
    assert embedder.calls == []
    assert session.added == []
    assert session.committed == 0


def test_session_complete_extracted_facts_rejected_before_persist(monkeypatch):
    secret = _fake_openai_key()
    llm = _CountingLLM(available=True, raw=f"Stored credential {secret} for later")
    embedder = _CountingEmbedder(available=True)
    main.app.state.llm_provider = llm
    main.app.state.embedding_provider = embedder
    main.app.state.settings = _settings(scanning=True)
    marked = {"n": 0}
    monkeypatch.setattr(main.crud, "mark_session_completed", lambda *a, **k: marked.__setitem__("n", marked["n"] + 1) or 1)
    monkeypatch.setattr(main.crud, "update_profile_from_recall", lambda *a, **k: None)
    session = _FakeSession()
    payload = SessionCompleteRequest(
        session_id="s-2",
        conversation_summary="We finished the caching work and documented the decision.",
    )

    with pytest.raises(HTTPException) as exc:
        main.session_complete(payload, namespace="default", auth=_auth(), session=session)

    assert exc.value.status_code == 422
    assert secret not in str(exc.value.detail)
    assert llm.summarize_calls  # extraction ran
    assert embedder.calls == []
    assert marked["n"] == 0
    assert session.added == []
    assert session.committed == 0


def test_session_complete_benign_summary_accepts_and_wal_omits_prose(monkeypatch):
    llm = _CountingLLM(available=True, raw="Chose PostgreSQL for durability.\nKeep Top-K small.")
    embedder = _CountingEmbedder(available=False)
    main.app.state.llm_provider = llm
    main.app.state.embedding_provider = embedder
    main.app.state.settings = _settings(scanning=True)
    monkeypatch.setattr(main.crud, "mark_session_completed", lambda *a, **k: 3)
    monkeypatch.setattr(main.crud, "update_profile_from_recall", lambda *a, **k: None)
    wal_calls: list = []
    monkeypatch.setattr(main.wal, "log_event", lambda op, **kw: wal_calls.append((op, kw)))
    session = _FakeSession()
    summary = "We decided to use FastAPI for the backend and keep recall Top-K small."
    payload = SessionCompleteRequest(session_id="s-3", conversation_summary=summary, agent_key="device-x-writer")

    result = main.session_complete(payload, namespace="default", auth=_auth(), session=session)

    assert result.facts_extracted == 2
    assert result.recall_events_completed == 3
    assert session.committed == 1
    assert len(session.added) == 2
    assert wal_calls
    op, kw = wal_calls[0]
    assert op == "session.complete"
    dumped = repr(kw)
    assert summary not in dumped
    assert "s-3" not in dumped
    assert "device-x-writer" not in dumped
    assert kw["payload"]["has_session_id"] is True
    assert kw["payload"]["has_agent_key"] is True
    assert "session_id" not in kw["payload"]
    assert "agent_key" not in kw["payload"]
    assert kw["result"]["facts_extracted"] == 2
    assert session.added[0].source == "device-x-writer"


def test_session_complete_unauthorized_namespace_denied(monkeypatch):
    llm, embedder = _prepare_main_state(monkeypatch)
    session = _FakeSession()
    payload = SessionCompleteRequest(
        session_id="s-4",
        conversation_summary="Benign session wrap-up about caching.",
    )
    with pytest.raises(HTTPException) as exc:
        main.session_complete(
            payload,
            namespace="other-ns",
            auth=_auth(namespaces=["default"]),
            session=session,
        )
    assert 403 == exc.value.status_code
    assert llm.summarize_calls == []
    assert session.committed == 0


def test_session_complete_scanner_exception_does_not_fall_through(monkeypatch):
    def _boom(_fields):
        raise RuntimeError("scanner backend failed")

    monkeypatch.setattr(main, "collect_field_rejections", _boom)
    llm, embedder = _prepare_main_state(monkeypatch, scanning=True)
    session = _FakeSession()
    payload = SessionCompleteRequest(
        session_id="s-5",
        conversation_summary="Benign session wrap-up about caching.",
    )
    with pytest.raises(RuntimeError, match="scanner backend failed"):
        main.session_complete(payload, namespace="default", auth=_auth(), session=session)
    assert llm.summarize_calls == []
    assert session.committed == 0


@pytest.mark.parametrize("field", ["agent_key", "session_id"])
def test_session_complete_rejects_secret_in_identity_before_extraction(
    monkeypatch, caplog, field,
):
    secret = _fake_openai_key()
    llm = _CountingLLM(available=True, raw="Chose PostgreSQL for durability.")
    embedder = _CountingEmbedder(available=True)
    _prepare_main_state(monkeypatch, scanning=True, llm=llm, embedder=embedder)
    wal_calls: list = []
    monkeypatch.setattr(main.wal, "log_event", lambda *a, **k: wal_calls.append((a, k)))
    session = _FakeSession()
    kwargs = {
        "session_id": "s-meta",
        "conversation_summary": "We finished the caching work and documented the decision.",
        "agent_key": "device-test-writer",
    }
    kwargs[field] = secret
    payload = SessionCompleteRequest(**kwargs)

    with caplog.at_level("WARNING"):
        with pytest.raises(HTTPException) as exc:
            main.session_complete(payload, namespace="default", auth=_auth(), session=session)

    assert exc.value.status_code == 422
    detail = str(exc.value.detail)
    assert "potential secrets" in detail
    assert field in detail
    assert secret not in detail
    assert secret not in caplog.text
    assert llm.summarize_calls == []
    assert embedder.calls == []
    assert session.added == []
    assert session.committed == 0
    assert wal_calls == []


def test_session_complete_benign_agent_key_persists_as_source_not_wal(monkeypatch):
    llm, embedder = _prepare_main_state(monkeypatch, scanning=True)
    llm.is_available = True
    llm.raw = "Chose PostgreSQL for durability.\nKeep Top-K small."
    monkeypatch.setattr(main.crud, "mark_session_completed", lambda *a, **k: 1)
    wal_calls: list = []
    monkeypatch.setattr(main.wal, "log_event", lambda op, **kw: wal_calls.append((op, kw)))
    session = _FakeSession()
    payload = SessionCompleteRequest(
        session_id="s-benign",
        conversation_summary="We decided to use FastAPI for the backend and keep recall Top-K small.",
        agent_key="device-x-writer",
    )
    result = main.session_complete(payload, namespace="default", auth=_auth(), session=session)
    assert result.facts_extracted == 2
    assert session.added[0].source == "device-x-writer"
    dumped = repr(wal_calls)
    assert "device-x-writer" not in dumped
    assert "s-benign" not in dumped
    assert wal_calls[0][1]["payload"]["has_agent_key"] is True
    assert wal_calls[0][1]["payload"]["has_session_id"] is True


# ── legacy ingest compatibility ────────────────────────────────────


def test_legacy_ingest_batch_still_rejects_secrets_before_embed(monkeypatch):
    from fastapi import HTTPException as FastAPIHTTPException
    from app import crud
    from app.schemas import IngestRequest, KnowledgeItemIn

    monkeypatch.setattr(crud, "ensure_namespace_exists", lambda *a, **k: None)
    secret = _fake_openai_key()
    embedder = _CountingEmbedder()
    session = _FakeSession()
    payload = IngestRequest(
        namespace="default",
        knowledge_items=[KnowledgeItemIn(content=f"Use key {secret}")],
    )
    with pytest.raises(FastAPIHTTPException) as exc:
        crud.ingest_batch(session, payload, embedder, _settings(scanning=True))
    assert exc.value.status_code == 422
    assert secret not in str(exc.value.detail)
    assert embedder.calls == []
    assert session.committed == 0


def test_legacy_ingest_wal_receipt_omits_raw_content(monkeypatch):
    captured: list = []
    monkeypatch.setattr(main.wal, "log_event", lambda op, **kw: captured.append((op, kw)))
    monkeypatch.setattr(
        main.crud,
        "ingest_batch",
        lambda **kwargs: {"entities": 0, "requirements": 0, "knowledge_items": 1, "events": 0},
    )
    from app.schemas import IngestRequest, KnowledgeItemIn

    payload = IngestRequest(
        namespace="default",
        knowledge_items=[KnowledgeItemIn(content="We decided to use FastAPI for the backend")],
    )
    result = main.ingest(payload, auth=_auth(), session=_FakeSession())
    assert result.counts["knowledge_items"] == 1
    assert captured
    op, kw = captured[0]
    assert op == "ingest.batch"
    assert "We decided to use FastAPI" not in repr(kw)
    assert kw["payload"]["knowledge_item_count"] == 1
    assert kw["result"]["counts"]["knowledge_items"] == 1
