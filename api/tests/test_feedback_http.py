"""HTTP/transaction-layer regression for the /feedback route (main.submit_feedback).

test_feedback_relevance_floor.py already pins crud.create_concept_feedback's
own logic (sub-threshold query -> matched=False/recorded=False, no rows added
to the fake session). This file pins the layer above it: main.py's own
transaction handling around that result -- a sub-threshold ("recorded": False)
result must roll back the session instead of committing, so nothing an
unrelated flush produced in the same request can leak through.
"""
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@127.0.0.1/test")

from app import main, crud, schemas
from app.security import AuthContext, ApiRole


@pytest.fixture(autouse=True)
def _no_audit_writes(monkeypatch):
    monkeypatch.setattr(main.wal, "log_event", lambda *args, **kwargs: None)


class _FakeSession:
    """Records commit/rollback calls; never touches a real DB."""

    def __init__(self):
        self.committed = 0
        self.rolled_back = 0

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1


class _FakeEmbedder:
    is_available = True

    def embed_text(self, _text):
        return [0.0] * 768


def _auth(role=ApiRole.writer.value):
    return AuthContext(key_id="k1", key_name="tester", role=role, namespaces=["default"])


def _ensure_app_state(*, fuzzy_feedback_enabled=False):
    # lifespan() never ran in this process (no live DB) -- set only what
    # submit_feedback actually reads.
    main.app.state.embedding_provider = _FakeEmbedder()
    main.app.state.settings = SimpleNamespace(
        fuzzy_feedback_enabled=fuzzy_feedback_enabled,
        feedback_match_max_distance=0.35,
    )


def test_fuzzy_no_match_rolls_back_and_does_not_commit(monkeypatch):
    _ensure_app_state(fuzzy_feedback_enabled=True)
    monkeypatch.setattr(
        crud,
        "create_concept_feedback",
        lambda **kwargs: {
            "matched": False,
            "recorded": False,
            "concept_id": "",
            "signal": "wrong",
            "nearest_concept_id": "c-1",
            "nearest_concept_title": "Some Concept",
            "detail": "no confident match; knowledge_item_id=n/a",
        },
    )
    session = _FakeSession()
    payload = schemas.FeedbackRequest(signal="wrong", query="unrelated recall", note="Unrelated.")

    result = main.submit_feedback(
        payload,
        namespace="default",
        auth=_auth(ApiRole.admin.value),
        session=session,
    )

    assert result.recorded is False
    assert session.rolled_back == 1
    assert session.committed == 0


def test_above_threshold_match_commits_and_does_not_rollback(monkeypatch):
    _ensure_app_state(fuzzy_feedback_enabled=True)
    monkeypatch.setattr(
        crud,
        "create_concept_feedback",
        lambda **kwargs: {
            "matched": True,
            "recorded": True,
            "concept_id": "c-1",
            "signal": "useful",
        },
    )
    session = _FakeSession()
    payload = schemas.FeedbackRequest(signal="useful", query="on-topic recall", note=None)

    result = main.submit_feedback(
        payload,
        namespace="default",
        auth=_auth(ApiRole.admin.value),
        session=session,
    )

    assert result.recorded is True
    assert session.committed == 1
    assert session.rolled_back == 0


def test_knowledge_item_feedback_path_always_commits(monkeypatch):
    """The knowledge_item_id branch is unconditional (crud.create_knowledge_feedback
    raises ValueError on failure rather than returning recorded=False), so it must
    still commit on success."""
    _ensure_app_state()
    monkeypatch.setattr(
        crud,
        "create_knowledge_feedback",
        lambda **kwargs: {"knowledge_item_id": "ki-1", "signal": "useful"},
    )
    ki_id = "11111111-1111-1111-1111-111111111111"
    session = _FakeSession()
    payload = schemas.FeedbackRequest(signal="useful", knowledge_item_id=ki_id)

    result = main.submit_feedback(payload, namespace="default", auth=_auth(), session=session)

    assert result.knowledge_item_id == "ki-1"
    assert session.committed == 1
    assert session.rolled_back == 0


def test_irrelevant_knowledge_feedback_does_not_globally_penalize_item():
    item = SimpleNamespace(
        namespace="default",
        strength=0.8,
        positive_feedback=2,
        negative_feedback=1,
        status="active",
    )

    class _ItemSession:
        def get(self, _model, _item_id):
            return item

        def flush(self):
            pass

    result = crud.create_knowledge_feedback(
        session=_ItemSession(),
        namespace="default",
        signal="irrelevant",
        agent_key="tester",
        knowledge_item_id=uuid4(),
    )

    assert result["recorded"] is True
    assert result["signal"] == "irrelevant"
    assert item.strength == 0.8
    assert item.positive_feedback == 2
    assert item.negative_feedback == 1
    assert item.status == "active"


def test_irrelevant_feedback_requires_exact_knowledge_item_id(monkeypatch):
    _ensure_app_state()
    session = _FakeSession()
    payload = schemas.FeedbackRequest(signal="irrelevant", concept_id=str(uuid4()))

    with pytest.raises(main.HTTPException) as exc:
        main.submit_feedback(payload, namespace="default", auth=_auth(), session=session)

    assert exc.value.status_code == 400
    assert "knowledge_item_id" in exc.value.detail
    assert session.committed == 0
    assert session.rolled_back == 0


def test_fuzzy_feedback_is_disabled_by_default():
    _ensure_app_state()
    session = _FakeSession()
    payload = schemas.FeedbackRequest(signal="useful", query="guess the target")

    with pytest.raises(main.HTTPException) as exc:
        main.submit_feedback(payload, namespace="default", auth=_auth(), session=session)

    assert exc.value.status_code == 400
    assert "exact ID" in exc.value.detail


def test_fuzzy_feedback_requires_admin_when_enabled():
    _ensure_app_state(fuzzy_feedback_enabled=True)
    session = _FakeSession()
    payload = schemas.FeedbackRequest(signal="useful", query="guess the target")

    with pytest.raises(main.HTTPException) as exc:
        main.submit_feedback(payload, namespace="default", auth=_auth(), session=session)

    assert exc.value.status_code == 403
