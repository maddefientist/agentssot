"""Unit tests for evidence-based Ollama model claim verification."""
from types import SimpleNamespace

import httpx

from app.synthesis import verifier


class _Item:
    def __init__(self, content: str, item_id: str = "00000000-0000-0000-0000-000000000001"):
        self.id = item_id
        self.namespace = "default"
        self.content = content
        self.project_id = None
        self.entity_id = None
        self.entity_refs = []
        self.embedding = None


def _response(status: int, *, payload=None, text: str = "") -> httpx.Response:
    return httpx.Response(
        status,
        json=payload if payload is not None else None,
        text=None if payload is not None else text,
        request=httpx.Request("GET", "http://ollama.test"),
    )


def _settings(mode: str = "alert") -> SimpleNamespace:
    return SimpleNamespace(
        verifier_enabled=True,
        verifier_mode=mode,
        ollama_base_url="http://ollama.test",
        alert_webhook_url="http://alerts.test",
        alert_host_label="test",
        alert_enabled=True,
    )


def test_tags_hit_is_alive(monkeypatch):
    monkeypatch.setattr(
        verifier.httpx,
        "get",
        lambda *args, **kwargs: _response(200, payload={"models": [{"name": "qwen3.5:27b"}]}),
    )
    monkeypatch.setattr(
        verifier.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generate must not run")),
    )

    result = verifier.probe_model("qwen3.5:27b", "http://ollama.test")

    assert result.state == verifier.ProbeState.ALIVE
    assert "/api/tags" in result.evidence


def test_local_tags_miss_is_dead_without_generate(monkeypatch):
    monkeypatch.setattr(
        verifier.httpx,
        "get",
        lambda *args, **kwargs: _response(200, payload={"models": []}),
    )
    monkeypatch.setattr(
        verifier.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generate must not run")),
    )

    result = verifier.probe_model("missing:27b", "http://ollama.test")

    assert result.state == verifier.ProbeState.DEAD
    assert "local model" in result.evidence


def test_cloud_tags_miss_generate_200_is_alive(monkeypatch):
    monkeypatch.setattr(
        verifier.httpx,
        "get",
        lambda *args, **kwargs: _response(200, payload={"models": []}),
    )
    monkeypatch.setattr(
        verifier.httpx,
        "post",
        lambda *args, **kwargs: _response(200, payload={"response": "P"}),
    )

    result = verifier.probe_model("qwen3.5:397b-cloud", "http://ollama.test")

    assert result.state == verifier.ProbeState.ALIVE
    assert "generate HTTP 200" in result.evidence


def test_cloud_tags_miss_generate_404_is_dead(monkeypatch):
    monkeypatch.setattr(
        verifier.httpx,
        "get",
        lambda *args, **kwargs: _response(200, payload={"models": []}),
    )
    monkeypatch.setattr(
        verifier.httpx,
        "post",
        lambda *args, **kwargs: _response(404, payload={"error": "model not found"}),
    )

    result = verifier.probe_model("missing:cloud", "http://ollama.test")

    assert result.state == verifier.ProbeState.DEAD
    assert "HTTP 404" in result.evidence


def test_timeout_is_unknown_and_takes_no_action(monkeypatch):
    monkeypatch.setattr(
        verifier.httpx,
        "get",
        lambda *args, **kwargs: _response(200, payload={"models": []}),
    )

    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(verifier.httpx, "post", timeout)
    monkeypatch.setattr(
        verifier,
        "_load_candidates",
        lambda session: [_Item("Use missing:cloud as the canonical model")],
    )
    actions = {"alert": 0, "supersede": 0, "queue": 0}
    monkeypatch.setattr(
        verifier,
        "_post_summary_alert",
        lambda *args: actions.__setitem__("alert", actions["alert"] + 1),
    )
    monkeypatch.setattr(
        verifier,
        "_enqueue_findings",
        lambda *args: actions.__setitem__("queue", actions["queue"] + 1),
    )
    monkeypatch.setattr(
        verifier,
        "_supersede_findings",
        lambda *args: actions.__setitem__("supersede", actions["supersede"] + 1),
    )

    report = verifier.run_verifier(_settings("supersede"), session=SimpleNamespace())

    assert report.probes[0].state == verifier.ProbeState.UNKNOWN
    assert report.findings == []
    assert actions == {"alert": 0, "supersede": 0, "queue": 0}


def test_asserts_live_against_dead_records_contradiction():
    item = _Item("Use missing:27b; it is the current canonical model.")

    report = verifier.verify_items(
        [item],
        base_url="http://ollama.test",
        probe=lambda model, base: verifier.ProbeResult(
            model, verifier.ProbeState.DEAD, "GET /api/tags membership: absent"
        ),
    )

    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.knowledge_id == item.id
    assert finding.assertion == verifier.AssertionDirection.LIVE
    assert finding.observed == verifier.ProbeState.DEAD


def test_asserts_retired_against_alive_records_contradiction():
    item = _Item("old:cloud is retired and should be avoided.")

    report = verifier.verify_items(
        [item],
        probe=lambda model, base: verifier.ProbeResult(
            model, verifier.ProbeState.ALIVE, "generate HTTP 200"
        ),
    )

    assert len(report.findings) == 1
    assert report.findings[0].assertion == verifier.AssertionDirection.RETIRED
    assert report.findings[0].observed == verifier.ProbeState.ALIVE


def test_alert_mode_never_calls_apply_supersession(monkeypatch):
    item = _Item("Use missing:27b as the current model")
    monkeypatch.setattr(verifier, "_load_candidates", lambda session: [item])
    monkeypatch.setattr(
        verifier,
        "probe_model",
        lambda model, base: verifier.ProbeResult(model, verifier.ProbeState.DEAD, "tags miss"),
    )
    monkeypatch.setattr(verifier, "_enqueue_findings", lambda *args: 0)
    monkeypatch.setattr(verifier, "_post_summary_alert", lambda *args: True)
    monkeypatch.setattr(
        verifier,
        "apply_supersession",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not supersede in alert mode")),
    )

    report = verifier.run_verifier(_settings("alert"), session=SimpleNamespace())

    assert len(report.findings) == 1
    assert report.alert_attempted is True
    assert report.superseded == 0


def test_model_regex_rejects_port_like_ids():
    content = (
        "Ollama listens at localhost:9877 at 12:30. "
        "Use ollama/Qwen3.5:27B and gemma4:cloud; llama3:latest is retired."
    )

    assert verifier.extract_model_ids(content) == [
        "gemma4:cloud",
        "llama3:latest",
        "qwen3.5:27b",
    ]
    assert "localhost:9877" not in verifier.extract_model_ids(content)
    # Timestamp-like numeric suffixes must not leak through as model ids.
    assert "12:30" not in verifier.extract_model_ids(content)


def test_successor_phrase_not_classified_retired():
    content = "kimi-k2.6:cloud is retired, superseded by kimi-k2.7-code:cloud"

    assert (
        verifier.classify_assertion(content, "kimi-k2.7-code:cloud")
        == verifier.AssertionDirection.LIVE
    )
    assert (
        verifier.classify_assertion(content, "kimi-k2.6:cloud")
        == verifier.AssertionDirection.RETIRED
    )


def test_use_x_instead_is_live():
    content = "glm-5:cloud retired — use glm-5.2:cloud instead"

    assert verifier.classify_assertion(content, "glm-5.2:cloud") == verifier.AssertionDirection.LIVE
    assert verifier.classify_assertion(content, "glm-5:cloud") == verifier.AssertionDirection.RETIRED


def test_arrow_successor_is_live():
    content = "qwen3.5:cloud -> qwen3.5:397b-cloud"

    assert (
        verifier.classify_assertion(content, "qwen3.5:397b-cloud")
        == verifier.AssertionDirection.LIVE
    )


def test_verify_items_enforces_candidate_limit():
    items = [
        _Item(f"Use qwen3.5:27b (item {n})", item_id=f"00000000-0000-0000-0000-{n:012d}")
        for n in range(verifier.CANDIDATE_LIMIT + 10)
    ]

    report = verifier.verify_items(
        items,
        probe=lambda model, base: verifier.ProbeResult(
            model, verifier.ProbeState.DEAD, "tags miss"
        ),
    )

    assert report.candidates_scanned == verifier.CANDIDATE_LIMIT
    assert len(items) == verifier.CANDIDATE_LIMIT + 10


def test_run_verifier_does_not_close_caller_session(monkeypatch):
    monkeypatch.setattr(verifier, "_load_candidates", lambda session: [])
    monkeypatch.setattr(verifier, "_post_summary_alert", lambda *args: True)

    class _SentinelSession:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    session = _SentinelSession()

    verifier.run_verifier(_settings("alert"), session=session)

    assert session.closed is False


def test_unknown_takes_no_action_in_alert_mode(monkeypatch):
    monkeypatch.setattr(
        verifier.httpx,
        "get",
        lambda *args, **kwargs: _response(200, payload={"models": []}),
    )

    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(verifier.httpx, "post", timeout)
    monkeypatch.setattr(
        verifier,
        "_load_candidates",
        lambda session: [_Item("Use missing:cloud as the canonical model")],
    )
    actions = {"alert": 0, "supersede": 0, "queue": 0}
    monkeypatch.setattr(
        verifier,
        "_post_summary_alert",
        lambda *args: actions.__setitem__("alert", actions["alert"] + 1),
    )
    monkeypatch.setattr(
        verifier,
        "_enqueue_findings",
        lambda *args: actions.__setitem__("queue", actions["queue"] + 1),
    )
    monkeypatch.setattr(
        verifier,
        "_supersede_findings",
        lambda *args: actions.__setitem__("supersede", actions["supersede"] + 1),
    )

    report = verifier.run_verifier(_settings("alert"), session=SimpleNamespace())

    assert report.probes[0].state == verifier.ProbeState.UNKNOWN
    assert report.findings == []
    assert report.alert_attempted is False
    assert actions == {"alert": 0, "supersede": 0, "queue": 0}


def test_alert_mode_enqueues_knowledge_item_level_review():
    finding = verifier.Finding(
        "00000000-0000-0000-0000-000000000001",
        "default",
        "missing:27b",
        verifier.AssertionDirection.LIVE,
        verifier.ProbeState.DEAD,
        "tags miss",
    )

    class _Session:
        def __init__(self):
            self.added = []

        def scalar(self, statement):
            return None

        def add(self, value):
            self.added.append(value)

    session = _Session()

    assert verifier._enqueue_findings(session, [finding]) == 1
    assert len(session.added) == 1
    queued = session.added[0]
    assert queued.kind == verifier.ReviewQueueKind.contradiction
    assert str(queued.primary_id) == finding.knowledge_id
    assert queued.secondary_id is None


def test_supersede_mode_calls_only_lifecycle_path_once_per_knowledge_item(monkeypatch):
    item = _Item("Use missing:27b and absent:cloud as current models")
    findings = [
        verifier.Finding(
            item.id,
            item.namespace,
            model,
            verifier.AssertionDirection.LIVE,
            verifier.ProbeState.DEAD,
            "verified dead",
        )
        for model in ("missing:27b", "absent:cloud")
    ]

    class _Session:
        def __init__(self):
            self.added = []

        def add(self, value):
            self.added.append(value)

        def flush(self):
            self.added[-1].id = "00000000-0000-0000-0000-000000000099"

    calls = []
    monkeypatch.setattr(verifier, "apply_supersession", lambda old, new: calls.append((old, new)))
    session = _Session()

    count = verifier._supersede_findings(
        session,
        [item],
        findings,
        verifier.datetime(2026, 7, 23, tzinfo=verifier.UTC),
    )

    assert count == 1
    assert len(calls) == 1
    assert calls[0][0] is item
    assert "missing:27b is DEAD" in calls[0][1].content
    assert "absent:cloud is DEAD" in calls[0][1].content
