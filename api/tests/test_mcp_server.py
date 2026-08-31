"""api/app/plugin/mcp_server.py — MCP client config refresh + timeout degradation.

Mirrors test_admin_auth.py's harness (sys.path insertion + reload) but points
at this repo's copy of mcp_server.py rather than the installed plugin
location, since the fixes under test (_read_agent_config, timeout
degradation) live only in the repo source.

Skips entirely if the `mcp` package isn't installed in the active
interpreter -- mcp_server.py declares its dependencies via inline PEP 723
script metadata for `uv run`, not api/requirements.txt, so it is not always
present in the API's own test environment.
"""
import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

PLUGIN_PATH = Path(__file__).resolve().parents[1] / "app" / "plugin"
sys.path.insert(0, str(PLUGIN_PATH))


def _write_agent_json(path: Path, **overrides) -> Path:
    cfg = {
        "base_url": "http://127.0.0.1:8088",
        "api_key": "ssot_original_key",
        "default_namespace": "default",
        "device_name": "test-device",
    }
    cfg.update(overrides)
    agent_path = path / "agent.json"
    agent_path.write_text(json.dumps(cfg))
    return agent_path


@pytest.fixture
def mcp_server(tmp_path, monkeypatch):
    agent_path = _write_agent_json(tmp_path)
    monkeypatch.setenv("HIVE_AGENT_JSON", str(agent_path))
    import mcp_server as mod
    importlib.reload(mod)
    return mod


def test_read_agent_config_reflects_live_edits_without_reimport(mcp_server, tmp_path):
    """A long-lived session must pick up an operator's base_url/api_key/
    namespace/device rotation on the NEXT call, without restarting."""
    agent_path = tmp_path / "agent.json"
    cfg_before = mcp_server._read_agent_config()
    assert cfg_before["base_url"] == "http://127.0.0.1:8088"
    assert cfg_before["api_key"] == "ssot_original_key"
    assert cfg_before["default_ns"] == "default"
    assert cfg_before["agent_key"] == "device-test-device-writer"

    _write_agent_json(
        tmp_path,
        base_url="http://10.0.0.9:9999",
        api_key="ssot_rotated_key",
        default_namespace="other-ns",
        device_name="new-device",
    )

    cfg_after = mcp_server._read_agent_config()
    assert cfg_after["base_url"] == "http://10.0.0.9:9999"
    assert cfg_after["api_key"] == "ssot_rotated_key"
    assert cfg_after["default_ns"] == "other-ns"
    assert cfg_after["agent_key"] == "device-new-device-writer"


def test_read_agent_config_falls_back_on_unreadable_file(mcp_server, tmp_path):
    """A best-effort refresh must degrade to last-known-good, not raise."""
    agent_path = tmp_path / "agent.json"
    agent_path.write_text("not json")
    cfg = mcp_server._read_agent_config()
    assert cfg["base_url"] == mcp_server.BASE_URL
    assert cfg["api_key"] == mcp_server.API_KEY


def test_default_namespace_helper_reflects_live_edit(mcp_server, tmp_path):
    assert mcp_server._namespace_or_default("") == "default"
    _write_agent_json(tmp_path, default_namespace="rotated-ns")
    assert mcp_server._namespace_or_default("") == "rotated-ns"
    assert mcp_server._namespace_or_default("explicit-ns") == "explicit-ns"


def test_client_uses_freshly_read_base_url_and_key(mcp_server, tmp_path):
    _write_agent_json(tmp_path, base_url="http://fresh-host:8088", api_key="ssot_fresh_key")

    async def _build():
        return await mcp_server._client()

    client = asyncio.run(_build())
    try:
        assert str(client.base_url) == "http://fresh-host:8088"
        assert client.headers["x-api-key"] == "ssot_fresh_key"
    finally:
        asyncio.run(client.aclose())


def test_admin_client_still_fails_loud_without_admin_json(mcp_server, tmp_path, monkeypatch):
    """Preserve fail-loud admin behavior: role='admin' must still raise
    PermissionError when admin.json is absent, even after the config-refresh
    change to _client()."""
    fake_home = tmp_path / "home"
    (fake_home / ".claude/agentssot/local").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))

    async def _build():
        return await mcp_server._client(role="admin")

    with pytest.raises(PermissionError):
        asyncio.run(_build())


def test_degraded_timeout_message_is_explicit_and_no_retry_hint(mcp_server):
    import httpx

    exc = httpx.TimeoutException("timed out")
    msg = mcp_server._degraded_timeout_message(exc, "try hive_query instead.")
    assert msg.startswith("degraded:")
    assert "synthesis window" in msg
    assert "try hive_query instead." in msg
    # No retry language anywhere in the message.
    assert "retry" not in msg.lower() and "retrying" not in msg.lower()


def test_hive_recall_returns_degraded_message_on_timeout_without_retry(mcp_server, monkeypatch):
    """hive_recall must distinguish a timeout from a hard connection error and
    must not attempt any automatic retry (single call to _client())."""
    import httpx

    call_count = {"n": 0}

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *a, **k):
            raise httpx.TimeoutException("simulated slow backend")

    async def _fake_client(role=None):
        call_count["n"] += 1
        return _FakeClient()

    monkeypatch.setattr(mcp_server, "_client", _fake_client)

    result = asyncio.run(mcp_server.hive_recall("test query"))

    assert result.startswith("degraded:")
    assert "synthesis window" in result
    assert call_count["n"] == 1, "hive_recall must not automatically retry a timed-out call"


def test_hive_recall_still_reports_hard_connection_error_distinctly(mcp_server, monkeypatch):
    import httpx

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *a, **k):
            raise httpx.ConnectError("connection refused")

    async def _fake_client(role=None):
        return _FakeClient()

    monkeypatch.setattr(mcp_server, "_client", _fake_client)

    result = asyncio.run(mcp_server.hive_recall("test query"))

    assert result.startswith("Connection error:")
    assert not result.startswith("degraded:")


def test_hive_recall_defaults_to_bounded_scope_honoring_fast_path(mcp_server, monkeypatch):
    captured = {}

    class _Response:
        status_code = 200

        def json(self):
            return {
                "items": [{
                    "id": "event-1",
                    "scope": "events",
                    "score": 0.2,
                    "snippet": "bounded event",
                }]
            }

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, path, json):
            captured["path"] = path
            captured["body"] = json
            return _Response()

    async def _fake_client(role=None):
        return _FakeClient()

    monkeypatch.setattr(mcp_server, "_client", _fake_client)
    result = asyncio.run(mcp_server.hive_recall("what happened", scope="events", top_k=3))

    assert captured["path"] == "/recall"
    assert captured["body"]["scope"] == "events"
    assert captured["body"]["top_k"] == 3
    assert captured["body"]["rerank"] is False
    assert "[events] bounded event" in result
    assert "item_id=event-1" in result


def test_hive_recall_deep_mode_is_explicitly_knowledge_only(mcp_server):
    result = asyncio.run(mcp_server.hive_recall("query", scope="all", deep=True))
    assert result == "Error: deep mode is a typed knowledge sweep; use scope='knowledge'."


def test_hive_recall_deep_mode_uses_a_total_result_budget(mcp_server, monkeypatch):
    captured = {}

    class _Response:
        status_code = 200

        def json(self):
            return {
                "buckets": {
                    tier: [
                        {"id": f"{tier}-{i}", "abstract": f"{tier} {i}"}
                        for i in range(3)
                    ]
                    for tier in ("command", "rule", "skill", "entity", "decision")
                },
                "diagnostics": {},
            }

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, path, json):
            captured["path"] = path
            captured["body"] = json
            return _Response()

    async def _fake_client(role=None):
        return _FakeClient()

    monkeypatch.setattr(mcp_server, "_client", _fake_client)
    result = asyncio.run(
        mcp_server.hive_recall("deep context", scope="knowledge", top_k=7, deep=True)
    )

    assert captured["path"] == "/api/v1/knowledge/recall"
    assert set(captured["body"]["top_per_tier"].values()) == {2}
    assert result.count("\n  • ") == 7


def test_default_mcp_profile_exposes_only_core_memory_tools(mcp_server):
    names = {tool.name for tool in asyncio.run(mcp_server.mcp.list_tools())}
    assert names == mcp_server._CORE_TOOL_NAMES


def test_irrelevant_feedback_requires_exact_knowledge_item(mcp_server):
    result = asyncio.run(mcp_server.hive_feedback("irrelevant", concept_id="concept-1"))
    assert result == "Error: 'irrelevant' requires the exact knowledge_item_id returned by recall"


def test_hive_status_keeps_agent_config_separate_from_server_config(
    mcp_server, tmp_path, monkeypatch
):
    """The server's `config` object must not overwrite the live agent config.

    That shadowing used to break the later namespace/profile lookups with a
    KeyError while the broad exception reduced the symptom to a vague status
    warning.
    """
    _write_agent_json(
        tmp_path,
        default_namespace="rotated-ns",
        device_name="rotated-device",
    )
    seen = []

    class _Response:
        status_code = 200

        def __init__(self, body):
            self._body = body

        def json(self):
            return self._body

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, path, params=None):
            seen.append(("GET", path, params))
            if path == "/health":
                return _Response({
                    "embedding_available": True,
                    "llm_available": True,
                    "reranker_available": True,
                    "synthesis_enabled": True,
                })
            if path == "/cortex/system-info":
                return _Response({
                    "config": {
                        "synthesis_model": "synth-model",
                        "embedding_model": "embed-model",
                        "synthesis_schedule_hour": 10,
                    },
                    "agents": [{
                        "agent_key": "device-rotated-device-writer",
                        "total_recalls": 2,
                        "total_feedback": 1,
                    }],
                })
            if path == "/cortex/data":
                return _Response({"total": 3, "knowledge_count": 4})
            raise AssertionError(path)

        async def post(self, path, json=None):
            seen.append(("POST", path, json))
            return _Response({
                "buckets": {"rule": []},
                "diagnostics": {
                    "vec_ms": 4,
                    "rerank_ms": 3,
                    "reranker_used": "test-reranker",
                },
            })

    async def _fake_client(role=None):
        return _FakeClient()

    monkeypatch.setattr(mcp_server, "_client", _fake_client)
    result = asyncio.run(mcp_server.hive_status())

    assert "Could not fetch system info" not in result
    assert "Synthesis model: synth-model" in result
    assert "Recalls: 2" in result
    assert all(
        payload.get("namespace") == "rotated-ns"
        for method, path, payload in seen
        if path in {"/api/v1/knowledge/recall", "/cortex/system-info", "/cortex/data"}
    )


def test_no_key_material_is_logged_or_printed():
    """Static guard: api_key values must never be passed to logging/print --
    only used to build request headers."""
    src = (PLUGIN_PATH / "mcp_server.py").read_text()
    offenders = []
    for i, line in enumerate(src.splitlines(), 1):
        lower = line.lower()
        if ("log" in lower or "print(" in lower) and "api_key" in lower:
            offenders.append((i, line.strip()))
    assert not offenders, f"possible key-material logging: {offenders}"


def test_inline_dependency_excludes_incompatible_mcp_v2():
    """A fresh `uv run` must not resolve the incompatible mcp 2.x package,
    which no longer exposes mcp.server.fastmcp and prevents server startup."""
    src = (PLUGIN_PATH / "mcp_server.py").read_text()
    assert '"mcp[cli]>=1.0.0,<2.0.0"' in src
