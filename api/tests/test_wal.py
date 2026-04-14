"""Tests for the write-ahead log.

Unit-level: redaction correctness, file append behavior, rotation by day.
These don't require the API to be running.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestRedaction:
    def test_redacts_known_sensitive_keys(self):
        from app.wal import _redact

        out = _redact({"api_key": "sekrit", "name": "alice"})
        assert out["api_key"] == "[REDACTED]"
        assert out["name"] == "alice"

    def test_redaction_is_case_insensitive(self):
        from app.wal import _redact

        out = _redact({"API_KEY": "x", "Authorization": "Bearer abc"})
        assert out["API_KEY"] == "[REDACTED]"
        assert out["Authorization"] == "[REDACTED]"

    def test_nested_redaction(self):
        from app.wal import _redact

        out = _redact({"outer": {"token": "x", "note": "keep me"}})
        assert out["outer"]["token"] == "[REDACTED]"
        assert out["outer"]["note"] == "keep me"

    def test_list_of_dicts_redaction(self):
        from app.wal import _redact

        out = _redact([{"password": "x"}, {"name": "ok"}])
        assert out[0]["password"] == "[REDACTED]"
        assert out[1]["name"] == "ok"

    def test_long_string_truncated(self):
        from app.wal import _redact

        s = "a" * 5000
        out = _redact({"content": s})
        # Large strings get truncated with a marker
        assert "truncated" in out["content"]
        assert len(out["content"]) < len(s)

    def test_depth_guard(self):
        from app.wal import _redact

        # Build a deeply nested structure
        d: dict = {}
        cur = d
        for _ in range(10):
            cur["next"] = {}
            cur = cur["next"]
        out = _redact(d)
        # Just assert it terminates without recursion error
        assert out is not None


class TestLogEvent:
    def test_log_event_writes_jsonl(self, tmp_path, monkeypatch):
        from app import wal, settings as settings_mod

        # Point wal_dir at a tmp path via a fake settings object
        settings_mod.get_settings.cache_clear()
        monkeypatch.setenv("WAL_DIR", str(tmp_path))
        monkeypatch.setenv("WAL_ENABLED", "true")
        # DATABASE_URL already set above

        wal.log_event(
            "knowledge.ingest",
            namespace="ns1",
            actor_key_id="key-123",
            payload={"content": "hi", "api_key": "dontlog"},
            result={"id": "abc"},
        )

        files = list(tmp_path.glob("write_log.*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["op"] == "knowledge.ingest"
        assert record["namespace"] == "ns1"
        assert record["actor_key_id"] == "key-123"
        assert record["payload"]["api_key"] == "[REDACTED]"
        assert record["result"]["id"] == "abc"

    def test_log_event_never_raises_on_bad_dir(self, monkeypatch):
        from app import wal, settings as settings_mod

        settings_mod.get_settings.cache_clear()
        monkeypatch.setenv("WAL_DIR", "/root/forbidden-path-that-should-not-be-writable")
        monkeypatch.setenv("WAL_ENABLED", "true")

        # Must not raise even if directory is not writable
        wal.log_event(
            "knowledge.ingest",
            namespace="ns",
            actor_key_id="k",
            payload={"content": "x"},
        )

    def test_disabled_wal_is_noop(self, tmp_path, monkeypatch):
        from app import wal, settings as settings_mod

        settings_mod.get_settings.cache_clear()
        monkeypatch.setenv("WAL_DIR", str(tmp_path))
        monkeypatch.setenv("WAL_ENABLED", "false")

        wal.log_event("knowledge.ingest", namespace="ns", actor_key_id="k", payload={})
        files = list(tmp_path.glob("write_log.*.jsonl"))
        assert files == []
