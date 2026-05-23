"""Tests for Synapse plane — Phase 1: data + REST surface.

Run against a live API:
    SSOT_TEST_API_KEY=<key> pytest tests/test_synapse.py -v

Unit (schema-only) tests run without any env vars.
"""

from __future__ import annotations

import json
import os
import sys
import uuid

import pytest

BASE_URL = os.environ.get("SSOT_TEST_URL", "http://localhost:8088")
API_KEY = os.environ.get("SSOT_TEST_API_KEY", "")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"


# ═══════════════════════════════════════════════════════════════════
# UNIT TESTS — schema validation only, no DB/HTTP needed
# ═══════════════════════════════════════════════════════════════════


class TestSynapseSchemas:
    def test_session_register_valid(self):
        from app.synapse.schemas import SessionRegister

        r = SessionRegister(session_id="s1", host="myhost", cwd="/home/x", agent="claude")
        assert r.session_id == "s1"
        assert r.repo is None
        assert r.current_file is None

    def test_session_register_with_optionals(self):
        from app.synapse.schemas import SessionRegister

        r = SessionRegister(
            session_id="s2",
            host="h",
            cwd="/tmp",
            repo="my-repo",
            agent="claude",
            current_file="/tmp/foo.py",
            current_op="edit",
        )
        assert r.repo == "my-repo"
        assert r.current_file == "/tmp/foo.py"

    def test_session_register_rejects_empty_session_id(self):
        from app.synapse.schemas import SessionRegister
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SessionRegister(session_id="", host="h", cwd="/tmp", agent="a")

    def test_heartbeat_schema(self):
        from app.synapse.schemas import SessionHeartbeat

        hb = SessionHeartbeat(session_id="s1", current_file="/a/b.py")
        assert hb.session_id == "s1"
        assert hb.current_op is None

    def test_event_create_schema(self):
        from app.synapse.schemas import EventCreate

        ev = EventCreate(session_id="s1", kind="edit", file="/tmp/x.py", line_start=1, line_end=5)
        assert ev.kind == "edit"
        assert ev.payload is None

    def test_session_out_from_attrs(self):
        from datetime import datetime, timezone

        from app.synapse.schemas import SessionOut

        now = datetime.now(timezone.utc)
        s = SessionOut(
            session_id="x",
            host="h",
            cwd="/",
            repo=None,
            agent="a",
            started_at=now,
            last_seen=now,
            current_file=None,
            current_op=None,
        )
        assert s.session_id == "x"

    def test_collision_out_schema(self):
        from datetime import datetime, timezone

        from app.synapse.schemas import CollisionOut

        now = datetime.now(timezone.utc)
        c = CollisionOut(session_id="s1", host="h", cwd="/", last_event_ts=now, kind="edit")
        assert c.kind == "edit"


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — require running API + SSOT_TEST_API_KEY
# ═══════════════════════════════════════════════════════════════════

requires_api = pytest.mark.skipif(not API_KEY, reason="SSOT_TEST_API_KEY not set")


@requires_api
class TestSynapseIntegration:
    @pytest.fixture
    def client(self):
        import httpx

        with httpx.Client(base_url=BASE_URL, timeout=10) as c:
            yield c

    @pytest.fixture
    def headers(self):
        return {"X-API-Key": API_KEY}

    def _uid(self) -> str:
        return f"test-{uuid.uuid4().hex[:8]}"

    # ── auth ──────────────────────────────────────────────────────

    def test_no_key_returns_401(self, client):
        r = client.post("/synapse/session", json={"session_id": "x", "host": "h", "cwd": "/", "agent": "a"})
        assert r.status_code == 401

    def test_bad_key_returns_401(self):
        import httpx
        # Separate client with longer timeout: bcrypt must check all keys for an unknown key
        with httpx.Client(base_url=BASE_URL, timeout=30) as c:
            r = c.post(
                "/synapse/session",
                json={"session_id": "x", "host": "h", "cwd": "/", "agent": "a"},
                headers={"X-API-Key": "ssot_notakey"},
            )
        assert r.status_code == 401

    # ── register ──────────────────────────────────────────────────

    def test_register_session(self, client, headers):
        sid = self._uid()
        r = client.post(
            "/synapse/session",
            json={"session_id": sid, "host": "myhost", "cwd": "/opt/proj", "agent": "claude"},
            headers=headers,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == sid
        assert data["host"] == "myhost"
        assert data["repo"] is None

        # cleanup
        client.delete(f"/synapse/session/{sid}", headers=headers)

    def test_register_upserts_existing(self, client, headers):
        sid = self._uid()
        payload = {"session_id": sid, "host": "host1", "cwd": "/a", "agent": "claude"}
        client.post("/synapse/session", json=payload, headers=headers)

        # re-register with different host
        payload["host"] = "host2"
        r = client.post("/synapse/session", json=payload, headers=headers)
        assert r.status_code == 200
        assert r.json()["host"] == "host2"

        client.delete(f"/synapse/session/{sid}", headers=headers)

    # ── heartbeat ─────────────────────────────────────────────────

    def test_heartbeat_updates_last_seen(self, client, headers):
        sid = self._uid()
        client.post(
            "/synapse/session",
            json={"session_id": sid, "host": "h", "cwd": "/", "agent": "a"},
            headers=headers,
        )

        r = client.post(
            "/synapse/heartbeat",
            json={"session_id": sid, "current_file": "/tmp/foo.py"},
            headers=headers,
        )
        assert r.status_code == 200
        assert r.json()["current_file"] == "/tmp/foo.py"

        client.delete(f"/synapse/session/{sid}", headers=headers)

    def test_heartbeat_404_unknown(self, client, headers):
        r = client.post(
            "/synapse/heartbeat",
            json={"session_id": "does-not-exist"},
            headers=headers,
        )
        assert r.status_code == 404

    # ── event ─────────────────────────────────────────────────────

    def test_create_event(self, client, headers):
        sid = self._uid()
        client.post(
            "/synapse/session",
            json={"session_id": sid, "host": "h", "cwd": "/", "agent": "a"},
            headers=headers,
        )

        r = client.post(
            "/synapse/event",
            json={"session_id": sid, "kind": "edit", "file": "/tmp/foo.py", "line_start": 1, "line_end": 10},
            headers=headers,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["kind"] == "edit"
        assert data["file"] == "/tmp/foo.py"
        assert data["id"] > 0

        client.delete(f"/synapse/session/{sid}", headers=headers)

    def test_event_updates_session_file(self, client, headers):
        """edit/write/bash events update parent session's current_file."""
        sid = self._uid()
        client.post(
            "/synapse/session",
            json={"session_id": sid, "host": "h", "cwd": "/", "agent": "a"},
            headers=headers,
        )
        client.post(
            "/synapse/event",
            json={"session_id": sid, "kind": "write", "file": "/opt/changed.py"},
            headers=headers,
        )

        # Check active to see updated current_file
        r = client.get("/synapse/active", headers=headers)
        sessions = {s["session_id"]: s for s in r.json()}
        assert sessions[sid]["current_file"] == "/opt/changed.py"
        assert sessions[sid]["current_op"] == "write"

        client.delete(f"/synapse/session/{sid}", headers=headers)

    def test_event_404_unknown_session(self, client, headers):
        r = client.post(
            "/synapse/event",
            json={"session_id": "ghost", "kind": "edit"},
            headers=headers,
        )
        assert r.status_code == 404

    # ── active listing ────────────────────────────────────────────

    def test_active_lists_sessions(self, client, headers):
        sid1 = self._uid()
        sid2 = self._uid()
        for sid in (sid1, sid2):
            client.post(
                "/synapse/session",
                json={"session_id": sid, "host": "h", "cwd": "/", "agent": "a"},
                headers=headers,
            )

        r = client.get("/synapse/active", headers=headers)
        assert r.status_code == 200
        ids = {s["session_id"] for s in r.json()}
        assert sid1 in ids
        assert sid2 in ids

        for sid in (sid1, sid2):
            client.delete(f"/synapse/session/{sid}", headers=headers)

    # ── collision detection ───────────────────────────────────────

    def test_collision_detects_second_session(self, client, headers):
        """Two sessions touching the same file; second caller should see the first."""
        sid1 = self._uid()
        sid2 = self._uid()

        for sid in (sid1, sid2):
            client.post(
                "/synapse/session",
                json={"session_id": sid, "host": "h", "cwd": "/opt/proj", "agent": "claude"},
                headers=headers,
            )

        shared_file = f"/opt/proj/shared_{self._uid()}.py"

        # Both sessions touch the shared file
        for sid in (sid1, sid2):
            client.post(
                "/synapse/event",
                json={"session_id": sid, "kind": "edit", "file": shared_file},
                headers=headers,
            )

        # From sid2's perspective, sid1 should appear as a collision
        r = client.get(
            "/synapse/collisions",
            params={"file": shared_file, "exclude_session": sid2},
            headers=headers,
        )
        assert r.status_code == 200
        collision_ids = {c["session_id"] for c in r.json()}
        assert sid1 in collision_ids
        assert sid2 not in collision_ids

        for sid in (sid1, sid2):
            client.delete(f"/synapse/session/{sid}", headers=headers)

    def test_collision_requires_file_param(self, client, headers):
        r = client.get("/synapse/collisions", headers=headers)
        assert r.status_code == 422

    def test_no_collision_without_events(self, client, headers):
        sid = self._uid()
        client.post(
            "/synapse/session",
            json={"session_id": sid, "host": "h", "cwd": "/", "agent": "a"},
            headers=headers,
        )

        r = client.get(
            "/synapse/collisions",
            params={"file": f"/nonexistent/{self._uid()}.py"},
            headers=headers,
        )
        assert r.status_code == 200
        assert r.json() == []

        client.delete(f"/synapse/session/{sid}", headers=headers)

    # ── delete ────────────────────────────────────────────────────

    def test_delete_session(self, client, headers):
        sid = self._uid()
        client.post(
            "/synapse/session",
            json={"session_id": sid, "host": "h", "cwd": "/", "agent": "a"},
            headers=headers,
        )

        r = client.delete(f"/synapse/session/{sid}", headers=headers)
        assert r.status_code == 204

        # Should 404 now
        r = client.post(
            "/synapse/heartbeat",
            json={"session_id": sid},
            headers=headers,
        )
        assert r.status_code == 404

    def test_delete_cascades_events(self, client, headers):
        """Deleting a session should not error even when events exist."""
        sid = self._uid()
        client.post(
            "/synapse/session",
            json={"session_id": sid, "host": "h", "cwd": "/", "agent": "a"},
            headers=headers,
        )
        client.post(
            "/synapse/event",
            json={"session_id": sid, "kind": "edit", "file": "/x.py"},
            headers=headers,
        )

        r = client.delete(f"/synapse/session/{sid}", headers=headers)
        assert r.status_code == 204

    # ── collision dedup (DISTINCT ON fix) ─────────────────────────

    def test_collision_dedup_same_session_once(self, client, headers):
        """Multiple events from the same session on the same file → exactly one row."""
        sid1 = self._uid()
        sid2 = self._uid()
        shared_file = f"/dedup_test_{self._uid()}.py"

        for sid in (sid1, sid2):
            client.post(
                "/synapse/session",
                json={"session_id": sid, "host": "h", "cwd": "/", "agent": "a"},
                headers=headers,
            )

        # Post multiple events for sid1 on the same file (simulates same-ms or rapid inserts)
        for kind in ("read", "edit", "write"):
            client.post(
                "/synapse/event",
                json={"session_id": sid1, "kind": kind, "file": shared_file},
                headers=headers,
            )

        # Also one event from sid2 so it appears as actor
        client.post(
            "/synapse/event",
            json={"session_id": sid2, "kind": "edit", "file": shared_file},
            headers=headers,
        )

        r = client.get(
            "/synapse/collisions",
            params={"file": shared_file, "exclude_session": sid2},
            headers=headers,
        )
        assert r.status_code == 200
        data = r.json()
        # sid1 must appear exactly once
        sid1_rows = [row for row in data if row["session_id"] == sid1]
        assert len(sid1_rows) == 1, f"Expected 1 row for sid1, got {len(sid1_rows)}: {data}"

        for sid in (sid1, sid2):
            client.delete(f"/synapse/session/{sid}", headers=headers)

    def test_collision_exclude_session_filters_named_session(self, client, headers):
        """exclude_session filters out the named session and leaves others in.

        Regression test for the bug where `AND e.session_id != :exclude_session`
        inside the DISTINCT ON subquery referenced the outer alias `e`, causing a
        PostgreSQL 'missing FROM-clause entry for table e' 500 error.
        """
        sid_a = self._uid()
        sid_b = self._uid()
        sid_c = self._uid()
        shared_file = f"/excl_test_{self._uid()}.py"

        for sid in (sid_a, sid_b, sid_c):
            client.post(
                "/synapse/session",
                json={"session_id": sid, "host": "h", "cwd": "/", "agent": "a"},
                headers=headers,
            )

        # All three sessions touch the shared file
        for sid in (sid_a, sid_b, sid_c):
            client.post(
                "/synapse/event",
                json={"session_id": sid, "kind": "edit", "file": shared_file},
                headers=headers,
            )

        # Exclude sid_a — should see sid_b and sid_c but NOT sid_a
        r = client.get(
            "/synapse/collisions",
            params={"file": shared_file, "exclude_session": sid_a},
            headers=headers,
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        returned_ids = {row["session_id"] for row in data}
        assert sid_a not in returned_ids, f"exclude_session sid_a still appears: {data}"
        assert sid_b in returned_ids, f"sid_b missing from results: {data}"
        assert sid_c in returned_ids, f"sid_c missing from results: {data}"

        for sid in (sid_a, sid_b, sid_c):
            client.delete(f"/synapse/session/{sid}", headers=headers)

    # ── SSE stream ────────────────────────────────────────────────

    def test_sse_stream_ready_event(self, client, headers):
        """Connecting to /synapse/stream returns a 'ready' SSE event immediately."""
        import threading

        ready_lines: list[str] = []
        done = threading.Event()

        def consume():
            try:
                with client.stream("GET", "/synapse/stream", headers=headers, timeout=5) as resp:
                    assert resp.status_code == 200
                    for line in resp.iter_lines():
                        if line:
                            ready_lines.append(line)
                        # Stop after we have at least one event: line + data line
                        if len(ready_lines) >= 2:
                            break
            except Exception:
                pass
            finally:
                done.set()

        t = threading.Thread(target=consume, daemon=True)
        t.start()
        done.wait(timeout=6)

        # First real content line should be "event: ready"
        assert any("event: ready" in ln for ln in ready_lines), f"Missing ready event: {ready_lines}"

    def test_sse_stream_receives_event(self, client, headers):
        """Posting an event while streaming yields it on the SSE stream."""
        import threading
        import time

        collected: list[dict] = []
        stop_event = threading.Event()

        sid = self._uid()
        repo_tag = f"ssot-test-{uuid.uuid4().hex[:6]}"

        client.post(
            "/synapse/session",
            json={"session_id": sid, "host": "h", "cwd": "/", "agent": "a", "repo": repo_tag},
            headers=headers,
        )

        def consume():
            try:
                with client.stream(
                    "GET",
                    f"/synapse/stream?repo={repo_tag}",
                    headers=headers,
                    timeout=8,
                ) as resp:
                    for line in resp.iter_lines():
                        if stop_event.is_set():
                            break
                        if line.startswith("data:"):
                            raw = line[len("data:"):].strip()
                            try:
                                obj = json.loads(raw)
                                if obj.get("kind") != "overflow":
                                    collected.append(obj)
                            except Exception:
                                pass
                        if len(collected) >= 1:
                            stop_event.set()
                            break
            except Exception:
                pass

        t = threading.Thread(target=consume, daemon=True)
        t.start()

        # Wait a moment so the subscriber is registered
        time.sleep(0.5)

        client.post(
            "/synapse/event",
            json={"session_id": sid, "kind": "edit", "file": "/sse_test.py"},
            headers=headers,
        )

        stop_event.wait(timeout=5)
        t.join(timeout=2)

        assert len(collected) >= 1, "No event received on SSE stream"
        assert collected[0]["session_id"] == sid
        assert collected[0]["kind"] == "edit"
        assert collected[0]["repo"] == repo_tag

        client.delete(f"/synapse/session/{sid}", headers=headers)

    def test_sse_stream_filter_excludes_non_matching(self, client, headers):
        """Events for repo=other should NOT appear when filtering repo=test-only."""
        import threading
        import time

        repo_tag = f"ssot-test-{uuid.uuid4().hex[:6]}"
        repo_other = f"ssot-other-{uuid.uuid4().hex[:6]}"

        sid = self._uid()
        client.post(
            "/synapse/session",
            json={"session_id": sid, "host": "h", "cwd": "/", "agent": "a", "repo": repo_other},
            headers=headers,
        )

        bad_events: list[dict] = []
        stop_flag = threading.Event()

        def consume():
            try:
                with client.stream(
                    "GET",
                    f"/synapse/stream?repo={repo_tag}",
                    headers=headers,
                    timeout=4,
                ) as resp:
                    for line in resp.iter_lines():
                        if stop_flag.is_set():
                            break
                        if line.startswith("data:"):
                            raw = line[len("data:"):].strip()
                            try:
                                obj = json.loads(raw)
                                if obj.get("kind") not in (None, "overflow") and "session_id" in obj:
                                    bad_events.append(obj)
                            except Exception:
                                pass
            except Exception:
                pass

        t = threading.Thread(target=consume, daemon=True)
        t.start()

        time.sleep(0.5)

        # Post event for the NON-matching repo
        client.post(
            "/synapse/event",
            json={"session_id": sid, "kind": "edit", "file": "/filter_test.py"},
            headers=headers,
        )

        # Wait a bit then stop
        time.sleep(1.5)
        stop_flag.set()
        t.join(timeout=2)

        assert bad_events == [], f"Non-matching events leaked through filter: {bad_events}"

        client.delete(f"/synapse/session/{sid}", headers=headers)

    def test_sse_stream_exclude_session(self, client, headers):
        """exclude_session param: a session should not receive its own events."""
        import threading
        import time

        repo_tag = f"ssot-excl-{uuid.uuid4().hex[:6]}"
        sid = self._uid()
        client.post(
            "/synapse/session",
            json={"session_id": sid, "host": "h", "cwd": "/", "agent": "a", "repo": repo_tag},
            headers=headers,
        )

        echoed: list[dict] = []
        stop_flag = threading.Event()

        def consume():
            try:
                with client.stream(
                    "GET",
                    f"/synapse/stream?repo={repo_tag}&exclude_session={sid}",
                    headers=headers,
                    timeout=4,
                ) as resp:
                    for line in resp.iter_lines():
                        if stop_flag.is_set():
                            break
                        if line.startswith("data:"):
                            raw = line[len("data:"):].strip()
                            try:
                                obj = json.loads(raw)
                                if "session_id" in obj:
                                    echoed.append(obj)
                            except Exception:
                                pass
            except Exception:
                pass

        t = threading.Thread(target=consume, daemon=True)
        t.start()

        time.sleep(0.5)

        client.post(
            "/synapse/event",
            json={"session_id": sid, "kind": "read", "file": "/excl_test.py"},
            headers=headers,
        )

        time.sleep(1.5)
        stop_flag.set()
        t.join(timeout=2)

        # The excluded session's own events must not appear
        own = [e for e in echoed if e.get("session_id") == sid]
        assert own == [], f"Received own event despite exclude_session: {own}"

        client.delete(f"/synapse/session/{sid}", headers=headers)

    def test_sse_stream_no_auth_returns_401(self, client):
        """SSE endpoint requires auth."""
        r = client.get("/synapse/stream", timeout=3)
        assert r.status_code == 401
