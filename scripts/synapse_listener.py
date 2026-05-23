#!/usr/bin/env python3
"""synapse_listener.py — per-host fleet state consumer daemon.

Subscribes to GET /synapse/stream and materialises a live snapshot of all
OTHER sessions to ~/.claude/synapse/active_fleet.json.  Hooks and MCP tools
read that file without any network round-trip.

Run as a systemd --user unit (see synapse-listener.service).
Also runnable as a foreground process for testing:

    python3 /opt/agentssot/scripts/synapse_listener.py

Exits cleanly on SIGTERM.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [synapse-listener] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("synapse-listener")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict[str, Any]:
    cfg_path = Path(
        os.environ.get("HIVE_AGENT_JSON", Path.home() / ".claude/agentssot/local/agent.json")
    )
    if not cfg_path.exists():
        logger.error("agent.json not found at %s — exiting", cfg_path)
        sys.exit(1)
    try:
        data = json.loads(cfg_path.read_text())
    except Exception as exc:
        logger.error("Cannot parse agent.json: %s — exiting", exc)
        sys.exit(1)
    base_url = data.get("base_url", "http://192.168.1.225:8088").rstrip("/")
    api_key = data.get("api_key") or data.get("admin_api_key", "")
    device_name = data.get("device_name", "unknown")
    if not api_key:
        logger.error("No api_key in agent.json — exiting")
        sys.exit(1)
    return {"base_url": base_url, "api_key": api_key, "device_name": device_name}


cfg = _load_config()
BASE_URL: str = cfg["base_url"]
API_KEY: str = cfg["api_key"]
DEVICE_NAME: str = cfg["device_name"]

HEADERS = {"X-API-Key": API_KEY}
SNAPSHOT_PATH = Path.home() / ".claude/synapse/active_fleet.json"
SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_sessions: dict[str, dict[str, Any]] = {}  # session_id -> session dict
_state_lock = threading.Lock()
_shutdown = threading.Event()

# Backoff parameters for SSE reconnect
_BACKOFF_INITIAL = 1.0   # seconds
_BACKOFF_MAX = 30.0
_REFRESH_INTERVAL = 30   # seconds between full /synapse/active polls

# ---------------------------------------------------------------------------
# Snapshot writer
# ---------------------------------------------------------------------------

def _write_snapshot() -> None:
    """Atomically write the current fleet state to SNAPSHOT_PATH."""
    with _state_lock:
        sessions_list = list(_sessions.values())

    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sessions": sessions_list,
    }
    tmp = SNAPSHOT_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.rename(SNAPSHOT_PATH)
    except Exception as exc:
        logger.warning("Failed to write snapshot: %s", exc)


# ---------------------------------------------------------------------------
# Full refresh from /synapse/active
# ---------------------------------------------------------------------------

def _full_refresh(client: httpx.Client) -> None:
    """Replace in-memory state with a fresh /synapse/active snapshot."""
    try:
        resp = client.get("/synapse/active", params={"since_seconds": 600}, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Full refresh failed: %s", exc)
        return

    rows = resp.json()
    new_state: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = row.get("session_id", "")
        if not sid:
            continue
        # Drop our own device's sessions
        if row.get("host") == DEVICE_NAME:
            continue
        new_state[sid] = {
            "session_id": sid,
            "host": row.get("host"),
            "cwd": row.get("cwd"),
            "repo": row.get("repo"),
            "current_file": row.get("current_file"),
            "current_op": row.get("current_op"),
            "last_seen": row.get("last_seen"),
        }

    with _state_lock:
        _sessions.clear()
        _sessions.update(new_state)

    _write_snapshot()
    logger.info("Full refresh: %d remote sessions", len(new_state))


# ---------------------------------------------------------------------------
# SSE event processor
# ---------------------------------------------------------------------------

def _apply_event(event: dict[str, Any]) -> None:
    """Update in-memory state from a single SSE event payload."""
    kind = event.get("kind")
    if kind == "overflow":
        logger.warning("SSE overflow marker received — forcing full refresh")
        return  # caller handles

    sid = event.get("session_id")
    if not sid:
        return

    host = event.get("host")
    # Drop own-device events (belt-and-suspenders; we may not have excluded them at stream level)
    if host == DEVICE_NAME:
        return

    with _state_lock:
        if kind == "session_end":
            _sessions.pop(sid, None)
            logger.debug("session_end: removed %s", sid)
        else:
            existing = _sessions.get(sid, {})
            existing.update({
                "session_id": sid,
                "host": host or existing.get("host"),
                "cwd": event.get("cwd") or existing.get("cwd"),
                "repo": event.get("repo") or existing.get("repo"),
                "current_file": event.get("file") or existing.get("current_file"),
                "current_op": event.get("kind") or existing.get("current_op"),
                "last_seen": event.get("ts") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
            _sessions[sid] = existing


# ---------------------------------------------------------------------------
# Background periodic refresh thread
# ---------------------------------------------------------------------------

def _refresh_loop() -> None:
    """Runs in a daemon thread; does a full /synapse/active every 30s."""
    with httpx.Client(base_url=BASE_URL, headers=HEADERS) as client:
        while not _shutdown.wait(timeout=_REFRESH_INTERVAL):
            _full_refresh(client)
    logger.debug("refresh_loop exiting")


# ---------------------------------------------------------------------------
# SSE stream consumer
# ---------------------------------------------------------------------------

def _consume_stream() -> None:
    """Main SSE loop with exponential backoff on disconnect."""
    backoff = _BACKOFF_INITIAL

    with httpx.Client(base_url=BASE_URL, headers=HEADERS) as client:
        # Seed state before opening the stream so we start with a consistent snapshot
        _full_refresh(client)

        while not _shutdown.is_set():
            logger.info("Connecting to SSE stream at %s/synapse/stream", BASE_URL)
            try:
                with client.stream(
                    "GET",
                    "/synapse/stream",
                    timeout=httpx.Timeout(None, connect=15.0),
                ) as resp:
                    resp.raise_for_status()
                    backoff = _BACKOFF_INITIAL  # connected — reset backoff
                    logger.info("SSE stream connected")

                    pending = ""
                    for raw_line in resp.iter_lines():
                        if _shutdown.is_set():
                            break

                        line = raw_line.strip()

                        # SSE comment / heartbeat (": ping")
                        if line.startswith(":"):
                            continue

                        # blank line = end of event block; flush pending
                        if not line:
                            if pending:
                                try:
                                    event = json.loads(pending)
                                    if event.get("kind") == "overflow":
                                        logger.warning("Overflow: forcing refresh")
                                        _full_refresh(client)
                                    else:
                                        _apply_event(event)
                                        _write_snapshot()
                                except json.JSONDecodeError:
                                    pass
                                pending = ""
                            continue

                        # "event: ready" lines — skip
                        if line.startswith("event:"):
                            continue

                        # "data: ..." lines — accumulate
                        if line.startswith("data:"):
                            pending = line[len("data:"):].strip()

            except httpx.HTTPStatusError as exc:
                logger.error("SSE HTTP error %s — reconnecting in %.0fs", exc.response.status_code, backoff)
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as exc:
                logger.error("SSE connection lost: %s — reconnecting in %.0fs", exc, backoff)
            except Exception as exc:
                logger.exception("Unexpected SSE error — reconnecting in %.0fs", backoff)

            if not _shutdown.is_set():
                _shutdown.wait(timeout=backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX)

    logger.info("SSE consumer exiting")


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _handle_sigterm(signum: int, frame: Any) -> None:
    logger.info("SIGTERM received — shutting down cleanly")
    _shutdown.set()


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("synapse-listener starting (device=%s, base_url=%s)", DEVICE_NAME, BASE_URL)

    # Start background refresh thread
    t_refresh = threading.Thread(target=_refresh_loop, daemon=True, name="refresh-loop")
    t_refresh.start()

    # Main thread runs SSE consumer (blocks until shutdown)
    _consume_stream()

    logger.info("synapse-listener stopped")
