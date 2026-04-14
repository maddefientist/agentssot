"""Write-ahead log for ingest/delete operations.

One JSONL line per operation, daily-rotated file, with a redaction pass over
sensitive field names. WAL writes are best-effort: a failure to log must never
block a user-facing write.

Read by: operators debugging "what was written, when, by which key."
Not read by: the application. This is an audit artifact, not a durability log.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .settings import get_settings

logger = logging.getLogger("agentssot.wal")

# Keys whose values are redacted regardless of where they appear in the payload.
# Match is case-insensitive on the key name.
_REDACT_KEYS = {
    "api_key", "apikey", "token", "password", "passphrase", "secret",
    "authorization", "bearer", "private_key", "openai_api_key",
    "x-api-key", "cookie",
}

_REDACTED = "[REDACTED]"

# Cap serialized content to keep WAL lines bounded.
_MAX_CONTENT_CHARS = 2000

_lock = threading.Lock()


def _redact(obj: Any, depth: int = 0) -> Any:
    """Return a copy of obj with sensitive values replaced. Pure, non-mutating."""
    if depth > 6:
        return "[...]"
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in _REDACT_KEYS:
                out[k] = _REDACTED
            else:
                out[k] = _redact(v, depth + 1)
        return out
    if isinstance(obj, list):
        return [_redact(x, depth + 1) for x in obj]
    if isinstance(obj, str) and len(obj) > _MAX_CONTENT_CHARS:
        return obj[:_MAX_CONTENT_CHARS] + f"...[truncated {len(obj) - _MAX_CONTENT_CHARS} chars]"
    return obj


def _wal_dir() -> Path | None:
    s = get_settings()
    if not getattr(s, "wal_enabled", True):
        return None
    d = Path(getattr(s, "wal_dir", "/var/lib/agentssot/wal"))
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("WAL dir unavailable (%s); disabling WAL for this write", e)
        return None
    return d


def _current_path(d: Path) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return d / f"write_log.{day}.jsonl"


def log_event(
    op: str,
    *,
    namespace: str | None,
    actor_key_id: str | None,
    payload: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    """Append one line to today's WAL. Never raises.

    op: short verb, e.g. "knowledge.ingest", "knowledge.delete", "concept.delete"
    actor_key_id: the ApiKey.id performing the write, NOT the raw key.
    payload: the request body (will be redacted).
    result: selected fields from the response (ids, counts). Also redacted.
    """
    d = _wal_dir()
    if d is None:
        return
    try:
        line = json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "op": op,
                "namespace": namespace,
                "actor_key_id": actor_key_id,
                "payload": _redact(payload) if payload is not None else None,
                "result": _redact(result) if result is not None else None,
            },
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError) as e:
        logger.warning("WAL serialization failed for op=%s: %s", op, e)
        return

    path = _current_path(d)
    with _lock:
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
        except OSError as e:
            logger.warning("WAL append failed (%s); continuing", e)


def prune_older_than(days: int = 30) -> int:
    """Delete WAL files older than N days. Returns count removed."""
    d = _wal_dir()
    if d is None:
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    for p in d.glob("write_log.*.jsonl"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            continue
    return removed
