"""Synapse LISTEN/NOTIFY listener.

Holds a single persistent asyncpg connection, LISTENs on ``synapse_events``,
and fans out each notification to all registered subscriber queues.

Subscriber registry is module-level — one per worker process (uvicorn worker).
Each worker independently opens its own LISTEN connection; Postgres delivers
NOTIFY to all listeners so no worker misses events.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger("agentssot.synapse.listener")

# ──────────────────────────────────────────────────────────────────────────────
# Subscriber registry
# ──────────────────────────────────────────────────────────────────────────────

_QUEUE_MAXSIZE = 100


@dataclass
class Subscriber:
    queue: asyncio.Queue
    filters: dict  # keys: repo, host, session_id, file — all optional
    exclude_session_id: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


# {subscriber_id: Subscriber}
_registry: dict[str, Subscriber] = {}
_registry_lock = asyncio.Lock()


async def register(sub: Subscriber) -> str:
    """Add subscriber to registry. Returns its id."""
    async with _registry_lock:
        _registry[sub.id] = sub
    logger.debug("subscriber registered id=%s filters=%s", sub.id, sub.filters)
    return sub.id


async def unregister(sub_id: str) -> None:
    """Remove subscriber from registry."""
    async with _registry_lock:
        _registry.pop(sub_id, None)
    logger.debug("subscriber unregistered id=%s", sub_id)


def _matches(payload: dict, sub: Subscriber) -> bool:
    """Return True iff the event payload matches the subscriber's filters."""
    # Check exclude_session_id first
    if sub.exclude_session_id and payload.get("session_id") == sub.exclude_session_id:
        return False
    # All set filter keys must match
    for key in ("repo", "host", "session_id", "file"):
        want = sub.filters.get(key)
        if want is not None and payload.get(key) != want:
            return False
    return True


async def _fan_out(payload: dict) -> None:
    """Push payload to every matching subscriber queue."""
    async with _registry_lock:
        subs = list(_registry.values())

    for sub in subs:
        if not _matches(payload, sub):
            continue
        if sub.queue.full():
            # Drop oldest item, then enqueue overflow marker
            try:
                sub.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                sub.queue.put_nowait({"kind": "overflow"})
            except asyncio.QueueFull:
                pass
        else:
            try:
                sub.queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Listener task
# ──────────────────────────────────────────────────────────────────────────────

_RECONNECT_BASE = 1.0
_RECONNECT_CAP = 30.0


def _build_dsn(database_url: str) -> str:
    """Convert a psycopg-style or standard DSN to asyncpg-compatible form.

    asyncpg uses the standard postgresql:// scheme.
    psycopg3 uses postgresql+psycopg:// or similar — strip the driver suffix.
    """
    dsn = database_url
    # Strip SQLAlchemy driver suffix: postgresql+psycopg -> postgresql
    if "+psycopg" in dsn:
        dsn = dsn.replace("+psycopg", "")
    elif "+asyncpg" in dsn:
        dsn = dsn.replace("+asyncpg", "")
    return dsn


async def listener_loop(database_url: str) -> None:
    """Main loop: connect, LISTEN, reconnect on failure with exponential backoff."""
    import asyncpg  # local import so unit tests that mock this can patch easily

    dsn = _build_dsn(database_url)
    backoff = _RECONNECT_BASE

    while True:
        conn = None
        try:
            conn = await asyncpg.connect(dsn)
            logger.info("synapse listener: connected, registering LISTEN synapse_events")

            await conn.add_listener("synapse_events", _handle_notification)
            backoff = _RECONNECT_BASE  # reset after successful connect

            # Keep alive until CancelledError or connection drop
            while not conn.is_closed():
                await asyncio.sleep(5)

        except asyncio.CancelledError:
            logger.info("synapse listener: shutdown requested")
            if conn and not conn.is_closed():
                try:
                    await conn.remove_listener("synapse_events", _handle_notification)
                    await conn.close()
                except Exception:
                    pass
            return

        except Exception as exc:
            logger.error(
                "synapse listener: connection error (%s), reconnecting in %.0fs",
                exc,
                backoff,
            )

        finally:
            if conn and not conn.is_closed():
                try:
                    await conn.close()
                except Exception:
                    pass

        try:
            await asyncio.sleep(backoff)
        except asyncio.CancelledError:
            logger.info("synapse listener: shutdown during backoff")
            return

        backoff = min(backoff * 2, _RECONNECT_CAP)


def _handle_notification(
    connection: object,
    pid: int,
    channel: str,
    payload_str: str,
) -> None:
    """Called by asyncpg in the event loop for each NOTIFY. Schedule fan-out."""
    try:
        payload = json.loads(payload_str)
    except (json.JSONDecodeError, TypeError):
        logger.warning("synapse listener: invalid JSON in notification: %r", payload_str)
        return

    loop = asyncio.get_event_loop()
    loop.create_task(_fan_out(payload))
