"""Hive-backed conversational session state.

The gateway holds no conversation state in process memory — a crash or redeploy
must lose nothing, and a thread started on one channel (Telegram/voice later)
must continue on another (the HUD). State therefore lives in Postgres (the same
database that backs hive), in a dedicated ``gateway_session`` table, kept out of
the knowledge graph so chatter never pollutes recall.

``SessionStore`` is written against a small ``Backend`` protocol (``load`` /
``save``) so the policy (trimming, turn shape) is unit-testable with an
in-memory backend, while production uses the SQL backend.
"""
from __future__ import annotations

import json
from typing import Any, Protocol

DEFAULT_MAX_TURNS = 40


class Backend(Protocol):
    async def load(self, session_id: str) -> list[dict[str, Any]]: ...
    async def save(self, session_id: str, turns: list[dict[str, Any]]) -> None: ...


class InMemoryBackend:
    """Non-persistent backend for tests."""

    def __init__(self) -> None:
        self._store: dict[str, list[dict[str, Any]]] = {}

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._store.get(session_id, []))

    async def save(self, session_id: str, turns: list[dict[str, Any]]) -> None:
        self._store[session_id] = list(turns)


class SqlBackend:
    """Postgres-backed store using the project's raw-SQL idiom.

    Each call opens its own short-lived session from the injected
    ``session_factory`` (the app's ``SessionLocal``) so the backend is safe to
    share across requests.
    """

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        from sqlalchemy import text

        with self._session_factory() as session:
            row = session.execute(
                text("SELECT turns FROM gateway_session WHERE session_id = :sid"),
                {"sid": session_id},
            ).first()
        if not row or row[0] is None:
            return []
        turns = row[0]
        # psycopg returns JSONB as parsed Python; tolerate a TEXT fallback.
        if isinstance(turns, str):
            turns = json.loads(turns)
        return list(turns)

    async def save(self, session_id: str, turns: list[dict[str, Any]]) -> None:
        from sqlalchemy import text

        with self._session_factory() as session:
            session.execute(
                text(
                    "INSERT INTO gateway_session (session_id, turns, updated_at) "
                    "VALUES (:sid, CAST(:turns AS JSONB), NOW()) "
                    "ON CONFLICT (session_id) DO UPDATE "
                    "SET turns = CAST(:turns AS JSONB), updated_at = NOW()"
                ),
                {"sid": session_id, "turns": json.dumps(turns)},
            )
            session.commit()


class SessionStore:
    """Load/append conversation turns, trimmed to the most recent ``max_turns``."""

    def __init__(self, backend: Backend, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        self._backend = backend
        self._max_turns = max_turns

    async def history(self, session_id: str) -> list[dict[str, Any]]:
        return await self._backend.load(session_id)

    async def append(self, session_id: str, turn: dict[str, Any]) -> list[dict[str, Any]]:
        turns = await self._backend.load(session_id)
        turns.append(turn)
        if len(turns) > self._max_turns:
            turns = turns[-self._max_turns :]
        await self._backend.save(session_id, turns)
        return turns
