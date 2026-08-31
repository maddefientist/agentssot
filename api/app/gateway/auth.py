"""Short-lived, single-use credentials for browser gateway transports.

Browsers cannot attach ``X-API-Key`` to WebSocket or EventSource handshakes.
The regular authenticated HTTP API therefore exchanges an API key for a
single-use ticket.  Tickets are scoped to one transport and never persisted.
"""
from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Callable, Literal

from ..security import AuthContext

TicketScope = Literal["websocket", "status"]


@dataclass(frozen=True)
class _Ticket:
    auth: AuthContext
    scope: TicketScope
    expires_at: float


class TicketManager:
    """Issue and consume bounded in-memory gateway tickets."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 60.0,
        max_entries: int = 1024,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._tickets: dict[str, _Ticket] = {}

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _purge(self, now: float) -> None:
        expired = [key for key, ticket in self._tickets.items() if ticket.expires_at <= now]
        for key in expired:
            self._tickets.pop(key, None)
        while len(self._tickets) >= self._max_entries:
            self._tickets.pop(next(iter(self._tickets)))

    def issue(self, auth: AuthContext, scope: TicketScope) -> str:
        now = self._clock()
        self._purge(now)
        token = secrets.token_urlsafe(32)
        self._tickets[self._digest(token)] = _Ticket(
            auth=auth,
            scope=scope,
            expires_at=now + self.ttl_seconds,
        )
        return token

    def consume(self, token: str | None, scope: TicketScope) -> AuthContext | None:
        if not token:
            return None
        now = self._clock()
        ticket = self._tickets.pop(self._digest(token), None)
        if not ticket or ticket.expires_at <= now or ticket.scope != scope:
            return None
        return ticket.auth
