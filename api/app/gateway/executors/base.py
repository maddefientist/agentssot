"""Executor contract — the swappable brain regions behind the gateway.

Every executor exposes the same shape: a ``name`` and an async ``execute`` that
yields :class:`~app.gateway.protocol.Event` items. The gateway neither knows nor
cares what backs an executor (local model, direct DB calls, a cloud ladder, a
subprocess) — adding or swapping one never touches the service, router, or HUD.

``ctx`` is a plain dict the service assembles per turn. Conventional keys:
    text      — the raw user text (str)
    args      — structured args from the router/classifier (dict)
    history   — prior turns for this session (list[dict])
    session_id — the thread id (str)
Executors read what they need and ignore the rest, so the contract can grow
without breaking existing executors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from ..protocol import Event


class Executor(ABC):
    """Base class for all executors."""

    #: Stable identifier, surfaced in logs and the HUD health panel.
    name: str = "executor"

    @abstractmethod
    def execute(self, intent: str, ctx: dict[str, Any]) -> AsyncIterator[Event]:
        """Handle one turn, yielding Events. Implemented as an async generator."""
        raise NotImplementedError
