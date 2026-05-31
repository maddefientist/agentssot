"""Executor registry — maps an intent class to a ready executor instance.

The gateway service looks an intent up here and dispatches. Wiring primitives
(hive functions, streamers, runners) are injected so the same registry shape
works in tests (fakes) and production (real backends). Every intent the router
can emit MUST have an entry; missing intents fall back to chat-local at the
service layer.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Optional

from ..config import ORCHESTRATE_LADDER
from ..protocol import Event
from .base import Executor
from .chat_local import ChatLocalExecutor
from .dispatch import DispatchExecutor
from .hive_tool import HiveToolExecutor
from .orchestrate import OrchestrateExecutor


class DeferredBriefingExecutor(Executor):
    """Placeholder for the (deferred) proactive briefing capability.

    The HUD renders a briefing artifact when one exists; *generating* one is a
    separate, later thread. Until then this is honest about its absence rather
    than faking content.
    """

    name = "briefing"

    async def execute(self, intent: str, ctx: dict[str, Any]) -> AsyncIterator[Event]:
        yield Event.token(
            "No briefing has been compiled yet — proactive briefing generation "
            "is a deferred capability. Ask me directly and I'll pull what you need."
        )
        yield Event.done({"action": "briefing", "deferred": True})


def build_registry(
    *,
    recall_fn: Callable[[str], Any],
    stats_fn: Callable[[Optional[str]], Any],
    chat_streamer: Callable[[list[dict[str, str]]], "AsyncIterator[str]"],
    orchestrate_runner: Callable[[dict[str, Any], dict[str, Any]], "AsyncIterator[str]"],
    dispatch_runner: Callable[[str, dict[str, Any]], "AsyncIterator[str]"],
    teach_fn: Optional[Callable[[str], Any]] = None,
    ladder: Optional[list[dict[str, Any]]] = None,
    briefing_executor: Optional[Executor] = None,
) -> dict[str, Executor]:
    return {
        "chat-local": ChatLocalExecutor(chat_streamer),
        "hive-tool": HiveToolExecutor(recall_fn, stats_fn, teach_fn),
        "orchestrate": OrchestrateExecutor(ladder or ORCHESTRATE_LADDER, orchestrate_runner),
        "dispatch": DispatchExecutor(dispatch_runner),
        "briefing": briefing_executor or DeferredBriefingExecutor(),
    }
