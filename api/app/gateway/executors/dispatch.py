"""Dispatch executor — fleet jobs, builds, chains via hari-core / chain.sh.

The actual job runner (a subprocess wrapper that streams stdout) is injected, so
this class stays a thin streaming adapter and unit tests need no real processes.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Callable

from ..protocol import Event
from .base import Executor

# runner(text, ctx) -> async iterator of output line strings
Runner = Callable[[str, dict[str, Any]], "AsyncIterator[str]"]


class DispatchExecutor(Executor):
    name = "dispatch"

    def __init__(self, runner: Runner) -> None:
        self._runner = runner

    async def execute(self, intent: str, ctx: dict[str, Any]) -> AsyncIterator[Event]:
        try:
            async for line in self._runner(ctx.get("text", ""), ctx):
                yield Event.token(line)
            yield Event.done({"dispatched": True})
        except Exception as exc:  # noqa: BLE001
            yield Event.error(f"dispatch failed: {exc}", retryable=True)
