"""Orchestrate executor — the reliability heart.

Real reasoning / tool-use / orchestration runs here, behind a fallback ladder
(Opus → deepseek-v4-pro → glm-flash → local). The ladder is *config*
(:data:`app.gateway.config.ORCHESTRATE_LADDER`); how each rung is actually
invoked is the injected ``runner``'s job. This keeps the failover policy
declarative and the executor model-agnostic — exactly the "strength in the
architecture, not the model" principle.

Behaviour:
- Try each rung top-to-bottom.
- On moving to a later rung (because an earlier one failed/was unavailable),
  emit a visible ``{"fallover": True, "to": <rung>}`` event so the HUD shows the
  drop (you SEE when Opus is down and it fell to deepseek).
- On a rung succeeding, stream its tokens then emit ``done`` with the model.
- Only when every rung fails do we emit a typed, retryable ``error`` — a single
  session never crashes.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Awaitable, Callable

from ..protocol import Event
from .base import Executor

logger = logging.getLogger("agentssot.gateway.orchestrate")

# runner(rung, ctx) -> async iterator of token strings; raises on failure.
Runner = Callable[[dict[str, Any], dict[str, Any]], "AsyncIterator[str]"]


class OrchestrateExecutor(Executor):
    name = "orchestrate"

    def __init__(self, ladder: list[dict[str, Any]], runner: Runner) -> None:
        self._ladder = ladder
        self._runner = runner

    async def execute(self, intent: str, ctx: dict[str, Any]) -> AsyncIterator[Event]:
        last_error: Exception | None = None

        for index, rung in enumerate(self._ladder):
            if index > 0:
                logger.warning(
                    "orchestrate fallover -> %s (prev failed: %s)",
                    rung.get("name"),
                    last_error,
                )
                yield Event.event({"fallover": True, "to": rung.get("name")})

            try:
                produced = False
                async for token in self._runner(rung, ctx):
                    produced = True
                    yield Event.token(token)
                # Treat an empty stream as a non-failure only if a token was
                # produced; an entirely empty rung falls over (likely a stub).
                if not produced:
                    raise RuntimeError(f"{rung.get('name')} produced no output")
                yield Event.done({"model": rung.get("name")})
                return
            except Exception as exc:  # noqa: BLE001 — ladder tolerates any rung failure
                last_error = exc
                continue

        yield Event.error(
            f"orchestrate ladder exhausted ({len(self._ladder)} rungs); "
            f"last error: {last_error}",
            retryable=True,
        )
