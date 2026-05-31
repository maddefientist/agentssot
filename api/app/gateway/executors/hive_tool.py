"""Hive-tool executor — deterministic memory operations, no LLM.

recall / teach / stats run as direct hive calls, so they are instant and never
hallucinate. The actual hive functions (wrapping ``crud`` + a DB session +
providers) are injected; this class only decides the sub-action and shapes the
Events. Injected functions may be sync or async — both are tolerated.
"""
from __future__ import annotations

import inspect
import re
from typing import Any, AsyncIterator, Callable, Optional

from ..protocol import Event
from .base import Executor

_TEACH_RE = re.compile(r"\b(teach|remember that|note that|store this|make a note)\b", re.I)
_STATS_RE = re.compile(r"\b(stats|status|how many|memory count|are you (up|online|alive))\b", re.I)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _summarize(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No matching memories."
    lines = []
    for r in results[:5]:
        title = r.get("title") or r.get("snippet") or r.get("content") or str(r)
        lines.append(f"• {str(title)[:160]}")
    return "\n".join(lines)


class HiveToolExecutor(Executor):
    name = "hive-tool"

    def __init__(
        self,
        recall_fn: Callable[[str], Any],
        stats_fn: Callable[[Optional[str]], Any],
        teach_fn: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self._recall_fn = recall_fn
        self._stats_fn = stats_fn
        self._teach_fn = teach_fn

    def _infer_action(self, text: str) -> str:
        if _TEACH_RE.search(text):
            return "teach"
        if _STATS_RE.search(text):
            return "stats"
        return "recall"

    async def execute(self, intent: str, ctx: dict[str, Any]) -> AsyncIterator[Event]:
        text = ctx.get("text", "")
        args = ctx.get("args") or {}
        action = args.get("action") or self._infer_action(text)

        try:
            if action == "stats":
                stats = await _maybe_await(self._stats_fn(ctx.get("namespace")))
                yield Event.event({"hive": "stats", "stats": stats})
                yield Event.done({"action": "stats"})
            elif action == "teach":
                if self._teach_fn is None:
                    yield Event.token(
                        "Teaching from the HUD isn't wired yet — say it in a "
                        "normal session and the SessionEnd hook will persist it."
                    )
                    yield Event.done({"action": "teach", "deferred": True})
                else:
                    result = await _maybe_await(self._teach_fn(text))
                    yield Event.event({"hive": "teach", "result": result})
                    yield Event.token("Stored.")
                    yield Event.done({"action": "teach"})
            else:
                results = await _maybe_await(self._recall_fn(text))
                results = list(results or [])
                yield Event.event({"hive": "recall", "results": results})
                yield Event.token(_summarize(results))
                yield Event.done({"action": "recall", "count": len(results)})
        except Exception as exc:  # noqa: BLE001
            yield Event.error(f"hive-tool failed: {exc}", retryable=False)
