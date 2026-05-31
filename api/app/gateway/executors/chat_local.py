"""Local chat executor — cheap, instant casual conversation via Ollama.

No cloud, no ladder: just the small local model with the session's recent
history for continuity. The token streamer is injected so the executor logic is
testable without a live model.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Awaitable, Callable

from ..config import LOCAL_MODEL, OLLAMA_URL
from ..protocol import Event
from .base import Executor

# streamer(messages) -> async iterator of token strings
Streamer = Callable[[list[dict[str, str]]], "AsyncIterator[str]"]

SYSTEM = (
    "You are Madi, the operator's AI agent. Reply concisely and directly, "
    "in your own grounded voice. You are speaking through the HUD."
)


class ChatLocalExecutor(Executor):
    name = "chat-local"

    def __init__(self, streamer: Streamer) -> None:
        self._streamer = streamer

    def _build_messages(self, ctx: dict[str, Any]) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM}]
        for turn in ctx.get("history", []):
            role = "assistant" if turn.get("role") == "madi" else "user"
            messages.append({"role": role, "content": turn.get("text", "")})
        messages.append({"role": "user", "content": ctx.get("text", "")})
        return messages

    async def execute(self, intent: str, ctx: dict[str, Any]) -> AsyncIterator[Event]:
        messages = self._build_messages(ctx)
        try:
            async for token in self._streamer(messages):
                yield Event.token(token)
            yield Event.done({"model": LOCAL_MODEL})
        except Exception as exc:  # noqa: BLE001
            yield Event.error(f"local chat failed: {exc}", retryable=True)


def make_ollama_streamer(
    base_url: str = OLLAMA_URL, model: str = LOCAL_MODEL
) -> Streamer:
    """Stream tokens from a local Ollama model via the chat API (NDJSON)."""
    import json

    import httpx

    async def _stream(messages: list[dict[str, str]]) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/chat",
                json={"model": model, "stream": True, "messages": messages},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece

    return _stream
