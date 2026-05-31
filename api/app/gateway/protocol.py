"""Wire protocol for the Madi gateway.

Two dataclasses cross the boundary between channels (HUD WebSocket today,
Telegram/voice later) and the gateway:

- ``InboundMessage`` — what a channel sends in.
- ``Event`` — a single streamed item the gateway sends back. Executors yield
  these; the service relays them verbatim to the channel.

Keeping the wire types here (and free of FastAPI / executor imports) means the
router, executors, session store, and tests all share one vocabulary without a
dependency cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

EventType = Literal["token", "event", "error", "done"]


@dataclass
class InboundMessage:
    """A message arriving from a channel.

    ``text`` is the raw user input. ``session_id`` ties turns into a thread
    (persisted in hive). ``intent`` is optional: a channel may force an intent
    (e.g. a slash command) and bypass classification; ``None`` means "let the
    router decide".
    """

    text: str
    session_id: str
    intent: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InboundMessage":
        return cls(
            text=d.get("text", ""),
            session_id=d.get("session_id", ""),
            intent=d.get("intent"),
        )


@dataclass
class Event:
    """One streamed item from an executor.

    type:
      token — an incremental chunk of assistant output (data: str)
      event — a structured side-signal, e.g. a fallover or a tool result
              (data: dict)
      error — a typed failure; data carries ``message`` and ``retryable``
      done  — terminal marker; data carries metadata (e.g. which model served)
    """

    type: EventType
    data: Any = None

    @classmethod
    def token(cls, text: str) -> "Event":
        return cls("token", text)

    @classmethod
    def event(cls, payload: dict[str, Any]) -> "Event":
        return cls("event", payload)

    @classmethod
    def error(cls, message: str, retryable: bool = False) -> "Event":
        return cls("error", {"message": message, "retryable": retryable})

    @classmethod
    def done(cls, meta: Optional[dict[str, Any]] = None) -> "Event":
        return cls("done", meta or {})

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data}
