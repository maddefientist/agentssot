"""Gateway service — one turn, end to end.

handle(msg):
  1. snapshot prior history (before this turn)
  2. persist the user turn
  3. classify intent (explicit > rules > classifier > default)
  4. emit a routing event so the HUD shows which executor/intent is live
  5. dispatch to the executor and relay its Events verbatim, collecting tokens
  6. persist Madi's assembled reply

State lives in hive via the SessionStore, so a restart mid-conversation loses
nothing and a thread can later continue on another channel.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from .config import HIVE_NAMESPACE
from .protocol import Event, InboundMessage
from .router import IntentRouter
from .session import SessionStore


class GatewayService:
    def __init__(
        self,
        router: IntentRouter,
        registry: dict[str, Any],
        session: SessionStore,
        namespace: str = HIVE_NAMESPACE,
    ) -> None:
        self._router = router
        self._registry = registry
        self._session = session
        self._namespace = namespace

    def _executor_for(self, intent: str):
        return self._registry.get(intent) or self._registry["chat-local"]

    async def handle(self, msg: InboundMessage) -> AsyncIterator[Event]:
        # Prior history first, so the executor sees context WITHOUT the current
        # turn (executors append ctx["text"] themselves — no duplication).
        history = await self._session.history(msg.session_id)
        await self._session.append(msg.session_id, {"role": "user", "text": msg.text})

        intent, args = await self._router.classify(msg.text, explicit=msg.intent)
        executor = self._executor_for(intent)

        yield Event.event({"routing": True, "intent": intent, "executor": executor.name})

        collected: list[str] = []
        async for event in executor.execute(intent, {
            "text": msg.text,
            "args": args,
            "history": history,
            "session_id": msg.session_id,
            "namespace": self._namespace,
        }):
            if event.type == "token":
                collected.append(event.data)
            yield event

        reply = "".join(collected).strip()
        if reply:
            await self._session.append(
                msg.session_id, {"role": "madi", "text": reply, "intent": intent}
            )
