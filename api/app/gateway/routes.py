"""FastAPI surface for the gateway: a WebSocket command channel + an SSE status
stream. Kept thin — all logic lives in the service and feeders.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .protocol import InboundMessage

ServiceFactory = Callable[[], Any]  # () -> GatewayService
StatusSnapshot = Callable[[], Awaitable[dict[str, Any]]]


def format_sse(payload: dict[str, Any]) -> str:
    """Render one Server-Sent-Events frame."""
    return f"data: {json.dumps(payload)}\n\n"


def build_router(
    service_factory: ServiceFactory,
    status_snapshot: StatusSnapshot,
    *,
    poll_interval: float = 3.0,
) -> APIRouter:
    router = APIRouter(prefix="/gateway", tags=["gateway"])

    @router.websocket("/ws")
    async def gateway_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        service = service_factory()
        try:
            while True:
                data = await websocket.receive_json()
                msg = InboundMessage.from_dict(data)
                async for event in service.handle(msg):
                    await websocket.send_json(event.to_dict())
        except WebSocketDisconnect:
            return

    @router.get("/sse/status")
    async def gateway_sse_status() -> StreamingResponse:
        async def gen():
            while True:
                snap = await status_snapshot()
                yield format_sse(snap)
                await asyncio.sleep(poll_interval)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
