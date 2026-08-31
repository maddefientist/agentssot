"""FastAPI surface for the gateway: a WebSocket command channel + an SSE status
stream. Kept thin — all logic lives in the service and feeders.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from contextlib import suppress
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, Depends, Header, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse

from ..models import ApiRole
from ..security import AuthContext, ensure_namespace_access, require_admin, require_api_key
from .auth import TicketManager, TicketScope
from .config import HIVE_NAMESPACE
from .protocol import InboundMessage

ServiceFactory = Callable[[AuthContext], Any]
StatusSnapshot = Callable[[], Awaitable[dict[str, Any]]]
AuthorizationCheck = Callable[[AuthContext], bool]

_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def format_sse(payload: dict[str, Any]) -> str:
    """Render one Server-Sent-Events frame."""
    return f"data: {json.dumps(payload)}\n\n"


def build_router(
    service_factory: ServiceFactory,
    status_snapshot: StatusSnapshot,
    *,
    authorization_check: AuthorizationCheck,
    poll_interval: float = 3.0,
    ticket_manager: TicketManager | None = None,
    gateway_namespace: str = HIVE_NAMESPACE,
    max_text_chars: int = 8000,
    max_frame_bytes: int = 16384,
    max_messages_per_minute: int = 30,
    max_connections: int = 8,
    connection_ttl_seconds: float = 300.0,
    revalidate_seconds: float = 15.0,
) -> APIRouter:
    if (
        max_text_chars <= 0
        or max_frame_bytes <= 0
        or max_messages_per_minute <= 0
        or max_connections <= 0
        or connection_ttl_seconds <= 0
        or revalidate_seconds <= 0
    ):
        raise ValueError("gateway limits must be positive")
    if max_frame_bytes < max_text_chars:
        raise ValueError("max_frame_bytes must be at least max_text_chars")
    router = APIRouter(prefix="/gateway", tags=["gateway"])
    tickets = ticket_manager or TicketManager()
    connection_slots = asyncio.Semaphore(max_connections)

    @router.post("/tickets/{scope}")
    async def gateway_ticket(
        scope: TicketScope,
        auth: AuthContext = Depends(require_api_key),
    ) -> dict[str, Any]:
        # The gateway can recall shared memory and invoke model/tool executors;
        # it is deliberately more privileged than ordinary read-only recall.
        require_admin(auth)
        ensure_namespace_access(auth, gateway_namespace, {ApiRole.reader.value})
        return {
            "ticket": tickets.issue(auth, scope),
            "expires_in_seconds": tickets.ttl_seconds,
        }

    @router.websocket("/ws")
    async def gateway_ws(websocket: WebSocket) -> None:
        offered = [value.strip() for value in websocket.headers.get("sec-websocket-protocol", "").split(",")]
        token = offered[1] if len(offered) == 2 and offered[0] == "agentssot-ticket" else None
        auth = tickets.consume(token, "websocket")
        if auth is None:
            await websocket.close(code=4401, reason="invalid or expired gateway ticket")
            return
        if not await asyncio.to_thread(authorization_check, auth):
            await websocket.close(code=4403, reason="gateway authorization revoked")
            return
        try:
            await asyncio.wait_for(connection_slots.acquire(), timeout=0.05)
        except asyncio.TimeoutError:
            await websocket.close(code=4429, reason="gateway connection limit reached")
            return
        await websocket.accept(subprotocol="agentssot-ticket")
        service = service_factory(auth)
        recent_messages: deque[float] = deque()
        deadline = time.monotonic() + connection_ttl_seconds
        next_revalidation = time.monotonic() + revalidate_seconds
        try:
            while True:
                now = time.monotonic()
                if now >= deadline:
                    await websocket.close(code=1000, reason="gateway connection lifetime reached")
                    return
                if now >= next_revalidation:
                    if not await asyncio.to_thread(authorization_check, auth):
                        await websocket.close(code=4403, reason="gateway authorization revoked")
                        return
                    next_revalidation = now + revalidate_seconds
                receive_timeout = max(0.001, min(deadline, next_revalidation) - now)
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=receive_timeout)
                except asyncio.TimeoutError:
                    continue
                if len(raw.encode("utf-8")) > max_frame_bytes:
                    await websocket.close(code=1009, reason="gateway frame too large")
                    return
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "data": {"message": "message must be valid JSON", "retryable": False}}
                    )
                    continue
                now = time.monotonic()
                while recent_messages and recent_messages[0] <= now - 60.0:
                    recent_messages.popleft()
                if len(recent_messages) >= max_messages_per_minute:
                    await websocket.send_json(
                        {"type": "error", "data": {"message": "gateway rate limit exceeded", "retryable": True}}
                    )
                    continue
                recent_messages.append(now)
                try:
                    msg = InboundMessage.from_dict(data)
                    if len(msg.text) > max_text_chars:
                        raise ValueError(f"text exceeds {max_text_chars} characters")
                    if not _SESSION_ID.fullmatch(msg.session_id):
                        raise ValueError("session_id has an invalid format")
                except ValueError as exc:
                    await websocket.send_json(
                        {"type": "error", "data": {"message": str(exc), "retryable": False}}
                    )
                    continue
                # A caller can resume only sessions created under the same API
                # key; the untrusted client id is never a global database key.
                msg.session_id = f"{auth.key_id}:{msg.session_id}"
                stream = service.handle(msg).__aiter__()
                next_event: asyncio.Task | None = None
                try:
                    while True:
                        if next_event is None:
                            next_event = asyncio.create_task(stream.__anext__())
                        now = time.monotonic()
                        if now >= deadline:
                            await websocket.close(code=1000, reason="gateway connection lifetime reached")
                            return
                        if now >= next_revalidation:
                            if not await asyncio.to_thread(authorization_check, auth):
                                await websocket.close(code=4403, reason="gateway authorization revoked")
                                return
                            next_revalidation = now + revalidate_seconds
                        wait_for = max(0.001, min(deadline, next_revalidation) - now)
                        done, _pending = await asyncio.wait({next_event}, timeout=wait_for)
                        if not done:
                            continue
                        try:
                            event = next_event.result()
                        except StopAsyncIteration:
                            next_event = None
                            break
                        next_event = None
                        await websocket.send_json(event.to_dict())
                finally:
                    if next_event is not None and not next_event.done():
                        next_event.cancel()
                        with suppress(asyncio.CancelledError):
                            await next_event
                    with suppress(Exception):
                        await stream.aclose()
        except WebSocketDisconnect:
            return
        finally:
            connection_slots.release()

    @router.get("/sse/status")
    async def gateway_sse_status(
        ticket: str | None = Header(default=None, alias="X-Gateway-Ticket"),
    ) -> StreamingResponse:
        auth = tickets.consume(ticket, "status")
        if auth is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired gateway ticket",
            )
        if not await asyncio.to_thread(authorization_check, auth):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Gateway authorization revoked",
            )
        try:
            await asyncio.wait_for(connection_slots.acquire(), timeout=0.05)
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Gateway connection limit reached",
            ) from exc

        async def gen():
            try:
                deadline = time.monotonic() + connection_ttl_seconds
                next_revalidation = time.monotonic()
                while True:
                    now = time.monotonic()
                    if now >= deadline:
                        return
                    if now >= next_revalidation:
                        if not await asyncio.to_thread(authorization_check, auth):
                            return
                        next_revalidation = now + revalidate_seconds

                    snapshot_task = asyncio.create_task(status_snapshot())
                    try:
                        while not snapshot_task.done():
                            now = time.monotonic()
                            if now >= deadline:
                                return
                            if now >= next_revalidation:
                                if not await asyncio.to_thread(authorization_check, auth):
                                    return
                                next_revalidation = now + revalidate_seconds
                            wait_for = max(
                                0.001,
                                min(deadline, next_revalidation) - now,
                            )
                            await asyncio.wait({snapshot_task}, timeout=wait_for)
                        snap = snapshot_task.result()
                    finally:
                        if not snapshot_task.done():
                            snapshot_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await snapshot_task

                    yield format_sse(snap)
                    await asyncio.sleep(
                        min(
                            poll_interval,
                            revalidate_seconds,
                            max(0.0, deadline - time.monotonic()),
                        )
                    )
            finally:
                connection_slots.release()

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
