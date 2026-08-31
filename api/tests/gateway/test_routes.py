import asyncio
import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.gateway.auth import TicketManager
from app.gateway.protocol import Event, InboundMessage
from app.gateway.routes import build_router, format_sse
from app.models import ApiRole
from app.security import AuthContext, require_api_key


class FakeService:
    def __init__(self):
        self.messages = []

    async def handle(self, msg: InboundMessage):
        self.messages.append(msg)
        yield Event.event({"routing": True, "intent": "chat-local", "executor": "chat-local"})
        yield Event.token(f"echo:{msg.text}")
        yield Event.done({"model": "fake"})


def make_app(auth=None, authorization_check=None, service=None, **router_options):
    app = FastAPI()
    service = service or FakeService()
    tickets = router_options.pop("ticket_manager", TicketManager(ttl_seconds=60))

    async def snap():
        return {"hive": {"knowledge": 1}, "executors": None, "fleet": None, "chains": None}

    snapshot = router_options.pop("status_snapshot", snap)

    app.include_router(build_router(
        lambda _auth: service,
        snapshot,
        authorization_check=authorization_check or (lambda _auth: True),
        ticket_manager=tickets,
        **router_options,
    ))
    if auth is not None:
        app.dependency_overrides[require_api_key] = lambda: auth
    return app, service


def admin_auth():
    return AuthContext(
        key_id="admin-key-id",
        key_name="admin",
        role=ApiRole.admin.value,
        namespaces=["default"],
    )


def issue_ticket(client, scope):
    response = client.post(f"/gateway/tickets/{scope}", headers={"X-API-Key": "test"})
    assert response.status_code == 200
    return response.json()["ticket"]


def test_format_sse_frame():
    frame = format_sse({"a": 1})
    assert frame == 'data: {"a": 1}\n\n'


def test_ws_round_trip():
    app, service = make_app(admin_auth())
    client = TestClient(app)
    ticket = issue_ticket(client, "websocket")
    with client.websocket_connect(
        "/gateway/ws", subprotocols=["agentssot-ticket", ticket]
    ) as ws:
        ws.send_json({"text": "hello", "session_id": "s1"})
        first = ws.receive_json()
        assert first == {"type": "event", "data": {"routing": True, "intent": "chat-local", "executor": "chat-local"}}
        tok = ws.receive_json()
        assert tok == {"type": "token", "data": "echo:hello"}
        done = ws.receive_json()
        assert done["type"] == "done"
    assert service.messages[0].session_id == "admin-key-id:s1"


def test_ws_handles_multiple_messages():
    app, _service = make_app(admin_auth())
    client = TestClient(app)
    ticket = issue_ticket(client, "websocket")
    with client.websocket_connect(
        "/gateway/ws", subprotocols=["agentssot-ticket", ticket]
    ) as ws:
        for text in ["one", "two"]:
            ws.send_json({"text": text, "session_id": "s1"})
            ws.receive_json()  # routing
            tok = ws.receive_json()
            assert tok["data"] == f"echo:{text}"
            ws.receive_json()  # done


def test_ws_rejects_missing_ticket():
    app, _service = make_app(admin_auth())
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/gateway/ws"):
            pass
    assert exc.value.code == 4401


def test_ticket_is_single_use():
    app, _service = make_app(admin_auth())
    client = TestClient(app)
    ticket = issue_ticket(client, "websocket")
    with client.websocket_connect(
        "/gateway/ws", subprotocols=["agentssot-ticket", ticket]
    ):
        pass
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            "/gateway/ws", subprotocols=["agentssot-ticket", ticket]
        ):
            pass
    assert exc.value.code == 4401


def test_ticket_requires_authentication():
    app, _service = make_app()
    client = TestClient(app)
    response = client.post("/gateway/tickets/websocket")
    assert response.status_code == 401


def test_ticket_requires_admin_role():
    reader = AuthContext(
        key_id="reader-key-id",
        key_name="reader",
        role=ApiRole.reader.value,
        namespaces=["default"],
    )
    app, _service = make_app(reader)
    client = TestClient(app)
    response = client.post("/gateway/tickets/websocket", headers={"X-API-Key": "test"})
    assert response.status_code == 403


def test_ticket_requires_gateway_namespace():
    auth = admin_auth()
    auth.namespaces = ["other"]
    app, _service = make_app(auth)
    client = TestClient(app)
    response = client.post("/gateway/tickets/status", headers={"X-API-Key": "test"})
    assert response.status_code == 403


def test_sse_rejects_missing_ticket():
    app, _service = make_app(admin_auth())
    client = TestClient(app)
    response = client.get("/gateway/sse/status")
    assert response.status_code == 401


def test_url_ticket_is_not_accepted_or_logged_as_a_credential():
    app, _service = make_app(admin_auth())
    client = TestClient(app)
    ticket = issue_ticket(client, "websocket")
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/gateway/ws?ticket={ticket}"):
            pass
    assert exc.value.code == 4401


def test_ticket_scope_cannot_cross_transports():
    app, _service = make_app(admin_auth())
    client = TestClient(app)
    ticket = issue_ticket(client, "status")
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            "/gateway/ws", subprotocols=["agentssot-ticket", ticket]
        ):
            pass
    assert exc.value.code == 4401


def test_revoked_authorization_rejected_before_ws_accept():
    app, _service = make_app(admin_auth(), authorization_check=lambda _auth: False)
    client = TestClient(app)
    ticket = issue_ticket(client, "websocket")
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            "/gateway/ws", subprotocols=["agentssot-ticket", ticket]
        ):
            pass
    assert exc.value.code == 4403


def test_revocation_closes_already_open_ws():
    state = {"active": True}
    app, _service = make_app(
        admin_auth(),
        authorization_check=lambda _auth: state["active"],
        revalidate_seconds=0.01,
        connection_ttl_seconds=1,
    )
    client = TestClient(app)
    ticket = issue_ticket(client, "websocket")
    with client.websocket_connect(
        "/gateway/ws", subprotocols=["agentssot-ticket", ticket]
    ) as ws:
        state["active"] = False
        time.sleep(0.03)
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 4403


def test_revocation_cancels_an_inflight_service_turn():
    class SlowService:
        async def handle(self, _msg):
            await asyncio.sleep(1)
            yield Event.token("must not escape after revocation")

    state = {"active": True}
    app, _service = make_app(
        admin_auth(),
        authorization_check=lambda _auth: state["active"],
        service=SlowService(),
        revalidate_seconds=0.01,
        connection_ttl_seconds=1,
    )
    client = TestClient(app)
    ticket = issue_ticket(client, "websocket")
    with client.websocket_connect(
        "/gateway/ws", subprotocols=["agentssot-ticket", ticket]
    ) as ws:
        ws.send_json({"text": "slow", "session_id": "s1"})
        state["active"] = False
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 4403


def test_ws_connection_lifetime_is_bounded():
    app, _service = make_app(
        admin_auth(),
        revalidate_seconds=0.01,
        connection_ttl_seconds=0.03,
    )
    client = TestClient(app)
    ticket = issue_ticket(client, "websocket")
    with client.websocket_connect(
        "/gateway/ws", subprotocols=["agentssot-ticket", ticket]
    ) as ws:
        time.sleep(0.05)
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 1000


def test_oversized_ws_frame_closes_before_json_decode():
    app, _service = make_app(
        admin_auth(), max_text_chars=32, max_frame_bytes=64
    )
    client = TestClient(app)
    ticket = issue_ticket(client, "websocket")
    with client.websocket_connect(
        "/gateway/ws", subprotocols=["agentssot-ticket", ticket]
    ) as ws:
        ws.send_text("{" + ("x" * 100))
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 1009


def test_message_rate_limit_is_enforced_per_connection():
    app, _service = make_app(admin_auth(), max_messages_per_minute=1)
    client = TestClient(app)
    ticket = issue_ticket(client, "websocket")
    with client.websocket_connect(
        "/gateway/ws", subprotocols=["agentssot-ticket", ticket]
    ) as ws:
        ws.send_json({"text": "one", "session_id": "s1"})
        ws.receive_json()
        ws.receive_json()
        ws.receive_json()
        ws.send_json({"text": "two", "session_id": "s1"})
        limited = ws.receive_json()
        assert limited["type"] == "error"
        assert "rate limit" in limited["data"]["message"]


def test_connection_limit_rejects_excess_socket():
    app, _service = make_app(admin_auth(), max_connections=1)
    client = TestClient(app)
    first_ticket = issue_ticket(client, "websocket")
    second_ticket = issue_ticket(client, "websocket")
    with client.websocket_connect(
        "/gateway/ws", subprotocols=["agentssot-ticket", first_ticket]
    ):
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(
                "/gateway/ws", subprotocols=["agentssot-ticket", second_ticket]
            ):
                pass
        assert exc.value.code == 4429


def test_sse_shares_the_global_connection_limit():
    app, _service = make_app(admin_auth(), max_connections=1)
    client = TestClient(app)
    ws_ticket = issue_ticket(client, "websocket")
    sse_ticket = issue_ticket(client, "status")
    with client.websocket_connect(
        "/gateway/ws", subprotocols=["agentssot-ticket", ws_ticket]
    ):
        response = client.get(
            "/gateway/sse/status", headers={"X-Gateway-Ticket": sse_ticket}
        )
        assert response.status_code == 429


def test_sse_requires_header_ticket_and_revalidates():
    state = {"active": True}
    app, _service = make_app(
        admin_auth(),
        authorization_check=lambda _auth: state["active"],
        poll_interval=0.005,
        connection_ttl_seconds=0.02,
    )
    client = TestClient(app)
    ticket = issue_ticket(client, "status")
    response = client.get("/gateway/sse/status", headers={"X-Gateway-Ticket": ticket})
    assert response.status_code == 200
    assert "data:" in response.text

    revoked = issue_ticket(client, "status")
    state["active"] = False
    response = client.get("/gateway/sse/status", headers={"X-Gateway-Ticket": revoked})
    assert response.status_code == 403


def _sse_endpoint(app):
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/gateway/sse/status"
    )


@pytest.mark.asyncio
async def test_revocation_cancels_inflight_sse_snapshot_and_releases_slot():
    state = {"active": True}
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked_snapshot():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    tickets = TicketManager(ttl_seconds=60)
    app, _service = make_app(
        admin_auth(),
        authorization_check=lambda _auth: state["active"],
        ticket_manager=tickets,
        status_snapshot=blocked_snapshot,
        revalidate_seconds=0.01,
        connection_ttl_seconds=1,
        max_connections=1,
    )
    endpoint = _sse_endpoint(app)
    response = await endpoint(ticket=tickets.issue(admin_auth(), "status"))
    next_frame = asyncio.create_task(response.body_iterator.__anext__())
    await asyncio.wait_for(started.wait(), timeout=0.2)
    state["active"] = False

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(next_frame, timeout=0.2)
    await asyncio.wait_for(cancelled.wait(), timeout=0.2)

    state["active"] = True
    replacement = await endpoint(ticket=tickets.issue(admin_auth(), "status"))
    await replacement.body_iterator.aclose()


@pytest.mark.asyncio
async def test_sse_lifetime_cancels_inflight_snapshot_and_releases_slot():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked_snapshot():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    tickets = TicketManager(ttl_seconds=60)
    app, _service = make_app(
        admin_auth(),
        ticket_manager=tickets,
        status_snapshot=blocked_snapshot,
        revalidate_seconds=0.01,
        connection_ttl_seconds=0.03,
        max_connections=1,
    )
    endpoint = _sse_endpoint(app)
    response = await endpoint(ticket=tickets.issue(admin_auth(), "status"))
    next_frame = asyncio.create_task(response.body_iterator.__anext__())
    await asyncio.wait_for(started.wait(), timeout=0.2)

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(next_frame, timeout=0.2)
    await asyncio.wait_for(cancelled.wait(), timeout=0.2)

    replacement = await endpoint(ticket=tickets.issue(admin_auth(), "status"))
    await replacement.body_iterator.aclose()
