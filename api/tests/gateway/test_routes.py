import asyncio
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.gateway.protocol import Event, InboundMessage
from app.gateway.routes import build_router, format_sse


class FakeService:
    async def handle(self, msg: InboundMessage):
        yield Event.event({"routing": True, "intent": "chat-local", "executor": "chat-local"})
        yield Event.token(f"echo:{msg.text}")
        yield Event.done({"model": "fake"})


def make_app():
    app = FastAPI()

    async def snap():
        return {"hive": {"knowledge": 1}, "executors": None, "fleet": None, "chains": None}

    app.include_router(build_router(lambda: FakeService(), snap))
    return app


def test_format_sse_frame():
    frame = format_sse({"a": 1})
    assert frame == 'data: {"a": 1}\n\n'


def test_ws_round_trip():
    client = TestClient(make_app())
    with client.websocket_connect("/gateway/ws") as ws:
        ws.send_json({"text": "hello", "session_id": "s1"})
        first = ws.receive_json()
        assert first == {"type": "event", "data": {"routing": True, "intent": "chat-local", "executor": "chat-local"}}
        tok = ws.receive_json()
        assert tok == {"type": "token", "data": "echo:hello"}
        done = ws.receive_json()
        assert done["type"] == "done"


def test_ws_handles_multiple_messages():
    client = TestClient(make_app())
    with client.websocket_connect("/gateway/ws") as ws:
        for text in ["one", "two"]:
            ws.send_json({"text": text, "session_id": "s1"})
            ws.receive_json()  # routing
            tok = ws.receive_json()
            assert tok["data"] == f"echo:{text}"
            ws.receive_json()  # done
