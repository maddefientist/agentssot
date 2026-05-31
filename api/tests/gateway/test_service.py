import asyncio

from app.gateway.protocol import Event, InboundMessage
from app.gateway.router import IntentRouter
from app.gateway.service import GatewayService
from app.gateway.session import InMemoryBackend, SessionStore


class FakeExecutor:
    def __init__(self, name, tokens):
        self.name = name
        self._tokens = tokens
        self.seen_ctx = None

    async def execute(self, intent, ctx):
        self.seen_ctx = ctx
        for t in self._tokens:
            yield Event.token(t)
        yield Event.done({"model": self.name})


def build(intent="chat-local", tokens=("hi", " there")):
    ex = FakeExecutor(intent, tokens)
    # router with a classifier that always returns the chosen intent
    async def classifier(_text):
        return intent

    router = IntentRouter(classifier=classifier)
    registry = {"chat-local": ex, intent: ex}
    store = SessionStore(InMemoryBackend())
    return GatewayService(router, registry, store), ex, store


def drain(service, msg):
    async def _go():
        return [e async for e in service.handle(msg)]

    return asyncio.run(_go())


def test_emits_routing_event_first():
    service, ex, _ = build(intent="orchestrate", tokens=("x",))
    events = drain(service, InboundMessage("ponder this", "s1"))
    assert events[0].type == "event"
    assert events[0].data == {"routing": True, "intent": "orchestrate", "executor": "orchestrate"}


def test_streams_executor_tokens_and_done():
    service, ex, _ = build(tokens=("hello", " world"))
    events = drain(service, InboundMessage("hello", "s1"))
    assert [e.data for e in events if e.type == "token"] == ["hello", " world"]
    assert events[-1].type == "done"


def test_persists_user_and_madi_turns():
    service, ex, store = build(tokens=("reply text",))
    drain(service, InboundMessage("user text", "s1"))
    hist = asyncio.run(store.history("s1"))
    assert [t["role"] for t in hist] == ["user", "madi"]
    assert hist[0]["text"] == "user text"
    assert hist[1]["text"] == "reply text"
    assert hist[1]["intent"] == "chat-local"


def test_executor_sees_prior_history_not_current_turn():
    service, ex, store = build(tokens=("ok",))
    drain(service, InboundMessage("first", "s1"))
    drain(service, InboundMessage("second", "s1"))
    # On the second turn, ctx history should hold only the first exchange
    roles = [t["role"] for t in ex.seen_ctx["history"]]
    texts = [t["text"] for t in ex.seen_ctx["history"]]
    assert roles == ["user", "madi"]
    assert texts == ["first", "ok"]
    assert "second" not in texts  # current turn not duplicated into history


def test_explicit_intent_routes_directly():
    service, ex, _ = build(intent="dispatch", tokens=("ran",))
    events = drain(service, InboundMessage("scan fleet", "s1", intent="dispatch"))
    assert events[0].data["intent"] == "dispatch"


def test_empty_reply_not_persisted_as_madi_turn():
    service, ex, store = build(tokens=())  # executor yields no tokens
    drain(service, InboundMessage("hi", "s1"))
    hist = asyncio.run(store.history("s1"))
    assert [t["role"] for t in hist] == ["user"]  # no empty madi turn
