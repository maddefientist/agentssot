import asyncio

from app.gateway.executors.chat_local import ChatLocalExecutor
from app.gateway.executors.dispatch import DispatchExecutor
from app.gateway.executors.hive_tool import HiveToolExecutor


def drain(executor, intent, ctx):
    async def _go():
        return [e async for e in executor.execute(intent, ctx)]

    return asyncio.run(_go())


# --- chat_local ---
def test_chat_local_streams_then_done():
    async def streamer(messages):
        for t in ["Hi", " there"]:
            yield t

    events = drain(ChatLocalExecutor(streamer), "chat-local", {"text": "hello"})
    assert [e.data for e in events if e.type == "token"] == ["Hi", " there"]
    assert events[-1].type == "done"


def test_chat_local_builds_history_into_messages():
    captured = {}

    async def streamer(messages):
        captured["messages"] = messages
        yield "ok"

    ctx = {
        "text": "and now?",
        "history": [
            {"role": "user", "text": "first"},
            {"role": "madi", "text": "reply"},
        ],
    }
    drain(ChatLocalExecutor(streamer), "chat-local", ctx)
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert captured["messages"][-1]["content"] == "and now?"


def test_chat_local_error_is_typed():
    async def streamer(messages):
        raise RuntimeError("model gone")
        yield  # pragma: no cover

    events = drain(ChatLocalExecutor(streamer), "chat-local", {"text": "x"})
    assert events[-1].type == "error"
    assert events[-1].data["retryable"] is True


# --- hive_tool ---
def test_hive_recall_default():
    def recall(query):
        return [{"title": "the gateway design"}, {"title": "obsidian terminal"}]

    def stats(ns):
        return {}

    events = drain(
        HiveToolExecutor(recall, stats), "hive-tool", {"text": "what about the gateway"}
    )
    recall_event = next(e for e in events if e.type == "event")
    assert recall_event.data["hive"] == "recall"
    assert len(recall_event.data["results"]) == 2
    assert events[-1].data == {"action": "recall", "count": 2}


def test_hive_stats_action_inferred():
    def recall(q):
        return []

    def stats(ns):
        return {"knowledge": 4012}

    events = drain(HiveToolExecutor(recall, stats), "hive-tool", {"text": "memory stats?"})
    ev = next(e for e in events if e.type == "event")
    assert ev.data["stats"] == {"knowledge": 4012}
    assert events[-1].data["action"] == "stats"


def test_hive_teach_action():
    taught = {}

    def recall(q):
        return []

    def stats(ns):
        return {}

    def teach(text):
        taught["text"] = text
        return {"id": "abc"}

    events = drain(
        HiveToolExecutor(recall, stats, teach),
        "hive-tool",
        {"text": "remember that the ladder starts with Opus"},
    )
    assert taught["text"].startswith("remember that")
    assert events[-1].data["action"] == "teach"


def test_hive_explicit_action_arg_overrides_inference():
    def recall(q):
        return [{"title": "x"}]

    def stats(ns):
        return {"k": 1}

    # text looks like stats, but args force recall
    events = drain(
        HiveToolExecutor(recall, stats),
        "hive-tool",
        {"text": "status check", "args": {"action": "recall"}},
    )
    assert events[-1].data["action"] == "recall"


def test_hive_async_fn_supported():
    async def recall(q):
        return [{"title": "async result"}]

    def stats(ns):
        return {}

    events = drain(HiveToolExecutor(recall, stats), "hive-tool", {"text": "recall stuff"})
    ev = next(e for e in events if e.type == "event")
    assert ev.data["results"][0]["title"] == "async result"


def test_hive_error_not_retryable():
    def recall(q):
        raise ValueError("db down")

    def stats(ns):
        return {}

    events = drain(HiveToolExecutor(recall, stats), "hive-tool", {"text": "recall x"})
    assert events[-1].type == "error"
    assert events[-1].data["retryable"] is False


# --- dispatch ---
def test_dispatch_streams_lines():
    async def runner(text, ctx):
        for line in ["queued job", "running", "done"]:
            yield line

    events = drain(DispatchExecutor(runner), "dispatch", {"text": "scan fleet"})
    assert [e.data for e in events if e.type == "token"] == ["queued job", "running", "done"]
    assert events[-1].type == "done"
    assert events[-1].data["dispatched"] is True


def test_dispatch_error_typed_retryable():
    async def runner(text, ctx):
        raise RuntimeError("chain.sh not found")
        yield  # pragma: no cover

    events = drain(DispatchExecutor(runner), "dispatch", {"text": "build x"})
    assert events[-1].type == "error"
    assert events[-1].data["retryable"] is True
