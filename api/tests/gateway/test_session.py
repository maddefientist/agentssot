import asyncio

from app.gateway.session import InMemoryBackend, SessionStore


def run(coro):
    return asyncio.run(coro)


def test_empty_history():
    store = SessionStore(InMemoryBackend())
    assert run(store.history("new")) == []


def test_append_then_history():
    store = SessionStore(InMemoryBackend())
    run(store.append("s1", {"role": "user", "text": "hi"}))
    run(store.append("s1", {"role": "madi", "text": "hello"}))
    hist = run(store.history("s1"))
    assert [t["role"] for t in hist] == ["user", "madi"]
    assert hist[1]["text"] == "hello"


def test_sessions_isolated():
    store = SessionStore(InMemoryBackend())
    run(store.append("a", {"role": "user", "text": "a-msg"}))
    run(store.append("b", {"role": "user", "text": "b-msg"}))
    assert run(store.history("a"))[0]["text"] == "a-msg"
    assert run(store.history("b"))[0]["text"] == "b-msg"


def test_trims_to_max_turns():
    store = SessionStore(InMemoryBackend(), max_turns=3)
    for i in range(5):
        run(store.append("s", {"role": "user", "text": str(i)}))
    hist = run(store.history("s"))
    assert len(hist) == 3
    assert [t["text"] for t in hist] == ["2", "3", "4"]


def test_append_returns_current_window():
    store = SessionStore(InMemoryBackend(), max_turns=2)
    run(store.append("s", {"text": "1"}))
    window = run(store.append("s", {"text": "2"}))
    window2 = run(store.append("s", {"text": "3"}))
    assert [t["text"] for t in window] == ["1", "2"]
    assert [t["text"] for t in window2] == ["2", "3"]


def test_load_returns_copy_not_live_reference():
    backend = InMemoryBackend()
    store = SessionStore(backend)
    run(store.append("s", {"text": "1"}))
    hist = run(store.history("s"))
    hist.append({"text": "mutation"})
    # mutating the returned list must not corrupt stored state
    assert len(run(store.history("s"))) == 1
