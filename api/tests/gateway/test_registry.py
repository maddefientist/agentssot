import asyncio

from app.gateway.config import VALID_INTENTS
from app.gateway.executors import build_registry
from app.gateway.executors.base import Executor


def _fakes():
    async def chat_streamer(messages):
        yield "hi"

    async def orchestrate_runner(rung, ctx):
        yield "thought"

    async def dispatch_runner(text, ctx):
        yield "ran"

    return dict(
        recall_fn=lambda q: [],
        stats_fn=lambda ns: {},
        chat_streamer=chat_streamer,
        orchestrate_runner=orchestrate_runner,
        dispatch_runner=dispatch_runner,
    )


def test_registry_covers_all_valid_intents():
    reg = build_registry(**_fakes())
    assert set(reg.keys()) == set(VALID_INTENTS)


def test_registry_values_are_executors():
    reg = build_registry(**_fakes())
    assert all(isinstance(e, Executor) for e in reg.values())


def test_registry_names_match_keys():
    reg = build_registry(**_fakes())
    for key, ex in reg.items():
        assert ex.name == key


def test_deferred_briefing_is_honest():
    reg = build_registry(**_fakes())

    async def _go():
        return [e async for e in reg["briefing"].execute("briefing", {})]

    events = asyncio.run(_go())
    assert events[-1].data.get("deferred") is True
    assert any("deferred" in (e.data or "") for e in events if e.type == "token")
