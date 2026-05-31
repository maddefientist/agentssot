import asyncio

from app.gateway.executors.hive_tool import HiveToolExecutor


def drain(executor, intent, ctx):
    async def _go():
        return [e async for e in executor.execute(intent, ctx)]

    return asyncio.run(_go())


def test_teach_without_teach_fn_is_deferred_honest():
    def recall(q):
        return []

    def stats(ns):
        return {}

    events = drain(
        HiveToolExecutor(recall, stats),  # no teach_fn
        "hive-tool",
        {"text": "remember that the sky is blue"},
    )
    assert events[-1].data == {"action": "teach", "deferred": True}
    assert any("isn't wired" in (e.data or "") for e in events if e.type == "token")
