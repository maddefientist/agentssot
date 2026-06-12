import asyncio

from app.gateway.feeders import snapshot_status


def run(coro):
    return asyncio.run(coro)


def test_all_sources_present():
    snap = run(
        snapshot_status(
            hive=lambda: {"knowledge": 4012},
            executors=lambda: [{"name": "opus", "ok": True}],
            fleet=lambda: {"up": 13, "total": 13},
            chains=lambda: [],
        )
    )
    assert snap["hive"] == {"knowledge": 4012}
    assert snap["fleet"]["up"] == 13
    assert snap["chains"] == []


def test_missing_sources_are_none():
    snap = run(snapshot_status())
    assert snap == {"hive": None, "executors": None, "fleet": None, "chains": None, "synapse": None}


def test_failing_source_isolated_to_its_slot():
    def boom():
        raise RuntimeError("fleet ssh timeout")

    snap = run(snapshot_status(hive=lambda: {"ok": 1}, fleet=boom))
    assert snap["hive"] == {"ok": 1}  # survives
    assert snap["fleet"] is None  # isolated


def test_async_source_supported():
    async def hive():
        return {"async": True}

    snap = run(snapshot_status(hive=hive))
    assert snap["hive"] == {"async": True}
