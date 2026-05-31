import asyncio

from app.gateway.executors.orchestrate import OrchestrateExecutor


def drain(executor, intent="orchestrate", ctx=None):
    ctx = ctx or {}

    async def _go():
        return [e async for e in executor.execute(intent, ctx)]

    return asyncio.run(_go())


LADDER = [
    {"name": "opus", "kind": "anthropic"},
    {"name": "deepseek-v4-pro", "kind": "chain"},
    {"name": "local", "kind": "ollama"},
]


def make_runner(fail_until=None, tokens=("hello", " world"), empty_for=()):
    """Build a fake runner that fails for rung names in ``fail_until`` and
    yields no tokens for rung names in ``empty_for``."""
    fail_until = set(fail_until or ())
    empty_for = set(empty_for or ())

    async def runner(rung, ctx):
        name = rung["name"]
        if name in fail_until:
            raise RuntimeError(f"{name} unavailable")
        if name in empty_for:
            return
        for t in tokens:
            yield t

    return runner


def test_first_rung_success_no_fallover():
    ex = OrchestrateExecutor(LADDER, make_runner())
    events = drain(ex)
    types = [e.type for e in events]
    assert "event" not in types  # no fallover
    assert types[-1] == "done"
    assert events[-1].data == {"model": "opus"}
    tokens = [e.data for e in events if e.type == "token"]
    assert tokens == ["hello", " world"]


def test_fallover_to_second_rung():
    ex = OrchestrateExecutor(LADDER, make_runner(fail_until={"opus"}))
    events = drain(ex)
    fallovers = [e for e in events if e.type == "event"]
    assert len(fallovers) == 1
    assert fallovers[0].data == {"fallover": True, "to": "deepseek-v4-pro"}
    assert events[-1].type == "done"
    assert events[-1].data == {"model": "deepseek-v4-pro"}


def test_fallover_cascades_to_local():
    ex = OrchestrateExecutor(LADDER, make_runner(fail_until={"opus", "deepseek-v4-pro"}))
    events = drain(ex)
    fallover_targets = [e.data["to"] for e in events if e.type == "event"]
    assert fallover_targets == ["deepseek-v4-pro", "local"]
    assert events[-1].data == {"model": "local"}


def test_ladder_exhausted_emits_retryable_error():
    ex = OrchestrateExecutor(
        LADDER, make_runner(fail_until={"opus", "deepseek-v4-pro", "local"})
    )
    events = drain(ex)
    assert events[-1].type == "error"
    assert events[-1].data["retryable"] is True
    # one fallover event per later rung attempted
    assert len([e for e in events if e.type == "event"]) == 2


def test_empty_rung_output_treated_as_failure_and_falls_over():
    ex = OrchestrateExecutor(LADDER, make_runner(empty_for={"opus"}))
    events = drain(ex)
    # opus produced nothing -> fallover to deepseek which succeeds
    assert [e.data["to"] for e in events if e.type == "event"] == ["deepseek-v4-pro"]
    assert events[-1].data == {"model": "deepseek-v4-pro"}


def test_tokens_precede_done():
    ex = OrchestrateExecutor(LADDER, make_runner())
    events = drain(ex)
    done_index = next(i for i, e in enumerate(events) if e.type == "done")
    token_indices = [i for i, e in enumerate(events) if e.type == "token"]
    assert all(i < done_index for i in token_indices)
