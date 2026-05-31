"""Production wiring — assemble a live GatewayService against the running app.

Everything model-/IO-bound is constructed here and injected into the (already
unit-tested) gateway components. Kept in one place so the moving parts — hive
recall via crud, the local Ollama streamer/classifier, the orchestrate fallback
ladder, dispatch via chain.sh — are visible together.

Reachability is environment-shaped, by design:
- ``anthropic`` rung lights up only where ANTHROPIC_API_KEY + SDK exist; on hari
  it fails fast and the ladder falls over (visibly) to the next rung.
- ``chain`` rungs run ~/.claude/scripts/chain.sh (present on hari) → deepseek /
  glm.
- ``ollama`` rung (local qwen) always works → the floor of the ladder.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from .. import crud
from ..db import SessionLocal
from ..schemas import RecallRequest
from .config import HIVE_NAMESPACE, LOCAL_MODEL, OLLAMA_URL, ORCHESTRATE_LADDER
from .chat_local import make_ollama_streamer
from .executors import build_registry
from .feeders import snapshot_status
from .router import IntentRouter, make_ollama_classifier
from .service import GatewayService
from .session import SessionStore, SqlBackend

CHAIN_SH = os.path.expanduser("~/.claude/scripts/chain.sh")
CHAIN_TIMEOUT = 180.0


def _build_brief(ctx: dict[str, Any]) -> str:
    """Flatten recent history + current text into a single prompt for a rung."""
    parts: list[str] = []
    for turn in ctx.get("history", [])[-8:]:
        who = "Madi" if turn.get("role") == "madi" else "Operator"
        parts.append(f"{who}: {turn.get('text', '')}")
    parts.append(f"Operator: {ctx.get('text', '')}")
    return "\n".join(parts)


async def _ollama_oneshot(model: str, prompt: str) -> AsyncIterator[str]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                piece = json.loads(line).get("message", {}).get("content", "")
                if piece:
                    yield piece


async def _anthropic_stream(model: str, prompt: str) -> AsyncIterator[str]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("anthropic SDK not installed") from exc
    client = anthropic.AsyncAnthropic()
    async with client.messages.stream(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def _chain_stream(chain: str, prompt: str) -> AsyncIterator[str]:
    if not Path(CHAIN_SH).exists():
        raise RuntimeError(f"chain.sh not found at {CHAIN_SH}")
    proc = await asyncio.create_subprocess_exec(
        "bash",
        CHAIN_SH,
        chain,
        prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=CHAIN_TIMEOUT)
    except asyncio.TimeoutError as exc:
        proc.kill()
        raise RuntimeError(f"chain {chain} timed out") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"chain {chain} exited {proc.returncode}: {stderr.decode()[:200]}")
    out = stdout.decode().strip()
    if not out:
        raise RuntimeError(f"chain {chain} produced no output")
    yield out


def _make_orchestrate_runner():
    async def runner(rung: dict[str, Any], ctx: dict[str, Any]) -> AsyncIterator[str]:
        kind = rung.get("kind")
        prompt = _build_brief(ctx)
        if kind == "anthropic":
            async for t in _anthropic_stream(rung["model"], prompt):
                yield t
        elif kind == "chain":
            async for t in _chain_stream(rung["chain"], prompt):
                yield t
        elif kind == "ollama":
            async for t in _ollama_oneshot(rung.get("model", LOCAL_MODEL), prompt):
                yield t
        else:
            raise RuntimeError(f"unknown rung kind: {kind}")

    return runner


def _make_dispatch_runner():
    """v1 dispatch: run the requested task through a real quick worker chain.

    Specialised fleet/build verbs (scan fleet, deploy X) are a later refinement;
    for now dispatch executes the task via chain.sh glm-quick and streams output.
    """

    async def runner(text: str, ctx: dict[str, Any]) -> AsyncIterator[str]:
        if Path(CHAIN_SH).exists():
            async for t in _chain_stream("glm-quick", text):
                yield t
        else:
            yield (
                "Dispatch (fleet jobs, builds, chains) runs on the host "
                "toolchain (chain.sh), which isn't reachable from the gateway "
                "container yet \u2014 a known v1 limitation. Request noted: "
                + text
            )

    return runner


def _recall_fn(app):
    def recall(query: str) -> list[dict[str, Any]]:
        with SessionLocal() as session:
            req = RecallRequest(
                namespace=HIVE_NAMESPACE, query_text=query, scope="all", top_k=8
            )
            try:
                rows = crud.recall(
                    session,
                    req,
                    app.state.embedding_provider,
                    app.state.reranker_provider,
                    app.state.settings,
                )
            except Exception:
                return []
        out = []
        for r in rows:
            out.append(
                {
                    "title": r.get("title") or (r.get("snippet") or "")[:80],
                    "snippet": r.get("snippet"),
                    "score": r.get("score"),
                }
            )
        return out

    return recall


def _stats_fn():
    def stats(ns):
        with SessionLocal() as session:
            try:
                return crud.get_namespace_stats(session, ns or HIVE_NAMESPACE)
            except Exception:
                return None

    return stats


def _executor_health() -> list[dict[str, Any]]:
    """Cheap availability read for the ladder rungs (no network calls)."""
    health = []
    for rung in ORCHESTRATE_LADDER:
        kind = rung.get("kind")
        if kind == "anthropic":
            ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
        elif kind == "chain":
            ok = Path(CHAIN_SH).exists()
        elif kind == "ollama":
            ok = True
        else:
            ok = False
        health.append({"name": rung.get("name"), "kind": kind, "available": ok})
    return health


def build_gateway(app):
    """Return ``(service_factory, status_snapshot)`` wired to the live app."""
    recall_fn = _recall_fn(app)
    stats_fn = _stats_fn()

    registry = build_registry(
        recall_fn=recall_fn,
        stats_fn=stats_fn,
        teach_fn=None,  # HUD-side teach deferred; SessionEnd hook still captures
        chat_streamer=make_ollama_streamer(),
        orchestrate_runner=_make_orchestrate_runner(),
        dispatch_runner=_make_dispatch_runner(),
    )
    router = IntentRouter(classifier=make_ollama_classifier())
    store = SessionStore(SqlBackend(SessionLocal))
    service = GatewayService(router, registry, store)

    async def status_snapshot() -> dict[str, Any]:
        return await snapshot_status(
            hive=lambda: stats_fn(None),
            executors=_executor_health,
            fleet=None,  # reuse fleet-dashboard (:9105) later, do not rebuild
            chains=None,
        )

    return (lambda: service), status_snapshot
