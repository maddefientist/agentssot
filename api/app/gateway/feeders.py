"""Status feeders — aggregate live system state for the HUD's ambient panels.

Each source is a zero-arg callable (sync or async). Every source is wrapped so a
single failing/absent source yields ``None`` for its slot rather than blowing up
the whole snapshot — the HUD renders NULL slots gracefully (existing display
rule). This is what lets the HUD show "Opus down, dropped to deepseek" without
the panel itself crashing.
"""
from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Optional, Union

Source = Callable[[], Union[Any, Awaitable[Any]]]


async def _safe(source: Optional[Source]) -> Any:
    if source is None:
        return None
    try:
        result = source()
        if inspect.isawaitable(result):
            result = await result
        return result
    except Exception:  # noqa: BLE001 — a dead source must not kill the snapshot
        return None


async def snapshot_status(
    *,
    hive: Optional[Source] = None,
    executors: Optional[Source] = None,
    fleet: Optional[Source] = None,
    chains: Optional[Source] = None,
) -> dict[str, Any]:
    """Return a single status snapshot. Any slot may be ``None`` if unavailable."""
    return {
        "hive": await _safe(hive),
        "executors": await _safe(executors),
        "fleet": await _safe(fleet),
        "chains": await _safe(chains),
    }
