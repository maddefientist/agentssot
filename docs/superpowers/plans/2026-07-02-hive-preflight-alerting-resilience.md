# Hive Preflight Validation + Webhook Alerting + API Resilience — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop hive (agentssot-api) from silently going unreachable when a configured model is retired or a provider is slow — via synthesis-model preflight validation, a channel-agnostic webhook alert path, and event-loop/DB resilience.

**Architecture:** Two new self-contained modules (`app/alerting.py`, `app/llm/model_validation.py`) plus a pure decision helper (`app/synthesis/preflight.py`) are wired into the synthesis loop and startup. Separately, the async recall path stops blocking the event loop (offload embedding I/O), `/health` becomes async+I/O-free, and the DB engine gets bounded pooling + a Postgres `statement_timeout`.

**Tech Stack:** Python 3.12, FastAPI/Starlette, SQLAlchemy 2 (psycopg), pydantic-settings, httpx (sync), pytest. Ollama at `http://host.docker.internal:11434`. Spec: `docs/superpowers/specs/2026-07-02-preflight-model-validation-and-alerting-design.md`.

## Execution Model (post-compaction subagent dispatch)

This plan is written to be executed by subagents after this session is compacted. Assign per task:

- **GLM 5.2** (`glm-5.2:cloud`) — implementation worker. Dispatch via the `chain-glm-implement` subagent (or `~/.claude/scripts/chain.sh glm-implement "<brief>"`). GLM follows briefs **literally** — every task below lists exact paths, line anchors, full code, and a **Forbidden (do NOT)** block. Do not let it improvise beyond the brief.
- **GPT 5.5** (`gpt-review`, paid opt-in) — correctness reviewer for the concurrency/hot-path/logic tasks: **Task 3** (skip-vs-degrade logic), **Task 4** (event-loop offload), **Task 6** (DB pool/timeout), and the **Final Review**. GPT-review **flags, does not fix** — route its findings back to GLM.
- Orchestrator (Opus, fresh session): dispatch each task to GLM, run `state-return-check.sh` on the returned `progress.md`, read the diff, then send correctness-critical tasks to GPT 5.5 before accepting.

**Branch (all tasks):** `feat/hive-preflight-alerting-resilience`
**Repo root:** `/opt/agentssot`  ·  **Python package root:** `/opt/agentssot/api`
**Run tests from:** `/opt/agentssot/api` with `python -m pytest <path> -v`

## Global Constraints

- Alerting and preflight must **fail safe**: never raise into or crash the synthesis loop or any request. Skip the run rather than loop; swallow webhook errors.
- New modules `app/alerting.py` and `app/llm/model_validation.py` must **NOT import `app.settings` at module top-level** (only lazily inside the settings-aware wrapper) so they stay unit-testable headless.
- Do **NOT** change any HTTP response schema, auth/role checks, or existing synthesis business logic beyond the wiring described here.
- Match existing structured logging style: `logger.warning("msg", extra={...})`.
- `/api/tags` lists both local and `…:cloud` models — one call validates `qwen3.6:27b` and `qwen3.5:397b-cloud` alike.
- Every unit test file that imports any `app.*` module must start with `import os; os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://ssot:test@localhost:5432/ssot")` before the `app` imports (Settings requires `DATABASE_URL`; `create_engine` is lazy so a dummy URL is safe).
- Commit after each task with the persona-style message convention (no `Co-Authored-By`).

---

## File Structure

- **Create** `api/app/alerting.py` — best-effort webhook alert sender.
- **Create** `api/app/llm/model_validation.py` — live Ollama model-existence checks.
- **Create** `api/app/synthesis/preflight.py` — pure preflight decision (validate + decide).
- **Modify** `api/app/settings.py` — add alert config fields.
- **Modify** `.env` and `.env.example` — add alert config.
- **Modify** `api/app/synthesis/loop.py` — wire preflight + skip/degrade + alert.
- **Modify** `api/app/main.py` — startup preflight log/alert; async `/health`.
- **Modify** `api/app/routers/knowledge.py` — offload blocking embedding calls (R1).
- **Modify** `api/app/db.py` — bounded pool + statement_timeout (R3).
- **Create** tests: `test_alerting.py`, `test_model_validation.py`, `test_synthesis_preflight.py`, `test_health_async.py`, `test_recall_nonblocking.py`, `test_db_pool_config.py` under `api/tests/`.

---

## Task 1: Alerting module + config

**Executor:** GLM 5.2  ·  **Reviewer:** none (standalone, low-risk)

**Files:**
- Create: `api/app/alerting.py`
- Create: `api/tests/test_alerting.py`
- Modify: `api/app/settings.py` (add fields after `synthesis_feedback_protection_days`, near line 68)
- Modify: `.env`, `.env.example` (append alert block)

**Interfaces:**
- Produces: `post_alert(webhook_url: str, event: str, severity: str, message: str, detail: dict|None=None, *, host_label: str="hive", enabled: bool=True) -> bool` and `send_alert(event: str, severity: str, message: str, detail: dict|None=None) -> bool` (settings-aware wrapper). Consumed by Tasks 3 and (startup) 5-wiring.

- [ ] **Step 1: Write the failing test** — `api/tests/test_alerting.py`

```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://ssot:test@localhost:5432/ssot")

from app import alerting


class _Resp:
    def __init__(self, status_code): self.status_code = status_code


def test_post_alert_sends_expected_payload(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _Resp(200)

    monkeypatch.setattr(alerting.httpx, "post", fake_post)
    ok = alerting.post_alert(
        "http://sink.local/hook", "synthesis.model_missing", "error",
        "both models gone", {"missing": ["x"]}, host_label="hari",
    )
    assert ok is True
    assert captured["url"] == "http://sink.local/hook"
    body = captured["json"]
    assert body["source"] == "hive"
    assert body["host"] == "hari"
    assert body["severity"] == "error"
    assert body["event"] == "synthesis.model_missing"
    assert body["detail"] == {"missing": ["x"]}
    assert "timestamp" in body


def test_post_alert_noops_when_disabled_or_no_url(monkeypatch):
    def boom(*a, **k):  # must never be called
        raise AssertionError("httpx.post should not be called")
    monkeypatch.setattr(alerting.httpx, "post", boom)
    assert alerting.post_alert("", "e", "info", "m") is False
    assert alerting.post_alert("http://x", "e", "info", "m", enabled=False) is False


def test_post_alert_swallows_errors(monkeypatch):
    def raiser(*a, **k):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(alerting.httpx, "post", raiser)
    assert alerting.post_alert("http://x", "e", "error", "m") is False  # no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /opt/agentssot/api && python -m pytest tests/test_alerting.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.alerting'`.

- [ ] **Step 3: Create `api/app/alerting.py`**

```python
"""Best-effort outbound alerting — channel-agnostic webhook POST.

Alerting must NEVER raise into or block hive. All failures are logged and
swallowed. Point ALERT_WEBHOOK_URL at ntfy / Discord / Slack / Madi / the
:9877 fleet server — swapping channels never touches this code.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger("agentssot.alerting")

ALERT_TIMEOUT_SECONDS = 5.0


def post_alert(
    webhook_url: str,
    event: str,
    severity: str,
    message: str,
    detail: dict[str, Any] | None = None,
    *,
    host_label: str = "hive",
    enabled: bool = True,
) -> bool:
    """POST a structured alert. Returns True if sent, False if skipped/failed.
    Never raises."""
    if not enabled or not webhook_url:
        logger.debug("alert suppressed (disabled or no url): %s", event)
        return False
    payload = {
        "source": "hive",
        "host": host_label,
        "severity": severity,
        "event": event,
        "message": message,
        "detail": detail or {},
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        resp = httpx.post(webhook_url, json=payload, timeout=ALERT_TIMEOUT_SECONDS)
        if resp.status_code >= 400:
            logger.warning("alert webhook returned %s for %s", resp.status_code, event)
            return False
        return True
    except Exception as exc:  # noqa: BLE001 — alerting must never propagate
        logger.warning("alert webhook failed for %s: %s", event, exc)
        return False


def send_alert(event: str, severity: str, message: str, detail: dict[str, Any] | None = None) -> bool:
    """Settings-aware wrapper used by app code. Imports settings lazily so this
    module stays import-clean for unit tests."""
    from .settings import get_settings

    s = get_settings()
    return post_alert(
        getattr(s, "alert_webhook_url", ""),
        event,
        severity,
        message,
        detail,
        host_label=getattr(s, "alert_host_label", "hive"),
        enabled=getattr(s, "alert_enabled", True),
    )
```

- [ ] **Step 4: Add settings fields** — `api/app/settings.py`, immediately after the line `synthesis_feedback_protection_days: int = Field(default=180, alias="SYNTHESIS_FEEDBACK_PROTECTION_DAYS")` (around line 68), insert:

```python
    # Alerting (channel-agnostic webhook; empty URL = alerting no-ops)
    alert_enabled: bool = Field(default=True, alias="ALERT_ENABLED")
    alert_webhook_url: str = Field(default="", alias="ALERT_WEBHOOK_URL")
    alert_host_label: str = Field(default="hari", alias="ALERT_HOST_LABEL")
```

- [ ] **Step 5: Add `.env` and `.env.example` block** — append to both `/opt/agentssot/.env` and `/opt/agentssot/.env.example`:

```
# Alerting (channel-agnostic webhook; empty = no-op)
ALERT_ENABLED=true
ALERT_WEBHOOK_URL=
ALERT_HOST_LABEL=hari
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /opt/agentssot/api && python -m pytest tests/test_alerting.py -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Commit**

```bash
cd /opt/agentssot && git add api/app/alerting.py api/tests/test_alerting.py api/app/settings.py .env .env.example
git commit -m "feat(alerting): channel-agnostic best-effort webhook alert path

Signed-off-by: Sentry Sam <sam@agentssot>"
```

**Forbidden (do NOT):** import `app.settings` at module top of `alerting.py`; raise from `post_alert`/`send_alert`; add ret/queueing/dedup persistence (out of scope); touch any other settings field.

---

## Task 2: Model validation helper

**Executor:** GLM 5.2  ·  **Reviewer:** none (standalone)

**Files:**
- Create: `api/app/llm/model_validation.py`
- Create: `api/tests/test_model_validation.py`

**Interfaces:**
- Produces: `list_available_models(base_url: str, timeout: float=5.0) -> set[str]`; `validate_models(base_url: str, required: list[str], timeout: float=5.0) -> tuple[set[str], set[str]]` returning `(present, missing)`; exception `ModelListUnavailable(RuntimeError)`. Consumed by Task 3.

- [ ] **Step 1: Write the failing test** — `api/tests/test_model_validation.py`

```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://ssot:test@localhost:5432/ssot")

import pytest
from app.llm import model_validation as mv


class _Resp:
    def __init__(self, payload): self._payload = payload
    def raise_for_status(self): pass
    def json(self): return self._payload


def _fake_get(payload):
    def _get(url, timeout):
        assert url.endswith("/api/tags")
        return _Resp(payload)
    return _get


def test_validate_models_present_and_missing(monkeypatch):
    payload = {"models": [{"name": "qwen3.6:27b"}, {"name": "qwen3.5:397b-cloud"}]}
    monkeypatch.setattr(mv.httpx, "get", _fake_get(payload))
    present, missing = mv.validate_models("http://ollama:11434", ["qwen3.6:27b", "qwen3.5:27b"])
    assert present == {"qwen3.6:27b"}
    assert missing == {"qwen3.5:27b"}


def test_validate_models_all_present(monkeypatch):
    payload = {"models": [{"name": "a"}, {"name": "b"}]}
    monkeypatch.setattr(mv.httpx, "get", _fake_get(payload))
    present, missing = mv.validate_models("http://ollama:11434", ["a", "b"])
    assert missing == set()


def test_list_unavailable_raises(monkeypatch):
    def raiser(url, timeout):
        raise RuntimeError("conn refused")
    monkeypatch.setattr(mv.httpx, "get", raiser)
    with pytest.raises(mv.ModelListUnavailable):
        mv.list_available_models("http://ollama:11434")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /opt/agentssot/api && python -m pytest tests/test_model_validation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.llm.model_validation'`.

- [ ] **Step 3: Create `api/app/llm/model_validation.py`**

```python
"""Live model-existence checks against an Ollama /api/tags endpoint.

Single source of truth for "does this model exist". /api/tags lists both local
and cloud (…:cloud) models, so one call covers qwen3.6:27b and
qwen3.5:397b-cloud alike.
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("agentssot.llm.model_validation")

TAGS_TIMEOUT_SECONDS = 5.0


class ModelListUnavailable(RuntimeError):
    """Raised when the Ollama model list cannot be fetched."""


def list_available_models(base_url: str, timeout: float = TAGS_TIMEOUT_SECONDS) -> set[str]:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        resp = httpx.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise ModelListUnavailable(f"cannot fetch {url}: {exc}") from exc
    data = resp.json()
    return {m.get("name") for m in data.get("models", []) if m.get("name")}


def validate_models(
    base_url: str, required: list[str], timeout: float = TAGS_TIMEOUT_SECONDS
) -> tuple[set[str], set[str]]:
    """Return (present, missing) for the required model names."""
    available = list_available_models(base_url, timeout=timeout)
    req = {r for r in required if r}
    return (req & available, req - available)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /opt/agentssot/api && python -m pytest tests/test_model_validation.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /opt/agentssot && git add api/app/llm/model_validation.py api/tests/test_model_validation.py
git commit -m "feat(llm): live Ollama model-existence validation helper

Signed-off-by: Checkpoint Charlie <charlie@agentssot>"
```

**Forbidden (do NOT):** import settings; cache results; ping models via /api/generate (use /api/tags only); swallow the fetch error (must raise ModelListUnavailable so the caller can decide to skip).

---

## Task 3: Synthesis preflight — decision helper + loop/startup wiring

**Executor:** GLM 5.2  ·  **Reviewer:** GPT 5.5 (skip-vs-degrade correctness)

**Files:**
- Create: `api/app/synthesis/preflight.py`
- Create: `api/tests/test_synthesis_preflight.py`
- Modify: `api/app/synthesis/loop.py` (in `_run_synthesis_for_namespace`, right after `overrides = load_overrides(session)` ~line 165; and the `run_synthesis_batch(...)` call args ~lines 231-232)
- Modify: `api/app/main.py` (synthesis-loop startup block ~lines 159-163)

**Interfaces:**
- Consumes: `validate_models`, `ModelListUnavailable` (Task 2); `send_alert` (Task 1); `effective` (existing `app.runtime_config`).
- Produces: `PreflightResult` dataclass `(proceed: bool, primary: str|None, fallback: str|None, severity: str|None, event: str|None, message: str|None, detail: dict)` and `evaluate(base_url: str, primary: str, fallback: str) -> PreflightResult`.

- [ ] **Step 1: Write the failing test** — `api/tests/test_synthesis_preflight.py`

```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://ssot:test@localhost:5432/ssot")

import pytest
from app.synthesis import preflight
from app.llm.model_validation import ModelListUnavailable


def _patch(monkeypatch, present, missing):
    monkeypatch.setattr(preflight, "validate_models", lambda base, req, timeout=5.0: (set(present), set(missing)))


def test_all_present_proceeds_clean(monkeypatch):
    _patch(monkeypatch, ["p", "f"], [])
    r = preflight.evaluate("http://o", "p", "f")
    assert r.proceed and r.primary == "p" and r.fallback == "f" and r.severity is None


def test_primary_missing_degrades_to_fallback(monkeypatch):
    _patch(monkeypatch, ["f"], ["p"])
    r = preflight.evaluate("http://o", "p", "f")
    assert r.proceed and r.primary == "f" and r.fallback is None and r.severity == "warning"


def test_fallback_missing_runs_without_fallback(monkeypatch):
    _patch(monkeypatch, ["p"], ["f"])
    r = preflight.evaluate("http://o", "p", "f")
    assert r.proceed and r.primary == "p" and r.fallback is None and r.severity == "warning"


def test_both_missing_skips(monkeypatch):
    _patch(monkeypatch, [], ["p", "f"])
    r = preflight.evaluate("http://o", "p", "f")
    assert not r.proceed and r.severity == "error" and r.event == "synthesis.model_missing"


def test_unreachable_skips(monkeypatch):
    def raiser(base, req, timeout=5.0):
        raise ModelListUnavailable("down")
    monkeypatch.setattr(preflight, "validate_models", raiser)
    r = preflight.evaluate("http://o", "p", "f")
    assert not r.proceed and r.event == "synthesis.unreachable" and r.severity == "error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /opt/agentssot/api && python -m pytest tests/test_synthesis_preflight.py -v`
Expected: FAIL — `ImportError: cannot import name 'preflight'`.

- [ ] **Step 3: Create `api/app/synthesis/preflight.py`**

```python
"""Pure preflight decision for a synthesis run: validate the configured models
against live Ollama and decide proceed / degrade / skip. No DB, no side effects
(alerting + skipping happen in the caller based on the returned result)."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.model_validation import ModelListUnavailable, validate_models


@dataclass
class PreflightResult:
    proceed: bool
    primary: str | None
    fallback: str | None
    severity: str | None            # None = clean; else "warning" | "error"
    event: str | None               # alert event name, or None
    message: str | None
    detail: dict = field(default_factory=dict)


def evaluate(base_url: str, primary: str, fallback: str) -> PreflightResult:
    try:
        _present, missing = validate_models(base_url, [primary, fallback])
    except ModelListUnavailable as exc:
        return PreflightResult(
            False, None, None, "error", "synthesis.unreachable",
            f"Synthesis skipped: Ollama model list unreachable ({exc})",
            {"base_url": base_url},
        )
    primary_ok = primary not in missing
    fallback_ok = fallback not in missing
    if not primary_ok and not fallback_ok:
        return PreflightResult(
            False, None, None, "error", "synthesis.model_missing",
            f"Synthesis skipped: both models missing from Ollama: {sorted(missing)}",
            {"missing": sorted(missing), "primary": primary, "fallback": fallback},
        )
    if not primary_ok and fallback_ok:
        return PreflightResult(
            True, fallback, None, "warning", "synthesis.model_missing",
            f"Synthesis primary {primary} missing; running on fallback {fallback}",
            {"missing": [primary]},
        )
    if primary_ok and not fallback_ok:
        return PreflightResult(
            True, primary, None, "warning", "synthesis.model_missing",
            f"Synthesis fallback {fallback} missing; running with no fallback",
            {"missing": [fallback]},
        )
    return PreflightResult(True, primary, fallback, None, None, None, {})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /opt/agentssot/api && python -m pytest tests/test_synthesis_preflight.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Wire preflight into the loop** — `api/app/synthesis/loop.py`. Immediately after the line `overrides = load_overrides(session)` (~line 165), insert:

```python
        # --- Preflight: validate synthesis models before doing any work ---
        primary = effective(settings, overrides, "synthesis_model")
        fallback = effective(settings, overrides, "synthesis_fallback_model")
        from .preflight import evaluate as _preflight_eval
        from ..alerting import send_alert

        pf = _preflight_eval(settings.ollama_base_url, primary, fallback)
        if pf.severity:
            send_alert(pf.event, pf.severity, pf.message, {"namespace": namespace, **pf.detail})
        if not pf.proceed:
            logger.error(
                "synthesis preflight failed; skipping run",
                extra={"namespace": namespace, "event": pf.event},
            )
            stats["skipped"] = pf.event
            return stats
        primary, fallback = pf.primary, pf.fallback
```

- [ ] **Step 6: Use the validated models at the batch call** — in the same file, the `run_synthesis_batch(...)` call (~lines 231-232) currently reads:

```python
                    synthesis_model=effective(settings, overrides, "synthesis_model"),
                    fallback_model=effective(settings, overrides, "synthesis_fallback_model"),
```

Replace those two lines with:

```python
                    synthesis_model=primary,
                    fallback_model=fallback,
```

- [ ] **Step 7: Startup preflight (log + alert)** — `api/app/main.py`, inside the `if settings.effective_synthesis_enabled:` block (~lines 160-163), after the `logger.info("background synthesis loop started ...")` line, insert:

```python
        try:
            from .synthesis.preflight import evaluate as _preflight_eval
            from .alerting import send_alert as _send_alert

            _pf = _preflight_eval(
                settings.ollama_base_url, settings.synthesis_model, settings.synthesis_fallback_model
            )
            if _pf.severity:
                _send_alert(_pf.event, _pf.severity, f"[startup] {_pf.message}", _pf.detail)
                logger.warning("startup synthesis preflight: %s", _pf.message)
            else:
                logger.info("startup synthesis preflight OK")
        except Exception:
            logger.exception("startup synthesis preflight check errored (non-fatal)")
```

- [ ] **Step 8: Run the full synthesis + preflight test set**

Run: `cd /opt/agentssot/api && python -m pytest tests/test_synthesis_preflight.py tests/synthesis -v`
Expected: PASS (preflight 5 passed; existing synthesis tests unchanged/green).

- [ ] **Step 9: Commit**

```bash
cd /opt/agentssot && git add api/app/synthesis/preflight.py api/tests/test_synthesis_preflight.py api/app/synthesis/loop.py api/app/main.py
git commit -m "feat(synthesis): preflight-validate models; skip/degrade + alert instead of 404-looping

Signed-off-by: Preflight Petra <petra@agentssot>"
```

**Forbidden (do NOT):** remove or alter the per-cluster savepoint/try-except in the loop; change the synthesis prompt or clustering; call `send_alert` more than once per run in the loop path; make `evaluate()` perform alerting or DB access (it stays pure); change `run_synthesis_batch`'s signature.

**GPT 5.5 review focus:** Confirm the decision table matches the spec exactly (primary-missing→promote fallback to primary AND clear fallback; fallback-missing→keep primary, clear fallback; both-missing/unreachable→skip). Confirm `stats["skipped"]` early-return can't corrupt later loop assumptions. Confirm no double-alert and no path where a missing model still reaches `run_synthesis_batch`.

---

## Task 4: R1 — stop blocking the event loop in async recall/ingest

**Executor:** GLM 5.2  ·  **Reviewer:** GPT 5.5 (concurrency correctness)

**Files:**
- Modify: `api/app/routers/knowledge.py` (embedding calls at lines 468, 560, and 147 — all inside `async def` handlers)
- Create: `api/tests/test_recall_nonblocking.py`

**Context:** `recall_tiered` (async, line 449), `_recall_bucketed` (async, line 540), and `ingest_tiered` (async, line 132) call the **synchronous** `embedding_provider.embed_text(...)` directly on the event loop. `asyncio` is already imported (line 1); `asyncio.to_thread` is already used at line 160. Offload the embedding call so a slow Ollama can't freeze the loop.

- [ ] **Step 1: Write the failing (guard) test** — `api/tests/test_recall_nonblocking.py`

```python
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "app" / "routers" / "knowledge.py"


def test_embed_text_calls_are_offloaded_to_thread():
    """Every embed_text call in async handlers must be wrapped in
    asyncio.to_thread so it cannot block the event loop."""
    text = SRC.read_text()
    # No bare synchronous embed_text( that is not preceded by to_thread on the same line.
    offenders = []
    for i, line in enumerate(text.splitlines(), 1):
        if "embed_text(" in line and "to_thread" not in line and not line.lstrip().startswith("#"):
            offenders.append((i, line.strip()))
    assert not offenders, f"Un-offloaded embed_text calls: {offenders}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /opt/agentssot/api && python -m pytest tests/test_recall_nonblocking.py -v`
Expected: FAIL — lists the 3 bare `embed_text(` calls (lines 147, 468, 560).

- [ ] **Step 3: Offload the three embedding calls** — `api/app/routers/knowledge.py`.

At line ~468 (in `recall_tiered`), replace:
```python
        query_embedding = embedding_provider.embed_text(data.query)
```
with:
```python
        query_embedding = await asyncio.to_thread(embedding_provider.embed_text, data.query)
```

At line ~560 (in `_recall_bucketed`), replace the identical line:
```python
        query_embedding = embedding_provider.embed_text(data.query)
```
with:
```python
        query_embedding = await asyncio.to_thread(embedding_provider.embed_text, data.query)
```

At line ~147 (in `ingest_tiered`), replace:
```python
            embedding = embedding_provider.embed_text(data.content)
```
with:
```python
            embedding = await asyncio.to_thread(embedding_provider.embed_text, data.content)
```

- [ ] **Step 4: Run the guard test to verify it passes**

Run: `cd /opt/agentssot/api && python -m pytest tests/test_recall_nonblocking.py -v`
Expected: PASS.

- [ ] **Step 5: Sanity — import compiles and recall tests unaffected**

Run: `cd /opt/agentssot/api && python -c "import ast,sys; ast.parse(open('app/routers/knowledge.py').read()); print('ok')"`
Expected: `ok`.
Run (if a live stack is available): `cd /opt/agentssot/api && python -m pytest tests/test_recall_bucketed.py tests/test_recall_compat.py -v` — Expected: PASS or skipped (integration). If skipped due to no live API, note it in progress.md — do not treat as failure.

- [ ] **Step 6: Commit**

```bash
cd /opt/agentssot && git add api/app/routers/knowledge.py api/tests/test_recall_nonblocking.py
git commit -m "fix(api): offload embedding I/O off the event loop in async recall/ingest

Signed-off-by: Loopkeeper Lena <lena@agentssot>"
```

**Forbidden (do NOT):** wrap `session.execute(...)` calls in this task (DB offload is a separate consideration — leave the ORM session single-threaded); change recall/ingest response shape, ranking, or filters; touch the `_recall_bucketed` reranking logic; convert the endpoints to sync `def`.

**GPT 5.5 review focus:** Confirm the SQLAlchemy `session` is never used concurrently across threads (calls remain sequentially awaited — no `asyncio.gather` introduced). Confirm `to_thread(embedding_provider.embed_text, arg)` passes the arg correctly (positional). Flag any *other* `async def` in the routers that still performs sync httpx/DB I/O inline that this task missed (grep `async def` blocks for `session.execute`, `httpx.`, `.embed_text`, `classify(` without `to_thread`).

---

## Task 5: R2 — async, I/O-free `/health`

**Executor:** GLM 5.2 (glm-quick)  ·  **Reviewer:** none

**Files:**
- Modify: `api/app/main.py` (lines 429-441)
- Create: `api/tests/test_health_async.py`

- [ ] **Step 1: Write the failing (guard) test** — `api/tests/test_health_async.py`

```python
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_health_endpoint_is_async_and_dbless():
    text = SRC.read_text()
    m = re.search(r"@app\.get\(\"/health\"\)\s*\n\s*(async def|def) health\(", text)
    assert m, "could not locate /health handler"
    assert m.group(1) == "async def", "/health must be async so it never needs a threadpool token"
    # The health function body must not open a DB session.
    body = text.split('def health(', 1)[1].split('\ndef ', 1)[0].split('\n@app', 1)[0]
    assert "get_session" not in body and "SessionLocal" not in body, "/health must not touch the DB"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /opt/agentssot/api && python -m pytest tests/test_health_async.py -v`
Expected: FAIL — group(1) is `def`, not `async def`.

- [ ] **Step 3: Make `/health` async** — `api/app/main.py` line 430, change:

```python
def health() -> dict:
```
to:
```python
async def health() -> dict:
```
(Leave the body unchanged — it only reads cached `app.state.*.is_available` booleans and settings properties.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /opt/agentssot/api && python -m pytest tests/test_health_async.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /opt/agentssot && git add api/app/main.py api/tests/test_health_async.py
git commit -m "fix(api): make /health async + dbless so the probe stays honest under load

Signed-off-by: Nurse Nyx <nyx@agentssot>"
```

**Forbidden (do NOT):** add any `await`/DB/provider ping inside `/health` (that would reintroduce blocking); change the returned keys.

---

## Task 6: R3 — bound the DB pool + statement_timeout

**Executor:** GLM 5.2  ·  **Reviewer:** GPT 5.5 (pool/timeout correctness)

**Files:**
- Modify: `api/app/db.py` (lines 10-13)
- Create: `api/tests/test_db_pool_config.py`

- [ ] **Step 1: Write the failing test** — `api/tests/test_db_pool_config.py`

```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://ssot:test@localhost:5432/ssot")

from app import db


def test_engine_pool_is_bounded():
    eng = db.engine
    assert eng.pool.size() == 10
    # max_overflow is stored on the pool as _max_overflow
    assert eng.pool._max_overflow == 20
    assert eng.pool._timeout == 10


def test_engine_sets_statement_timeout():
    # connect_args options string carries the Postgres timeouts
    opts = db.engine.url  # sanity: url parsed
    assert opts is not None
    # The connect_args live on the engine dialect; assert our marker is present.
    assert "statement_timeout" in db._CONNECT_OPTIONS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /opt/agentssot/api && python -m pytest tests/test_db_pool_config.py -v`
Expected: FAIL — pool size is default (5), and `db._CONNECT_OPTIONS` does not exist.

- [ ] **Step 3: Update the engine** — `api/app/db.py`, replace the block:

```python
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
)
```
with:

```python
# Postgres server-side guards: no query pins a connection forever, and no
# transaction can idle indefinitely (protects the pool from a stuck synthesis
# session or a slow recall query taking the whole API down).
_CONNECT_OPTIONS = "-c statement_timeout=30000 -c idle_in_transaction_session_timeout=60000"

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_timeout=10,
    connect_args={"options": _CONNECT_OPTIONS},
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /opt/agentssot/api && python -m pytest tests/test_db_pool_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /opt/agentssot && git add api/app/db.py api/tests/test_db_pool_config.py
git commit -m "fix(db): bound the connection pool + add statement_timeout guards

Signed-off-by: Poolboy Pat <pat@agentssot>"
```

**Forbidden (do NOT):** change `settings.database_url`; alter `SessionLocal` kwargs; set `statement_timeout` so low it breaks the synthesis writes (30s statement / 60s idle-in-txn are the agreed values); use `NullPool`.

**GPT 5.5 review focus:** Confirm `connect_args={"options": ...}` is the correct psycopg (v3) form for the configured driver (`postgresql+psycopg://`). Confirm 30s `statement_timeout` won't abort legitimate long synthesis writes (they run in the batch thread as many small statements, not one long query — verify). Confirm `pool_size=10 + max_overflow=20` is compatible with Postgres `max_connections` on the `stocktrader`/hive Postgres container.

---

## Task 7: Final correctness review (GPT 5.5) + full suite

**Executor:** GPT 5.5 (`gpt-review`)  ·  then GLM 5.2 for any fixes

- [ ] **Step 1:** Provide GPT 5.5 the full branch diff (`git diff master...feat/hive-preflight-alerting-resilience`) plus this plan and the spec. Ask it to flag (not fix): any remaining event-loop-blocking path; any way preflight lets a missing model reach an Ollama call; alerting that could raise; pool/timeout regressions. It must output a findings list with file:line + severity.
- [ ] **Step 2:** Route any CONFIRMED findings back to GLM 5.2 as a follow-up brief; re-review until clean.
- [ ] **Step 3:** Run the new unit suite:

Run: `cd /opt/agentssot/api && python -m pytest tests/test_alerting.py tests/test_model_validation.py tests/test_synthesis_preflight.py tests/test_recall_nonblocking.py tests/test_health_async.py tests/test_db_pool_config.py -v`
Expected: all PASS.

- [ ] **Step 4:** Rebuild + restart the container and verify live:

```bash
cd /opt/agentssot && docker compose up -d --build api
sleep 5
curl -s -m 5 -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8088/health   # expect 200
docker inspect agentssot-api --format '{{.State.Health.Status}} streak={{.State.Health.FailingStreak}}'
```
Expected: `200`, `healthy streak=0`.

- [ ] **Step 5:** Land per repo `AGENTS.md` (`/land`), or open a PR titled `feat: hive preflight validation + webhook alerting + API resilience`.

---

## Self-Review (author checklist — completed)

- **Spec coverage:** §1 model validation → Task 2; §2 preflight decision table → Task 3; §3 alert path → Task 1; §4 alert triggers → Tasks 1+3; §5 config → Task 1; §6 startup preflight → Task 3 step 7; §7 R1/R2/R3 → Tasks 4/5/6; testing → each task + Task 7. All covered.
- **Placeholder scan:** no TBD/TODO; all code and commands concrete.
- **Type consistency:** `validate_models` returns `(present, missing)` used consistently in Tasks 2/3; `PreflightResult` fields match between definition, test, and loop wiring; `send_alert(event, severity, message, detail)` signature consistent across Tasks 1/3.

## Open Item (operator, non-blocking)
`ALERT_WEBHOOK_URL` ships empty (alerting no-ops until set). Set it to the chosen channel (ntfy / Discord-Slack / Madi / `:9877` fleet server) in `/opt/agentssot/.env` and `docker compose up -d api` to activate. No code change needed to switch channels.
