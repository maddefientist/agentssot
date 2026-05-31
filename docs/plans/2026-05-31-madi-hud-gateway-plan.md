# Madi HUD + Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standing, always-on Madi presence — a model-agnostic gateway (nervous system) plus an interactive HUD (first body) — as a new hero surface inside the existing Cortex FastAPI app.

**Architecture:** A deterministic `madi-gateway` module in `api/app` accepts HUD WebSocket connections, classifies intent via a hybrid router (rules + cheap local model), and dispatches to swappable executors behind a stable streaming interface. Conversational state lives in hive (restart-safe). An `orchestrate` executor applies an Opus→deepseek→local fallback ladder. The HUD is a single morphing surface (ambient ⇄ active) in the "Obsidian Terminal" identity, fed by SSE status endpoints.

**Tech Stack:** Python / FastAPI (existing `api/app`), WebSocket + SSE, hive (agentSSOT memory), Ollama (local chat + intent classification), chain.sh / hari-core (dispatch), vanilla JS + the app's existing shared-asset pattern for the HUD.

**Design spec:** `docs/plans/2026-05-31-madi-hud-gateway-design.md`

---

## Pre-flight (do once, before Phase 1)

- Another agent may have modified `/opt/agentssot` since the design was written. Compaction has occurred. Re-read the design spec above before starting.
- This plan assumes Python/pytest and FastAPI per the existing `api/app`. If Phase 0 recon contradicts any assumption here, update the plan before proceeding.

---

## File Structure

**New (greenfield — fully specified in this plan):**
- `api/app/gateway/__init__.py` — package marker
- `api/app/gateway/protocol.py` — wire types (in/out event dataclasses), executor `Event` type
- `api/app/gateway/router.py` — hybrid intent router (rules table + local-model fallback)
- `api/app/gateway/executors/base.py` — `Executor` protocol: `execute(intent, ctx) -> AsyncIterator[Event]`
- `api/app/gateway/executors/hive_tool.py` — deterministic hive recall/teach/status
- `api/app/gateway/executors/chat_local.py` — local Ollama chat
- `api/app/gateway/executors/orchestrate.py` — Opus→deepseek→local fallback ladder
- `api/app/gateway/executors/dispatch.py` — hari-core / chain.sh jobs
- `api/app/gateway/executors/__init__.py` — registry mapping intent class → executor instance
- `api/app/gateway/session.py` — hive-backed session state (load/append/save)
- `api/app/gateway/service.py` — gateway orchestration: connect → route → dispatch → stream
- `api/app/gateway/routes.py` — FastAPI router: `WS /gateway/ws`, `GET /gateway/sse/status`
- `api/app/gateway/feeders.py` — status feeders (hive activity, executor health, fleet, chains)
- `api/app/gateway/config.py` — fallback ladder + model names (config, not code)
- `api/app/ui/hud.html` — the HUD surface (ambient ⇄ active)
- `api/app/ui/assets/hud.css` — Obsidian Terminal styles (locked tokens)
- `api/app/ui/assets/hud.js` — WS command channel + SSE panels + mode morph
- `tests/gateway/` — mirrors the gateway package

**Modified (touchpoints pinned in Phase 0 recon):**
- the app's route-registration site (wire in `gateway/routes.py`)
- `api/app/ui/_nav.html` (add HUD entry)
- repo-root cleanup targets per the ledger

---

## Phase 0 — Recon & Consolidation Audit

### Task 0.1: Map the app's integration points

**Files:** none created — produces notes appended to the plan.

- [ ] **Step 1:** Read the FastAPI app entry + route registration.
  Run: `sed -n '1,80p' api/app/main.py 2>/dev/null || ls api/app`
  Capture: how routers are included (the `app.include_router(...)` site), the app object location, and the auth dependency (if any) used by existing routes.

- [ ] **Step 2:** Learn the hive client interface available in-process.
  Run: `grep -rn "def recall\|def teach\|def status\|class .*Hive\|hive" api/app --include=*.py | grep -iv test | head -40`
  Capture: the import path + method signatures for recall/teach/status used by existing UI endpoints. The gateway reuses this, not the MCP layer.

- [ ] **Step 3:** Learn the test pattern.
  Run: `ls tests && sed -n '1,40p' $(find tests -name 'conftest.py' | head -1)`
  Capture: fixture names for the app/client and any hive test double.

- [ ] **Step 4:** Learn the UI serving + shared-asset pattern.
  Run: `sed -n '1,60p' api/app/ui/_nav.html && grep -rn "ui/\|templates\|StaticFiles\|FileResponse\|version\|mtime" api/app --include=*.py | head -20`
  Capture: how `*.html` pages are served and how the mtime cache-bust stamp is applied (from commit `dbb4ed6`).

- [ ] **Step 5:** Confirm local model + chain availability on hari.
  Run: `curl -s http://192.168.1.225:11434/api/tags | jq '.models[].name' | grep -E 'qwen3.5:4b|qwen3.5:27b'` and `ls ~/.claude/chains/*.json`
  Capture: the exact local model tag for chat/classification, and the chain names for `dispatch`/`orchestrate`.

- [ ] **Step 6:** Append a short "Integration Facts" section to this plan with the captured paths/signatures. **All later tasks that touch existing files use these facts.**

### Task 0.2: Consolidation ledger audit (verify before any deletion)

**Files:** none — produces verdicts appended to the plan.

- [ ] **Step 1:** Verify repo-root orphans are dead.
  Run: `cd /opt/agentssot && git log --oneline -3 -- firebase-debug.log after dropbox; grep -rn "after/\|dropbox/" --include=*.py api scripts | head`
  Expected: no live references → mark RETIRE-confirmed. If referenced, downgrade to KEEP and note why.

- [ ] **Step 2:** Locate harivoice / OmniVoice / HariScribe / air Telegram / flightdeck on the fleet and record paths + whether running. (Audit only — no changes; these are ABSORB-later/KEEP per the ledger.)
  Run: `ssh hari "ls ~/hariscribe ~/omnivoice 2>/dev/null; systemctl --type=service 2>/dev/null | grep -iE 'voice|scribe|flight|telegram'"`

- [ ] **Step 2 done:** Record verdicts. Deletions for RETIRE-confirmed items happen in Phase 5 (after the HUD works), never before.

---

## Phase 1 — Gateway core (nervous system)

### Task 1.1: Wire protocol types

**Files:**
- Create: `api/app/gateway/__init__.py` (empty)
- Create: `api/app/gateway/protocol.py`
- Test: `tests/gateway/test_protocol.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/gateway/test_protocol.py
from api.app.gateway.protocol import InboundMessage, Event

def test_inbound_parses_minimal():
    msg = InboundMessage.from_dict({"text": "scan fleet", "session_id": "s1"})
    assert msg.text == "scan fleet"
    assert msg.session_id == "s1"
    assert msg.intent is None

def test_event_token_serializes():
    e = Event.token("hello")
    assert e.to_dict() == {"type": "token", "data": "hello"}

def test_event_error_serializes():
    e = Event.error("boom", retryable=True)
    d = e.to_dict()
    assert d["type"] == "error" and d["data"]["retryable"] is True
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_protocol.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write minimal implementation**
```python
# api/app/gateway/protocol.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class InboundMessage:
    text: str
    session_id: str
    intent: Optional[str] = None  # optional explicit intent override

    @classmethod
    def from_dict(cls, d: dict) -> "InboundMessage":
        return cls(text=d["text"], session_id=d["session_id"], intent=d.get("intent"))

@dataclass
class Event:
    type: str            # token | event | error | done
    data: Any = None

    @classmethod
    def token(cls, s: str) -> "Event": return cls("token", s)
    @classmethod
    def event(cls, payload: dict) -> "Event": return cls("event", payload)
    @classmethod
    def error(cls, msg: str, retryable: bool = False) -> "Event":
        return cls("error", {"message": msg, "retryable": retryable})
    @classmethod
    def done(cls, meta: Optional[dict] = None) -> "Event":
        return cls("done", meta or {})

    def to_dict(self) -> dict:
        return {"type": self.type, "data": self.data}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_protocol.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**
```bash
cd /opt/agentssot && git add api/app/gateway/__init__.py api/app/gateway/protocol.py tests/gateway/test_protocol.py
git commit -m "feat(gateway): wire protocol types"
```

### Task 1.2: Executor interface

**Files:**
- Create: `api/app/gateway/executors/__init__.py` (empty for now)
- Create: `api/app/gateway/executors/base.py`
- Test: `tests/gateway/test_base.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/gateway/test_base.py
import pytest
from api.app.gateway.executors.base import Executor
from api.app.gateway.protocol import Event

class Echo(Executor):
    name = "echo"
    async def execute(self, intent, ctx):
        yield Event.token(ctx["text"])
        yield Event.done()

@pytest.mark.asyncio
async def test_executor_streams():
    out = [e async for e in Echo().execute("chat", {"text": "hi"})]
    assert out[0].to_dict() == {"type": "token", "data": "hi"}
    assert out[-1].type == "done"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_base.py -v`
Expected: FAIL (module not found). (If `pytest-asyncio` missing, add to `api/requirements.txt` and `pip install`.)

- [ ] **Step 3: Write minimal implementation**
```python
# api/app/gateway/executors/base.py
from __future__ import annotations
from typing import AsyncIterator
from abc import ABC, abstractmethod
from api.app.gateway.protocol import Event

class Executor(ABC):
    name: str = "base"

    @abstractmethod
    async def execute(self, intent: str, ctx: dict) -> AsyncIterator[Event]:
        """Yield Events. Must yield a terminal Event.done() or Event.error()."""
        raise NotImplementedError
        yield  # pragma: no cover  (marks this an async generator)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_base.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
cd /opt/agentssot && git add api/app/gateway/executors/__init__.py api/app/gateway/executors/base.py tests/gateway/test_base.py
git commit -m "feat(gateway): executor streaming interface"
```

### Task 1.3: Hybrid intent router — rules layer

**Files:**
- Create: `api/app/gateway/router.py`
- Test: `tests/gateway/test_router.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/gateway/test_router.py
import pytest
from api.app.gateway.router import IntentRouter

def test_rules_match_known_patterns():
    r = IntentRouter(classifier=None)  # rules-only
    assert r.classify("recall greece property")[0] == "hive-tool"
    assert r.classify("scan fleet")[0] == "dispatch"
    assert r.classify("/build a new endpoint")[0] == "dispatch"
    assert r.classify("show briefing")[0] == "briefing"

def test_explicit_intent_override_wins():
    r = IntentRouter(classifier=None)
    assert r.classify("anything", explicit="chat-local")[0] == "chat-local"

def test_unmatched_falls_through_to_none_without_classifier():
    r = IntentRouter(classifier=None)
    assert r.classify("ponder the nature of time")[0] is None
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_router.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write minimal implementation**
```python
# api/app/gateway/router.py
from __future__ import annotations
import re
from typing import Optional, Callable, Tuple

# (regex, intent-class). First match wins. Deterministic, no model.
RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*/?(recall|remember|search)\b", re.I), "hive-tool"),
    (re.compile(r"^\s*/?(teach|note|save)\b", re.I), "hive-tool"),
    (re.compile(r"\b(status|health|stats)\b", re.I), "hive-tool"),
    (re.compile(r"^\s*/?(scan|build|deploy|chain|dispatch|run)\b", re.I), "dispatch"),
    (re.compile(r"\b(scan|status of the)\s+fleet\b", re.I), "dispatch"),
    (re.compile(r"\b(briefing|report|newspaper)\b", re.I), "briefing"),
]

class IntentRouter:
    def __init__(self, classifier: Optional[Callable[[str], Tuple[str, dict]]]):
        # classifier: text -> (intent_class, args). Used only when rules miss.
        self._classifier = classifier

    def classify(self, text: str, explicit: Optional[str] = None) -> Tuple[Optional[str], dict]:
        if explicit:
            return explicit, {}
        for pat, intent in RULES:
            if pat.search(text):
                return intent, {}
        if self._classifier is not None:
            return self._classifier(text)
        return None, {}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_router.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**
```bash
cd /opt/agentssot && git add api/app/gateway/router.py tests/gateway/test_router.py
git commit -m "feat(gateway): hybrid intent router rules layer"
```

### Task 1.4: Router — local-model classifier fallback

**Files:**
- Modify: `api/app/gateway/router.py` (add `make_ollama_classifier`)
- Test: `tests/gateway/test_router_classifier.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/gateway/test_router_classifier.py
from api.app.gateway.router import IntentRouter, parse_classifier_response

def test_parse_classifier_response_extracts_intent_and_args():
    raw = '{"intent": "chat-local", "args": {"topic": "time"}}'
    intent, args = parse_classifier_response(raw)
    assert intent == "chat-local" and args == {"topic": "time"}

def test_parse_classifier_defaults_to_chat_local_on_garbage():
    intent, args = parse_classifier_response("not json")
    assert intent == "chat-local" and args == {}

def test_router_uses_classifier_when_rules_miss():
    r = IntentRouter(classifier=lambda t: ("chat-local", {"x": 1}))
    assert r.classify("ponder time") == ("chat-local", {"x": 1})
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_router_classifier.py -v`
Expected: FAIL (`parse_classifier_response` undefined).

- [ ] **Step 3: Write minimal implementation** (append to `router.py`)
```python
# --- append to api/app/gateway/router.py ---
import json, httpx

_VALID = {"chat-local", "hive-tool", "orchestrate", "dispatch", "briefing"}

def parse_classifier_response(raw: str) -> tuple[str, dict]:
    try:
        obj = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        intent = obj.get("intent")
        if intent in _VALID:
            return intent, obj.get("args", {}) or {}
    except Exception:
        pass
    return "chat-local", {}  # safe default: just talk

def make_ollama_classifier(base_url: str, model: str):
    prompt = (
        "Classify the user message into exactly one intent and extract args. "
        "Intents: chat-local (smalltalk/questions), hive-tool (memory recall/teach/status), "
        "orchestrate (multi-step reasoning/code), dispatch (run a job/scan/build/chain), "
        "briefing (show the daily report). "
        'Respond ONLY as JSON: {"intent": "...", "args": {}}. Message: '
    )
    def classify(text: str) -> tuple[str, dict]:
        try:
            resp = httpx.post(f"{base_url}/api/generate",
                              json={"model": model, "prompt": prompt + text, "stream": False},
                              timeout=8.0)
            return parse_classifier_response(resp.json().get("response", ""))
        except Exception:
            return "chat-local", {}
    return classify
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_router_classifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
cd /opt/agentssot && git add api/app/gateway/router.py tests/gateway/test_router_classifier.py
git commit -m "feat(gateway): ollama intent classifier fallback"
```

### Task 1.5: Hive-backed session state

**Files:**
- Create: `api/app/gateway/session.py`
- Test: `tests/gateway/test_session.py`

> Uses the hive client interface captured in Task 0.1. Below, `HiveClient` is a thin port; in execution, bind it to the real client from Integration Facts.

- [ ] **Step 1: Write the failing test**
```python
# tests/gateway/test_session.py
import pytest
from api.app.gateway.session import SessionStore

class FakeHive:
    def __init__(self): self.kv = {}
    def get(self, ns, key): return self.kv.get((ns, key))
    def put(self, ns, key, val): self.kv[(ns, key)] = val

@pytest.mark.asyncio
async def test_session_roundtrip_via_hive():
    s = SessionStore(FakeHive(), namespace="madi-session")
    await s.append("s1", {"role": "user", "text": "hi"})
    await s.append("s1", {"role": "madi", "text": "hello"})
    turns = await s.history("s1")
    assert [t["role"] for t in turns] == ["user", "madi"]

@pytest.mark.asyncio
async def test_empty_session_history_is_list():
    s = SessionStore(FakeHive(), namespace="madi-session")
    assert await s.history("nope") == []
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_session.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write minimal implementation**
```python
# api/app/gateway/session.py
from __future__ import annotations
import json
from typing import Any

class SessionStore:
    """Conversation state lives in hive — process restart loses nothing."""
    def __init__(self, hive: Any, namespace: str = "madi-session", max_turns: int = 40):
        self._hive = hive
        self._ns = namespace
        self._max = max_turns

    async def history(self, session_id: str) -> list[dict]:
        raw = self._hive.get(self._ns, session_id)
        return json.loads(raw) if raw else []

    async def append(self, session_id: str, turn: dict) -> None:
        turns = await self.history(session_id)
        turns.append(turn)
        self._hive.put(self._ns, session_id, json.dumps(turns[-self._max:]))
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_session.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
cd /opt/agentssot && git add api/app/gateway/session.py tests/gateway/test_session.py
git commit -m "feat(gateway): hive-backed session store"
```

---

## Phase 2 — Executors (swappable brain regions)

### Task 2.1: Config — fallback ladder

**Files:**
- Create: `api/app/gateway/config.py`
- Test: `tests/gateway/test_config.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/gateway/test_config.py
from api.app.gateway.config import ORCHESTRATE_LADDER, LOCAL_MODEL, OLLAMA_URL

def test_ladder_is_ordered_and_nonempty():
    assert ORCHESTRATE_LADDER[0]["name"] == "opus"
    assert any(s["name"].startswith("deepseek") for s in ORCHESTRATE_LADDER)
    assert ORCHESTRATE_LADDER[-1]["name"] == "local"

def test_local_model_and_url_present():
    assert LOCAL_MODEL and OLLAMA_URL.startswith("http")
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_config.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation** (use real values from Task 0.1 Step 5)
```python
# api/app/gateway/config.py
import os

OLLAMA_URL = os.environ.get("MADI_OLLAMA_URL", "http://192.168.1.225:11434")
LOCAL_MODEL = os.environ.get("MADI_LOCAL_MODEL", "qwen3.5:4b")

# Fallback ladder is CONFIG, not code. Each rung names how to invoke it.
# 'kind' tells the orchestrate executor which client path to take.
ORCHESTRATE_LADDER = [
    {"name": "opus",            "kind": "anthropic", "model": "claude-opus-4-8"},
    {"name": "deepseek-v4-pro", "kind": "chain",     "chain": "deepseek-plan"},
    {"name": "deepseek-flash",  "kind": "chain",     "chain": "glm-quick"},
    {"name": "local",           "kind": "ollama",    "model": LOCAL_MODEL},
]
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
cd /opt/agentssot && git add api/app/gateway/config.py tests/gateway/test_config.py
git commit -m "feat(gateway): orchestrate fallback ladder config"
```

### Task 2.2: hive-tool executor

**Files:**
- Create: `api/app/gateway/executors/hive_tool.py`
- Test: `tests/gateway/test_exec_hive_tool.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/gateway/test_exec_hive_tool.py
import pytest
from api.app.gateway.executors.hive_tool import HiveToolExecutor

class FakeHive:
    def recall(self, query, **kw): return [{"text": f"hit:{query}"}]

@pytest.mark.asyncio
async def test_hive_tool_recall_streams_results():
    ex = HiveToolExecutor(FakeHive())
    out = [e async for e in ex.execute("hive-tool", {"text": "recall greece"})]
    assert any(e.type == "event" and "hit:" in str(e.data) for e in out)
    assert out[-1].type == "done"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_exec_hive_tool.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**
```python
# api/app/gateway/executors/hive_tool.py
from __future__ import annotations
import re
from typing import Any, AsyncIterator
from api.app.gateway.executors.base import Executor
from api.app.gateway.protocol import Event

class HiveToolExecutor(Executor):
    name = "hive-tool"
    def __init__(self, hive: Any):
        self._hive = hive

    async def execute(self, intent: str, ctx: dict) -> AsyncIterator[Event]:
        text = ctx["text"]
        query = re.sub(r"^\s*/?(recall|remember|search)\s+", "", text, flags=re.I)
        try:
            results = self._hive.recall(query)
        except Exception as e:
            yield Event.error(f"hive recall failed: {e}", retryable=True)
            return
        yield Event.event({"kind": "recall", "query": query, "results": results})
        yield Event.done({"count": len(results)})
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_exec_hive_tool.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
cd /opt/agentssot && git add api/app/gateway/executors/hive_tool.py tests/gateway/test_exec_hive_tool.py
git commit -m "feat(gateway): hive-tool executor"
```

### Task 2.3: chat-local executor

**Files:**
- Create: `api/app/gateway/executors/chat_local.py`
- Test: `tests/gateway/test_exec_chat_local.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/gateway/test_exec_chat_local.py
import pytest
from api.app.gateway.executors.chat_local import ChatLocalExecutor

async def fake_stream(model, prompt):
    for tok in ["hel", "lo"]:
        yield tok

@pytest.mark.asyncio
async def test_chat_local_streams_tokens():
    ex = ChatLocalExecutor(stream_fn=fake_stream, model="qwen3.5:4b")
    out = [e async for e in ex.execute("chat-local", {"text": "hi", "history": []})]
    assert "".join(e.data for e in out if e.type == "token") == "hello"
    assert out[-1].type == "done"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_exec_chat_local.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**
```python
# api/app/gateway/executors/chat_local.py
from __future__ import annotations
from typing import AsyncIterator, Callable
from api.app.gateway.executors.base import Executor
from api.app.gateway.protocol import Event

class ChatLocalExecutor(Executor):
    name = "chat-local"
    def __init__(self, stream_fn: Callable, model: str):
        self._stream = stream_fn
        self._model = model

    async def execute(self, intent: str, ctx: dict) -> AsyncIterator[Event]:
        history = ctx.get("history", [])
        convo = "\n".join(f"{t['role']}: {t['text']}" for t in history)
        prompt = (convo + "\n" if convo else "") + f"user: {ctx['text']}\nmadi:"
        try:
            async for tok in self._stream(self._model, prompt):
                yield Event.token(tok)
        except Exception as e:
            yield Event.error(f"local chat failed: {e}", retryable=True)
            return
        yield Event.done()
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_exec_chat_local.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
cd /opt/agentssot && git add api/app/gateway/executors/chat_local.py tests/gateway/test_exec_chat_local.py
git commit -m "feat(gateway): chat-local executor"
```

### Task 2.4: orchestrate executor + fallback ladder

**Files:**
- Create: `api/app/gateway/executors/orchestrate.py`
- Test: `tests/gateway/test_exec_orchestrate.py`

- [ ] **Step 1: Write the failing test** (the reliability heart of the system)
```python
# tests/gateway/test_exec_orchestrate.py
import pytest
from api.app.gateway.executors.orchestrate import OrchestrateExecutor

@pytest.mark.asyncio
async def test_falls_over_to_deepseek_when_opus_unavailable(capsys):
    calls = []
    async def runner(rung, ctx):
        calls.append(rung["name"])
        if rung["name"] == "opus":
            raise RuntimeError("opus 529 overloaded")
        yield "answer from " + rung["name"]
    ladder = [{"name": "opus"}, {"name": "deepseek-v4-pro"}, {"name": "local"}]
    ex = OrchestrateExecutor(ladder=ladder, runner=runner)
    out = [e async for e in ex.execute("orchestrate", {"text": "design X"})]
    assert calls == ["opus", "deepseek-v4-pro"]            # stopped at first success
    assert any("deepseek-v4-pro" in str(e.data) for e in out if e.type == "token")
    assert any(e.type == "event" and e.data.get("fallover") for e in out)  # visible
    assert out[-1].type == "done" and out[-1].data["model"] == "deepseek-v4-pro"

@pytest.mark.asyncio
async def test_errors_only_when_whole_ladder_fails():
    async def runner(rung, ctx):
        raise RuntimeError("down")
        yield  # pragma: no cover
    ex = OrchestrateExecutor(ladder=[{"name": "opus"}, {"name": "local"}], runner=runner)
    out = [e async for e in ex.execute("orchestrate", {"text": "x"})]
    assert out[-1].type == "error"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_exec_orchestrate.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**
```python
# api/app/gateway/executors/orchestrate.py
from __future__ import annotations
from typing import AsyncIterator, Callable
from api.app.gateway.executors.base import Executor
from api.app.gateway.protocol import Event

class OrchestrateExecutor(Executor):
    name = "orchestrate"
    def __init__(self, ladder: list[dict], runner: Callable):
        # runner(rung, ctx) -> async iterator of token strings; raises on failure
        self._ladder = ladder
        self._runner = runner

    async def execute(self, intent: str, ctx: dict) -> AsyncIterator[Event]:
        last_err = None
        for i, rung in enumerate(self._ladder):
            produced = False
            try:
                async for tok in self._runner(rung, ctx):
                    if not produced and i > 0:
                        yield Event.event({"fallover": True, "to": rung["name"]})
                    produced = True
                    yield Event.token(tok)
                if produced:
                    yield Event.done({"model": rung["name"]})
                    return
            except Exception as e:
                last_err = e
                continue
        yield Event.error(f"orchestrate ladder exhausted: {last_err}", retryable=False)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_exec_orchestrate.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**
```bash
cd /opt/agentssot && git add api/app/gateway/executors/orchestrate.py tests/gateway/test_exec_orchestrate.py
git commit -m "feat(gateway): orchestrate executor with visible fallback ladder"
```

### Task 2.5: dispatch executor

**Files:**
- Create: `api/app/gateway/executors/dispatch.py`
- Test: `tests/gateway/test_exec_dispatch.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/gateway/test_exec_dispatch.py
import pytest
from api.app.gateway.executors.dispatch import DispatchExecutor

@pytest.mark.asyncio
async def test_dispatch_runs_job_and_reports():
    async def fake_run(cmd):
        yield "line1"; yield "line2"
    ex = DispatchExecutor(run_fn=fake_run)
    out = [e async for e in ex.execute("dispatch", {"text": "scan fleet"})]
    assert any(e.type == "token" and e.data == "line1" for e in out)
    assert out[-1].type == "done"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_exec_dispatch.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**
```python
# api/app/gateway/executors/dispatch.py
from __future__ import annotations
import re
from typing import AsyncIterator, Callable
from api.app.gateway.executors.base import Executor
from api.app.gateway.protocol import Event

# Map a verb to a safe, allow-listed command template. No free-form shell.
JOBS = {
    "scan fleet": ["bash", "/home/hari/.claude/scripts/fleet-scan.sh"],
}

class DispatchExecutor(Executor):
    name = "dispatch"
    def __init__(self, run_fn: Callable):
        self._run = run_fn  # run_fn(cmd: list[str]) -> async iterator of stdout lines

    async def execute(self, intent: str, ctx: dict) -> AsyncIterator[Event]:
        key = re.sub(r"^\s*/", "", ctx["text"].strip().lower())
        cmd = JOBS.get(key)
        if cmd is None:
            yield Event.error(f"no allow-listed job for: {key!r}", retryable=False)
            return
        yield Event.event({"kind": "job-start", "cmd": cmd})
        try:
            async for line in self._run(cmd):
                yield Event.token(line)
        except Exception as e:
            yield Event.error(f"job failed: {e}", retryable=True)
            return
        yield Event.done({"job": key})
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_exec_dispatch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
cd /opt/agentssot && git add api/app/gateway/executors/dispatch.py tests/gateway/test_exec_dispatch.py
git commit -m "feat(gateway): dispatch executor with allow-listed jobs"
```

### Task 2.6: Executor registry

**Files:**
- Modify: `api/app/gateway/executors/__init__.py`
- Test: `tests/gateway/test_registry.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/gateway/test_registry.py
from api.app.gateway.executors import build_registry

def test_registry_maps_intents_to_executors():
    reg = build_registry(hive=object(), chat_stream=lambda *a: None,
                          orchestrate_runner=lambda *a: None, job_run=lambda *a: None)
    for intent in ["hive-tool", "chat-local", "orchestrate", "dispatch"]:
        assert intent in reg
    assert reg["chat-local"].name == "chat-local"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_registry.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**
```python
# api/app/gateway/executors/__init__.py
from api.app.gateway.executors.hive_tool import HiveToolExecutor
from api.app.gateway.executors.chat_local import ChatLocalExecutor
from api.app.gateway.executors.orchestrate import OrchestrateExecutor
from api.app.gateway.executors.dispatch import DispatchExecutor
from api.app.gateway.config import ORCHESTRATE_LADDER, LOCAL_MODEL

def build_registry(*, hive, chat_stream, orchestrate_runner, job_run) -> dict:
    return {
        "hive-tool":  HiveToolExecutor(hive),
        "chat-local": ChatLocalExecutor(chat_stream, LOCAL_MODEL),
        "orchestrate": OrchestrateExecutor(ORCHESTRATE_LADDER, orchestrate_runner),
        "dispatch":   DispatchExecutor(job_run),
    }
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
cd /opt/agentssot && git add api/app/gateway/executors/__init__.py tests/gateway/test_registry.py
git commit -m "feat(gateway): executor registry"
```

---

## Phase 3 — Service orchestration + routes + feeders

### Task 3.1: Gateway service (route → dispatch → persist → stream)

**Files:**
- Create: `api/app/gateway/service.py`
- Test: `tests/gateway/test_service.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/gateway/test_service.py
import pytest
from api.app.gateway.service import GatewayService
from api.app.gateway.protocol import Event, InboundMessage

class StubExec:
    name = "chat-local"
    async def execute(self, intent, ctx):
        yield Event.token("hi"); yield Event.done()

class StubRouter:
    def classify(self, text, explicit=None): return "chat-local", {}

class FakeSession:
    def __init__(self): self.turns = {}
    async def history(self, sid): return self.turns.get(sid, [])
    async def append(self, sid, turn): self.turns.setdefault(sid, []).append(turn)

@pytest.mark.asyncio
async def test_service_routes_streams_and_persists():
    sess = FakeSession()
    svc = GatewayService(router=StubRouter(), registry={"chat-local": StubExec()}, session=sess)
    events = [e async for e in svc.handle(InboundMessage.from_dict({"text": "hi", "session_id": "s1"}))]
    assert any(e.type == "token" for e in events)
    hist = await sess.history("s1")
    assert hist[0]["role"] == "user" and hist[-1]["role"] == "madi"

@pytest.mark.asyncio
async def test_service_unknown_intent_emits_error():
    svc = GatewayService(router=StubRouter(), registry={}, session=FakeSession())
    events = [e async for e in svc.handle(InboundMessage.from_dict({"text": "hi", "session_id": "s1"}))]
    assert events[-1].type == "error"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_service.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**
```python
# api/app/gateway/service.py
from __future__ import annotations
from typing import AsyncIterator
from api.app.gateway.protocol import Event, InboundMessage

class GatewayService:
    def __init__(self, *, router, registry: dict, session):
        self._router = router
        self._registry = registry
        self._session = session

    async def handle(self, msg: InboundMessage) -> AsyncIterator[Event]:
        await self._session.append(msg.session_id, {"role": "user", "text": msg.text})
        intent, args = self._router.classify(msg.text, explicit=msg.intent)
        ex = self._registry.get(intent) if intent else None
        if ex is None:
            yield Event.error(f"no executor for intent {intent!r}", retryable=False)
            return
        history = await self._session.history(msg.session_id)
        ctx = {"text": msg.text, "history": history, "args": args}
        collected = []
        async for ev in ex.execute(intent, ctx):
            if ev.type == "token":
                collected.append(ev.data)
            yield ev
        if collected:
            await self._session.append(msg.session_id, {"role": "madi", "text": "".join(collected)})
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_service.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**
```bash
cd /opt/agentssot && git add api/app/gateway/service.py tests/gateway/test_service.py
git commit -m "feat(gateway): service orchestration"
```

### Task 3.2: Status feeders

**Files:**
- Create: `api/app/gateway/feeders.py`
- Test: `tests/gateway/test_feeders.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/gateway/test_feeders.py
import pytest
from api.app.gateway.feeders import snapshot_status

@pytest.mark.asyncio
async def test_snapshot_includes_required_panels():
    async def hive_stat(): return {"items": 4012, "recalls": 18, "synapse_new": 3}
    async def exec_health(): return {"opus": "ok", "deepseek-v4-pro": "ok", "local": "ok"}
    async def fleet_stat(): return {"up": 13, "total": 13}
    async def chains_stat(): return {"running": 0}
    snap = await snapshot_status(hive_stat, exec_health, fleet_stat, chains_stat)
    assert snap["memory"]["items"] == 4012
    assert snap["executors"]["opus"] == "ok"
    assert snap["fleet"]["up"] == 13
    assert "chains" in snap

@pytest.mark.asyncio
async def test_snapshot_is_null_graceful():
    async def boom(): raise RuntimeError("x")
    async def ok(): return {"up": 1, "total": 1}
    snap = await snapshot_status(boom, boom, ok, boom)
    assert snap["memory"] is None and snap["fleet"]["up"] == 1  # NULL-graceful per display rule
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_feeders.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**
```python
# api/app/gateway/feeders.py
from __future__ import annotations
from typing import Callable, Awaitable

async def _safe(fn: Callable[[], Awaitable]):
    try:
        return await fn()
    except Exception:
        return None  # display layers must handle NULL gracefully

async def snapshot_status(hive_stat, exec_health, fleet_stat, chains_stat) -> dict:
    return {
        "memory":    await _safe(hive_stat),
        "executors": await _safe(exec_health),
        "fleet":     await _safe(fleet_stat),
        "chains":    await _safe(chains_stat),
    }
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_feeders.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
cd /opt/agentssot && git add api/app/gateway/feeders.py tests/gateway/test_feeders.py
git commit -m "feat(gateway): null-graceful status feeders"
```

### Task 3.3: FastAPI routes (WS + SSE) and app wiring

**Files:**
- Create: `api/app/gateway/routes.py`
- Modify: the app route-registration site (from Task 0.1 Step 1 — Integration Facts)
- Test: `tests/gateway/test_routes.py`

> Bind real clients here: the hive client (Task 0.1 Step 2), an async Ollama token-stream fn, an `orchestrate_runner` that switches on `rung["kind"]` (anthropic/chain/ollama), and a `job_run` that streams subprocess stdout. The runner + stream fns are thin I/O adapters; keep them in `routes.py` so the executors stay pure/testable.

- [ ] **Step 1: Write the failing test** (WS round-trip via FastAPI TestClient)
```python
# tests/gateway/test_routes.py
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.app.gateway.routes import build_router

def _app():
    app = FastAPI()
    app.include_router(build_router(service_factory=_stub_service))
    return app

class _StubSvc:
    async def handle(self, msg):
        from api.app.gateway.protocol import Event
        yield Event.token("pong"); yield Event.done()

def _stub_service():
    return _StubSvc()

def test_ws_echo_roundtrip():
    client = TestClient(_app())
    with client.websocket_connect("/gateway/ws") as ws:
        ws.send_json({"text": "ping", "session_id": "s1"})
        first = ws.receive_json()
        assert first == {"type": "token", "data": "pong"}
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_routes.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**
```python
# api/app/gateway/routes.py
from __future__ import annotations
import asyncio, json
from typing import Callable
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from api.app.gateway.protocol import InboundMessage
from api.app.gateway.feeders import snapshot_status

def build_router(*, service_factory: Callable, status_sources: dict | None = None) -> APIRouter:
    router = APIRouter(prefix="/gateway", tags=["gateway"])

    @router.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        svc = service_factory()
        try:
            while True:
                raw = await socket.receive_json()
                msg = InboundMessage.from_dict(raw)
                async for ev in svc.handle(msg):
                    await socket.send_json(ev.to_dict())
        except WebSocketDisconnect:
            return

    @router.get("/sse/status")
    async def sse_status():
        async def gen():
            srcs = status_sources or {}
            while True:
                snap = await snapshot_status(
                    srcs.get("hive"), srcs.get("exec"), srcs.get("fleet"), srcs.get("chains"))
                yield f"data: {json.dumps(snap)}\n\n"
                await asyncio.sleep(3)
        return StreamingResponse(gen(), media_type="text/event-stream")

    return router
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cd /opt/agentssot && python -m pytest tests/gateway/test_routes.py -v`
Expected: PASS.

- [ ] **Step 5: Wire into the app** using the Integration Facts site. Add next to the existing `include_router` calls:
```python
# in the app factory / main (exact location from Task 0.1)
from api.app.gateway.routes import build_router as build_gateway_router
from api.app.gateway.service import GatewayService
from api.app.gateway.router import IntentRouter, make_ollama_classifier
from api.app.gateway.session import SessionStore
from api.app.gateway.executors import build_registry
from api.app.gateway.config import OLLAMA_URL, LOCAL_MODEL
# bind real hive client (Integration Facts), chat_stream, orchestrate_runner, job_run here
# then:
def _gateway_service_factory():
    router = IntentRouter(classifier=make_ollama_classifier(OLLAMA_URL, LOCAL_MODEL))
    registry = build_registry(hive=HIVE, chat_stream=CHAT_STREAM,
                              orchestrate_runner=ORCH_RUNNER, job_run=JOB_RUN)
    return GatewayService(router=router, registry=registry,
                          session=SessionStore(HIVE))
app.include_router(build_gateway_router(service_factory=_gateway_service_factory,
                                        status_sources=STATUS_SOURCES))
```

- [ ] **Step 6: Run the app and smoke-test the WS.**
Run: `cd /opt/agentssot && (uvicorn <app-module>:app --port 8099 &) ; sleep 3 ; python -c "from websocket import create_connection; w=create_connection('ws://127.0.0.1:8099/gateway/ws'); import json; w.send(json.dumps({'text':'hi','session_id':'smoke'})); print(w.recv())" ; kill %1`
Expected: a JSON token event prints. (Use the real app module from Integration Facts.)

- [ ] **Step 7: Commit**
```bash
cd /opt/agentssot && git add api/app/gateway/routes.py tests/gateway/test_routes.py <modified app file>
git commit -m "feat(gateway): WS + SSE routes wired into app"
```

---

## Phase 4 — HUD frontend ("Obsidian Terminal")

> Follow the app's existing page-serving + mtime cache-bust pattern (Task 0.1 Step 4). The CSS uses the locked tokens from the design spec §4.

### Task 4.1: HUD shell + Obsidian Terminal CSS (ambient mode)

**Files:**
- Create: `api/app/ui/hud.html`
- Create: `api/app/ui/assets/hud.css`
- Modify: `api/app/ui/_nav.html` (add HUD link — exact pattern from Integration Facts)

- [ ] **Step 1:** Create `hud.css` with the locked tokens:
```css
/* api/app/ui/assets/hud.css */
:root{ --bg:#0a0a0c; --panel:#101013; --line:#26241f; --amber:#d9a441;
  --amber-dim:#8a6e3a; --ink:#ece7dd; --dim:#7d776c; }
body.hud{ margin:0; background:var(--bg); color:var(--ink);
  font-family:ui-monospace,Menlo,monospace; height:100vh; display:flex; flex-direction:column; }
.hud-ticker{ display:flex; gap:14px; align-items:center; padding:8px 12px;
  border-bottom:1px solid var(--line); font-size:12px; color:var(--dim); }
.hud-ticker b{ color:var(--amber); font-weight:400; }
.hud-mid{ flex:1; display:flex; gap:10px; padding:10px; min-height:0; }
.hud-col{ flex:0 0 200px; border:1px solid var(--line); border-radius:6px;
  background:var(--panel); padding:10px; font-size:12px; color:var(--dim); overflow:auto; }
.hud-col .hd{ color:var(--amber-dim); text-transform:uppercase; font-size:10px; letter-spacing:1px; }
.hud-stage{ flex:1; border:1px solid var(--line); border-radius:8px; position:relative;
  background:linear-gradient(#0d0d10,#0a0a0c); display:flex; flex-direction:column;
  align-items:center; padding:16px; overflow:auto; }
.hud-stage::after{ content:''; position:absolute; inset:0; pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent 0 3px,rgba(217,164,65,.025) 3px 4px); }
.hud-ring{ width:96px; height:96px; border-radius:50%; border:1px solid var(--amber);
  box-shadow:0 0 26px rgba(217,164,65,.22),0 0 5px rgba(217,164,65,.3) inset;
  display:flex; align-items:center; justify-content:center; color:var(--amber);
  letter-spacing:2px; font-size:12px; margin:18px 0; animation:breathe 4s ease-in-out infinite; }
@keyframes breathe{ 0%,100%{ box-shadow:0 0 22px rgba(217,164,65,.18),0 0 5px rgba(217,164,65,.25) inset; }
  50%{ box-shadow:0 0 34px rgba(217,164,65,.32),0 0 8px rgba(217,164,65,.4) inset; } }
.hud-brief{ width:100%; max-width:640px; border:1px dashed #2a2824; border-radius:6px; padding:14px; }
.hud-brief .kick{ font-size:10px; letter-spacing:1.5px; color:var(--amber-dim); text-transform:uppercase; }
.hud-brief h2{ font-family:Georgia,serif; font-weight:500; color:#f4f0e8; margin:6px 0; }
.hud-brief p{ font-family:Georgia,serif; color:#b8b1a4; line-height:1.5; }
.hud-bar{ width:100%; max-width:640px; margin-top:auto; display:flex; }
.hud-bar input{ flex:1; background:rgba(217,164,65,.04); border:1px solid var(--amber-dim);
  border-radius:4px; padding:10px 12px; color:var(--ink); font-family:inherit; font-size:13px; }
.dot{ width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:4px;
  background:var(--amber); box-shadow:0 0 5px var(--amber); }
.dot.off{ background:#3a3630; box-shadow:none; }
.hud-stream{ width:100%; max-width:760px; text-align:left; white-space:pre-wrap; line-height:1.5; }
body.hud.active .hud-ring{ width:44px; height:44px; margin:0 10px 0 0; animation:none; }
```

- [ ] **Step 2:** Create `hud.html` (structure for both modes; JS toggles `body.active`):
```html
<!-- api/app/ui/hud.html -->
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Madi</title>
<link rel="stylesheet" href="/ui/assets/hud.css?v=__MTIME__"></head>
<body class="hud">
  <div class="hud-ticker">
    <span>FLEET <b id="t-fleet">--/--</b></span>
    <span><span class="dot" id="t-opus"></span>Opus</span>
    <span><span class="dot" id="t-deepseek"></span>deepseek</span>
    <span><span class="dot" id="t-synapse"></span>synapse</span>
    <span id="t-clock" style="margin-left:auto"></span>
  </div>
  <div class="hud-mid">
    <div class="hud-col"><div class="hd">Memory</div><div id="p-memory">--</div>
      <div class="hd" style="margin-top:10px">Synapse</div><div id="p-synapse">--</div></div>
    <div class="hud-stage">
      <div style="display:flex;align-items:center"><div class="hud-ring">MADI</div>
        <span id="madi-state" style="color:var(--dim);font-size:11px"></span></div>
      <div class="hud-brief" id="brief"><div class="kick">Today</div>
        <h2>No briefing yet</h2><p>The proactive briefing is a later phase.</p></div>
      <div class="hud-stream" id="stream"></div>
      <form class="hud-bar" id="bar"><input id="cmd" autocomplete="off"
        placeholder="madi> speak or type a command_"></form>
    </div>
    <div class="hud-col"><div class="hd">Agents / Brain</div><div id="p-exec">--</div>
      <div class="hd" style="margin-top:10px">Chains</div><div id="p-chains">--</div></div>
  </div>
  <script src="/ui/assets/hud.js?v=__MTIME__"></script>
</body></html>
```

- [ ] **Step 3:** Add the HUD link to `_nav.html` following the existing item pattern (Integration Facts). Serve `hud.html` at route `/hud` using the same page-serving mechanism as the other UI pages (apply the `__MTIME__` cache-bust substitution exactly as existing pages do).

- [ ] **Step 4: Smoke-test render.**
Run: open `http://192.168.1.225:8088/hud` (or the app's host) — expect the ambient HUD: breathing ring, ticker, two side columns, command bar. No console errors for missing assets.

- [ ] **Step 5: Commit**
```bash
cd /opt/agentssot && git add api/app/ui/hud.html api/app/ui/assets/hud.css api/app/ui/_nav.html <serving file if changed>
git commit -m "feat(hud): Obsidian Terminal ambient shell"
```

### Task 4.2: HUD JS — command channel + mode morph + status panels

**Files:**
- Create: `api/app/ui/assets/hud.js`

- [ ] **Step 1:** Implement WS command channel, SSE status, and ambient⇄active morph:
```javascript
// api/app/ui/assets/hud.js
(function () {
  const $ = (id) => document.getElementById(id);
  const stream = $("stream"), bodyEl = document.body;

  // --- command channel (WebSocket into the gateway) ---
  const wsProto = location.protocol === "https:" ? "wss" : "ws";
  let ws, sessionId = "hud-" + (localStorage.madiSession || (localStorage.madiSession = Date.now()));
  function connect() {
    ws = new WebSocket(`${wsProto}://${location.host}/gateway/ws`);
    ws.onmessage = (m) => render(JSON.parse(m.data));
    ws.onclose = () => setTimeout(connect, 1500);
  }
  connect();

  let madiLine = null;
  function render(ev) {
    if (ev.type === "token") {
      if (!madiLine) { madiLine = document.createElement("div"); madiLine.style.color = "var(--amber)"; stream.appendChild(madiLine); }
      madiLine.textContent += ev.data;
    } else if (ev.type === "event" && ev.data && ev.data.fallover) {
      const n = document.createElement("div"); n.style.color = "var(--dim)";
      n.textContent = `— fell over to ${ev.data.to}`; stream.appendChild(n);
    } else if (ev.type === "error") {
      const n = document.createElement("div"); n.style.color = "#c0563f";
      n.textContent = "! " + ev.data.message; stream.appendChild(n);
    } else if (ev.type === "done") {
      madiLine = null; $("madi-state").textContent = "";
    }
    stream.scrollTop = stream.scrollHeight;
  }

  $("bar").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("cmd").value.trim(); if (!text) return;
    bodyEl.classList.add("active");                 // morph: ambient -> working
    const you = document.createElement("div"); you.textContent = "> " + text; stream.appendChild(you);
    $("madi-state").textContent = "thinking";
    ws.send(JSON.stringify({ text, session_id: sessionId }));
    $("cmd").value = "";
  });

  // --- status panels (SSE) ---
  const dot = (id, ok) => $(id).classList.toggle("off", !ok);
  const es = new EventSource("/gateway/sse/status");
  es.onmessage = (m) => {
    const s = JSON.parse(m.data);
    if (s.fleet) $("t-fleet").textContent = `${s.fleet.up}/${s.fleet.total}`;
    if (s.executors) { dot("t-opus", s.executors.opus === "ok"); dot("t-deepseek", (s.executors["deepseek-v4-pro"]||"")==="ok"); }
    dot("t-synapse", !!(s.memory));
    $("p-memory").textContent = s.memory ? `hive: ${s.memory.items}\nrecalls: ${s.memory.recalls}` : "—";
    $("p-synapse").textContent = s.memory ? `+${s.memory.synapse_new} links` : "—";
    $("p-exec").textContent = s.executors ? Object.entries(s.executors).map(([k,v])=>`${k}: ${v}`).join("\n") : "—";
    $("p-chains").textContent = s.chains ? `running: ${s.chains.running}` : "—";
  };

  // clock
  setInterval(() => { const d = new Date(); $("t-clock").textContent = d.toTimeString().slice(0,5); }, 1000);
})();
```

- [ ] **Step 2: Manual verification (the real acceptance test).**
  1. Open `/hud`. Type `recall greece`. Expect: morph to active, a `> recall greece` line, then a `hive-tool` event rendering results, then state clears.
  2. Type `what is the time` (freeform → classifier → chat-local). Expect streamed tokens from the local model.
  3. Confirm ticker dots reflect real executor health and fleet count from SSE.

- [ ] **Step 3: Commit**
```bash
cd /opt/agentssot && git add api/app/ui/assets/hud.js
git commit -m "feat(hud): command channel, mode morph, live status panels"
```

---

## Phase 5 — Integration, ledger cleanup, definition of done

### Task 5.1: Full gateway test sweep + fallback proof

- [ ] **Step 1:** Run the whole gateway suite.
  Run: `cd /opt/agentssot && python -m pytest tests/gateway -v`
  Expected: all green.

- [ ] **Step 2:** Prove the fallback ladder live. Temporarily set an invalid Anthropic key (env) so `opus` rung fails, then issue an `orchestrate`-class command in the HUD. Expect: a visible "fell over to deepseek-v4-pro" line in the stream and a logged fallover. Restore the key.

- [ ] **Step 3: Commit** any fixes.
```bash
cd /opt/agentssot && git commit -am "test(gateway): full sweep + live fallback proof" || echo "nothing to commit"
```

### Task 5.2: Execute the Consolidation Ledger (RETIRE-confirmed only)

> Only items confirmed dead in Task 0.2. ABSORB items (harivoice/air Telegram) are **not** deleted here — they're deleted in their own future channel-adapter projects once those work. This task enforces "no NEW orphans" + clears confirmed-dead ones.

- [ ] **Step 1:** For each RETIRE-confirmed repo-root artifact, delete and commit individually:
```bash
cd /opt/agentssot && git rm firebase-debug.log && git commit -m "chore: retire orphan firebase-debug.log"
# repeat for after/, dropbox/ ONLY if Task 0.2 confirmed them dead
```

- [ ] **Step 2:** If Task 0.2 confirmed flightdeck is superseded and not load-bearing, record its retirement decision in the design doc's ledger and open a follow-up (do not delete a running service blindly — stop, verify no consumers, then remove in a dedicated change).

- [ ] **Step 3:** Update the ledger table in `docs/plans/2026-05-31-madi-hud-gateway-design.md` with final verdicts + dates. Commit.
```bash
cd /opt/agentssot && git commit -am "docs: finalize consolidation ledger verdicts"
```

### Task 5.3: Definition of done checklist

- [ ] Gateway module live in `api/app`; `WS /gateway/ws` + `GET /gateway/sse/status` working against real hive/fleet/chain data.
- [ ] HUD ambient + active modes function at `/hud` in Obsidian Terminal identity.
- [ ] `orchestrate` fallback ladder verified live (Opus→deepseek) with visible logging.
- [ ] `python -m pytest tests/gateway` all green.
- [ ] Consolidation Ledger audited; RETIRE-confirmed orphans deleted; ABSORB items tracked with old paths intact (not yet deleted).
- [ ] No new orphans introduced (every new file is wired and referenced).

---

## Self-Review (completed by author)

- **Spec coverage:** gateway (T1.x), hybrid router (T1.3–1.4), executors + ladder (T2.x), session state (T1.5), feeders/SSE (T3.2–3.3), HUD two modes + identity (T4.x), consolidation ledger (T0.2, T5.2), reliability/fallover (T2.4, T5.1 Step 2), testing (every task). All design §3–§9 sections map to tasks.
- **Placeholders:** none — `__MTIME__` is a real substitution token handled by the existing serving layer (Task 0.1 Step 4); Integration Facts (Task 0.1) deliberately defers *existing-file line numbers* because a second agent is mid-edit, which is honest, not a placeholder.
- **Type consistency:** `Event`/`InboundMessage` (T1.1) reused everywhere; `Executor.execute(intent, ctx)` signature consistent across all executors; `build_registry(hive, chat_stream, orchestrate_runner, job_run)` keys match registry consumers; `snapshot_status(hive, exec, fleet, chains)` arg order consistent T3.2↔T3.3.
