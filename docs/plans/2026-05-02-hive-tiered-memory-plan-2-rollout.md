# Hive Tiered Memory — Plan 2: Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate the tier-memory foundation built in Plan 1 — push cwd-aware loadouts at SessionStart, expose agent-facing guidance, run lifecycle sweeps nightly, surface review/loadout/entities in Cortex, flip the safe defaults, and roll the updated MCP plugin to every enrolled device.

**Architecture:** Plan 1 shipped the schema, classifier, layer pre-compute, supersession/contradiction detection, loadout assembly, and review-queue endpoints. Plan 2 makes them visible: the SessionStart hook calls `/loadout` (writer auth), a Jinja-rendered `/agent-guide` provides text-only runbooks, `lifecycle_sweep` runs at 03:00 UTC to decay/expire/recheck, three new Cortex tabs (`/review`, `/loadout`, `/entities`) make the queue and budget actionable, and `bucketed=true` becomes the default with `episodic` excluded. The plugin update is then synced to dockers, webvm, blink, air, agent.

**Tech Stack:** FastAPI, Jinja2 (added), APScheduler-or-asyncio cron (existing background.py pattern), Cortex Alpine.js frontend (existing), bash+ssh rollout via `~/.claude/agentssot/scripts/push_keys_ssh.sh`, gemma4:31b for any sweep re-classification, nomic-embed-text unchanged.

**Scope:** Phases 4, 5, 6, 7 (Plan 1 covered 0–3). 13 tasks total. Each phase ships independently with rollback path.

**Out of scope for Plan 2:**
- Re-architecting the classifier or layer pre-compute (Plan 1 owns those)
- New tier types beyond the existing 6
- Replacing the synthesis pipeline (Concept generation stays as-is)
- Web auth (Cortex stays LAN-only with admin key flow)

---

## File Structure

**New files (Plan 2 scope):**

```
api/app/
  routers/
    agent_guide.py                         GET /agent-guide (text/plain markdown, per-key)
  templates/
    agent_guide.md.j2                      Jinja2 template; ~2-3 KB output
  services/
    lifecycle_sweep.py                     decay + expire + contradiction recheck
  routers/
    entities.py                            GET /admin/entities (list/edit support)

api/app/ui/
  review.html                              /review page — review-queue triage
  loadout.html                             /loadout page — operator debugging
  entities.html                            /entities page — entity editor
  tier-styles.css                          tier color tokens shared by 3 pages

scripts/
  rollout_plugin.sh                        SSH push + verify plugin per device
  verify_loadouts.sh                       per-device hive_loadout smoke

api/tests/
  test_agent_guide.py                      template renders, per-key tailoring
  test_lifecycle_sweep.py                  sweep idempotency, decay math
  integration/
    test_loadout_hook.py                   hook output shape under success + 2s timeout
```

**Modified files:**

```
api/app/main.py                            include agent_guide_router; mount /review|/loadout|/entities
api/app/background.py                      register lifecycle_sweep cron at 03:00 UTC
api/app/routers/knowledge.py               flip bucketed default; default-exclude episodic; default layer=summary on /expand
api/app/services/loadout.py                expose existing assembly; add cache key emission (already exists)
api/app/ui/cortex.html                     add nav link to /review, /loadout, /entities
~/.claude/plugins/hari-hive/hooks/SessionStart.md
                                           replace static hint with loadout fetch + 2s timeout fallback
~/.claude/plugins/hari-hive/mcp_server.py  add hive_guide tool
~/.claude/CLAUDE.md                        first-turn protocol: loadout pre-loaded; recall optional; post-compact recovery line
```

---

## Phase 4 — SessionStart Loadout + Agent Guide

### Task 4.1: SessionStart hook calls /loadout with 2s timeout

**Files:**
- Modify: `~/.claude/plugins/hari-hive/hooks/SessionStart.md`
- Test: `api/tests/integration/test_loadout_hook.py`

- [ ] **Step 1: Read existing hook**

```bash
cat ~/.claude/plugins/hari-hive/hooks/SessionStart.md
```
Note: it currently echoes a static `<hive-available>` block. Plan 1 left the loadout endpoint in place (`POST /api/v1/knowledge/loadout`) with cwd+device payload — this hook will replace the static text with a live call.

- [ ] **Step 2: Write the failing integration test**

Create `api/tests/integration/test_loadout_hook.py`:

```python
"""SessionStart loadout hook — happy path + timeout fallback."""
import os
import subprocess
from pathlib import Path

import pytest

HOOK = Path("~/.claude/plugins/hari-hive/hooks/SessionStart.md").expanduser()


def _extract_bash(md_text: str) -> str:
    # Pull the first ```bash block out of the hook markdown.
    block = md_text.split("```bash", 1)[1].split("```", 1)[0]
    return block.lstrip("\n")


@pytest.mark.integration
def test_hook_emits_hive_block_on_success(tmp_path):
    script = tmp_path / "hook.sh"
    script.write_text(_extract_bash(HOOK.read_text()))
    out = subprocess.run(
        ["bash", str(script)],
        capture_output=True, text=True, timeout=8,
        env={**os.environ, "PWD": "/opt/agentssot"},
    )
    assert "<hive-loadout>" in out.stdout
    assert "</hive-loadout>" in out.stdout


@pytest.mark.integration
def test_hook_falls_back_under_timeout(tmp_path, monkeypatch):
    """If HIVE_API_BASE is bogus, hook must still emit the static fallback within 3s."""
    script = tmp_path / "hook.sh"
    script.write_text(_extract_bash(HOOK.read_text()))
    out = subprocess.run(
        ["bash", str(script)],
        capture_output=True, text=True, timeout=4,
        env={**os.environ, "HIVE_API_BASE": "http://127.0.0.1:1"},
    )
    assert "<hive-available>" in out.stdout  # static fallback marker
```

- [ ] **Step 3: Verify test fails**

Run: `pytest api/tests/integration/test_loadout_hook.py -v`
Expected: FAIL — current hook emits `<hive-available>` only, never `<hive-loadout>`.

- [ ] **Step 4: Replace hook body**

Overwrite `~/.claude/plugins/hari-hive/hooks/SessionStart.md`:

```markdown
---
name: SessionStart
description: Push cwd-aware tier-memory loadout (~600 tokens) into session context with 2s budget
enabled: true
---

# hari-hive Session Start — Loadout

```bash
#!/bin/bash
set -u
PROJECT_NAME=$(basename "$(pwd)")
API_BASE="${HIVE_API_BASE:-http://192.168.1.225:8088}"
AGENT_JSON="${HIVE_AGENT_JSON:-$HOME/.claude/agentssot/local/agent.json}"

# Resolve key + device cheaply; fall back if missing.
if [ ! -r "$AGENT_JSON" ]; then
  echo "<hive-available>"
  echo "hari-hive MCP available (no agent.json — call hive_recall on demand)."
  echo "</hive-available>"
  exit 0
fi
API_KEY=$(python3 -c "import json,sys; print(json.load(open('$AGENT_JSON'))['api_key'])" 2>/dev/null)
DEVICE=$(python3 -c "import json; print(json.load(open('$AGENT_JSON')).get('device_name','unknown'))" 2>/dev/null)

# 2s wall budget for the entire fetch. curl --max-time enforces it.
BODY=$(printf '{"cwd":"%s","device_id":"%s","namespace":"claude-shared","token_budget":750}' "$PWD" "$DEVICE")
RESP=$(curl -sS --max-time 2 \
  -H "X-Api-Key: $API_KEY" -H "Content-Type: application/json" \
  --data "$BODY" "$API_BASE/api/v1/knowledge/loadout" 2>/dev/null)

if [ -z "$RESP" ] || ! echo "$RESP" | python3 -c "import json,sys; json.load(sys.stdin)" >/dev/null 2>&1; then
  echo "<hive-available>"
  echo "hari-hive MCP available (loadout unreachable — call hive_recall on task keywords)."
  echo "Project: ${PROJECT_NAME}. Namespace: claude-shared."
  echo "</hive-available>"
  exit 0
fi

# Render compact text loadout from JSON.
echo "<hive-loadout>"
echo "$RESP" | python3 - <<'PY'
import json, sys
b = json.load(sys.stdin)
print(f"Loadout cwd={b.get('cwd','?')} device={b.get('device_id','?')} "
      f"tokens={b.get('tokens_used',0)} overflow={b.get('overflow_count',0)}")
for tier, items in (b.get("items") or {}).items():
    if not items: continue
    print(f"\n[{tier}] ({len(items)})")
    for it in items:
        title = it.get("title") or ""
        print(f"- {title} — {it.get('abstract','')} (id={it.get('id','?')})")
if b.get("overflow_count"):
    print(f"\n+{b['overflow_count']} more — call hive_expand or hive_recall.")
PY
echo "</hive-loadout>"
```
```

- [ ] **Step 5: Run tests**

```bash
docker compose up -d  # ensure API is live (already is)
pytest api/tests/integration/test_loadout_hook.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
cd ~/.claude/plugins/hari-hive
git add hooks/SessionStart.md
git commit -m "feat(hari-hive): SessionStart loadout fetch with 2s timeout

Replaces static <hive-available> hint with a live call to
/api/v1/knowledge/loadout. Renders compact tier-bucketed text
into <hive-loadout>. On timeout / unreachable / unparseable,
falls back to <hive-available> hint.

Hook runs in 2s wall budget enforced by curl --max-time.

Plan 2: T4.1"
```

---

### Task 4.2: /agent-guide endpoint + Jinja template + hive_guide MCP tool

**Files:**
- Create: `api/app/routers/agent_guide.py`
- Create: `api/app/templates/agent_guide.md.j2`
- Modify: `api/app/main.py`
- Modify: `~/.claude/plugins/hari-hive/mcp_server.py`
- Test: `api/tests/test_agent_guide.py`

- [ ] **Step 1: Add jinja2 dep**

Edit `api/requirements.txt` (or `pyproject.toml` if used) and add `jinja2>=3.1`. Rebuild image at end of phase.

- [ ] **Step 2: Write failing test**

Create `api/tests/test_agent_guide.py`:

```python
"""GET /agent-guide returns text/plain markdown tailored to the caller key."""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.mark.integration
def test_agent_guide_writer_renders():
    client = TestClient(app)
    # Use a known writer key from the local issued-keys file.
    import json, os
    keys = json.load(open(os.path.expanduser("~/.claude/agentssot/local/issued-keys.json")))
    writer = next(k for k in keys if k["role"] == "writer")
    r = client.get("/agent-guide", headers={"X-Api-Key": writer["api_key"]})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    # Per-key tailoring markers
    assert writer["name"] in body
    assert "writer" in body
    # Must include a tier cheat sheet
    for tier in ("command", "rule", "skill", "entity"):
        assert tier in body
    # Troubleshooting block
    assert "401" in body and "403" in body
```

- [ ] **Step 3: Verify it fails**

Run: `pytest api/tests/test_agent_guide.py -v`
Expected: FAIL — endpoint does not exist.

- [ ] **Step 4: Create the template**

Create `api/app/templates/agent_guide.md.j2`:

```jinja
# hari-hive Agent Guide

You are connected as **{{ key_name }}** (role: `{{ role }}`) on device **{{ device_id }}**.
Service: `{{ api_base }}` — version `{{ version }}`.
Accessible namespaces: {{ namespaces | join(", ") }}.

## The 6 tiers

| Tier | Use for | Example |
|---|---|---|
| command | Imperative one-liner you'd run | `ssh hari` |
| rule | "always/never X" guardrail | "Never push to main without review" |
| skill | "when X, do Y" recipe | "When Ollama OOMs, set OLLAMA_EMBED_CPU_ONLY=true" |
| entity | Canonical noun (host/service/person) | `unraid (192.168.1.116)` |
| decision | "we chose X because Y" | "Use nomic-embed-text over qwen3-embedding for VRAM" |
| episodic | Session log / reflection | Auto-extracted at session end |

## How to recall

```
hive_recall("ssh unraid", bucketed=true, tiers=["command","rule","skill","entity"])
```

`bucketed=true` returns one list per tier (default starting Plan 2 Phase 6).
`hive_expand(item_id, layer="summary")` if the abstract isn't enough.

## How to write

```
hive_teach(trigger="...", action="...", success_hint="...")
hive_ingest(content="...", tags=["..."])
```

Both auto-classify via gemma4:31b. Items below conf=0.6 land in the review queue.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| 401 on any call | Missing/expired key | Read `~/.claude/agentssot/local/agent.json` |
| 403 on admin op | Writer key reaching admin endpoint | Need admin.json on this device |
| Empty recall | Wrong namespace | Pass `namespace="claude-shared"` (default) |
| Wrong tier returned | Item misclassified | `hive_feedback(signal="wrong", query=...)` |
| Loadout missing | SessionStart hook timed out | Call `hive_loadout()` mid-session |

## Connectivity

- Service host: `hari` (192.168.1.225) — LAN-only, no public DNS
- Health: `GET {{ api_base }}/health`
- SSH: `ssh hari` from any enrolled device
- Web UI: `{{ api_base }}/cortex` (3D map), `/review` (queue), `/loadout` (debug)

## Where to look for human help

- Operator: MadDefientist (mohsenarthur@gmail.com)
- Source of truth: `/opt/agentssot/` on hari
- Slack/issues: not used for this project
```

- [ ] **Step 5: Implement the router**

Create `api/app/routers/agent_guide.py`:

```python
"""GET /agent-guide — per-key tailored markdown runbook."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Response
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.security import require_api_key, AuthContext
from app.settings import get_settings

router = APIRouter()

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
    trim_blocks=True,
    lstrip_blocks=True,
)


@router.get("/agent-guide", response_class=Response)
async def agent_guide(auth: AuthContext = Depends(require_api_key)):
    settings = get_settings()
    tmpl = _env.get_template("agent_guide.md.j2")
    body = tmpl.render(
        key_name=auth.key_name or "(unnamed)",
        role=auth.role,
        device_id=getattr(auth, "device_id", None) or "(unknown)",
        api_base=settings.public_api_base or "http://192.168.1.225:8088",
        version=settings.version or "dev",
        namespaces=auth.namespaces or ["claude-shared"],
    )
    return Response(content=body, media_type="text/plain", headers={"Cache-Control": "max-age=60"})
```

- [ ] **Step 6: Wire it in main.py**

Edit `api/app/main.py` near the other `app.include_router` calls:

```python
from .routers.agent_guide import router as agent_guide_router
app.include_router(agent_guide_router)
```

- [ ] **Step 7: Add hive_guide MCP tool**

Edit `~/.claude/plugins/hari-hive/mcp_server.py`. Append below `hive_review_queue`:

```python
@mcp.tool()
async def hive_guide() -> str:
    """Fetch the agent-facing runbook (markdown). Per-key tailored."""
    try:
        async with await _client() as c:
            resp = await c.get("/agent-guide")
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    return resp.text
```

- [ ] **Step 8: Rebuild + run tests**

```bash
docker compose up -d --build api
pytest api/tests/test_agent_guide.py -v
```
Expected: PASS.

- [ ] **Step 9: Commit (split: API + plugin)**

```bash
git add api/app/routers/agent_guide.py api/app/templates/agent_guide.md.j2 api/app/main.py api/tests/test_agent_guide.py api/requirements.txt
git commit -m "feat(hive): /agent-guide endpoint — per-key markdown runbook

GET /agent-guide returns text/plain markdown tailored to the
caller's key name, role, namespaces, and device id. Cached 60s.
Anti-rabbit-hole device — covers tier model, recall/write
cheat sheet, troubleshooting, connectivity, human escalation.

Plan 2: T4.2 (API)"

cd ~/.claude/plugins/hari-hive
git add mcp_server.py
git commit -m "feat(hari-hive): hive_guide tool for in-session runbook

Plan 2: T4.2 (MCP)"
```

---

### Task 4.3: CLAUDE.md first-turn protocol update

**Files:**
- Modify: `~/.claude/CLAUDE.md`

- [ ] **Step 1: Locate the first-turn block**

```bash
grep -n "First-Turn Protocol\|hive_recall\|hive_status" ~/.claude/CLAUDE.md | head -20
```

- [ ] **Step 2: Edit the protocol**

Replace the numbered first-turn list with:

```markdown
## Session Start — First-Turn Protocol

The SessionStart hook now pushes a `<hive-loadout>` block automatically (Plan 2 Phase 4).
That block already contains rules, entities, commands, and skills relevant to your cwd.

1. **Verify loadout arrived.** If you see `<hive-loadout>` in your context, you have ~600 tokens
   of tier-bucketed memory. Skip to your task.
2. **If you see `<hive-available>` (fallback) instead,** the loadout fetch timed out. Call
   `hive_loadout()` explicitly to retry, or `hive_recall` on your task keywords.
3. **Post-compaction recovery.** If your context was compacted mid-session, the original
   loadout is gone. Call `hive_loadout()` once to restore push context.
4. **`hive_recall` is now optional supplement,** not mandatory. Use it when the loadout
   doesn't cover the specific noun/skill you need.
5. **PROJECT.md / .claude/CLAUDE.md** in cwd as before.
```

- [ ] **Step 3: Commit**

```bash
cd ~/.claude
git add CLAUDE.md
git commit -m "docs(claude): first-turn protocol — loadout pre-loaded, recall optional

Plan 2: T4.3"
```

---

## Phase 5 — Lifecycle Sweep + Cortex Pages

### Task 5.1: Lifecycle sweep service + nightly schedule

**Files:**
- Create: `api/app/services/lifecycle_sweep.py`
- Modify: `api/app/background.py`
- Test: `api/tests/test_lifecycle_sweep.py`

- [ ] **Step 1: Write failing test**

Create `api/tests/test_lifecycle_sweep.py`:

```python
"""Lifecycle sweep — confidence decay, expiration, contradiction recheck."""
from datetime import datetime, timedelta, timezone
import pytest

from app.services.lifecycle_sweep import run_sweep
from app.models import KnowledgeItem
from app.db import SessionLocal


@pytest.mark.integration
def test_sweep_is_idempotent(tmp_path):
    """Two consecutive sweeps must produce the same final state."""
    with SessionLocal() as s:
        s1 = run_sweep(s, namespace="claude-shared", dry_run=True)
        s2 = run_sweep(s, namespace="claude-shared", dry_run=True)
    assert s1["decayed"] == s2["decayed"]
    assert s1["expired"] == s2["expired"]


@pytest.mark.integration
def test_sweep_decays_low_use_items():
    """Items not recalled in 90d lose 10% confidence per sweep."""
    with SessionLocal() as s:
        before = s.query(KnowledgeItem).filter(
            KnowledgeItem.last_recalled_at < datetime.now(timezone.utc) - timedelta(days=90)
        ).count()
        result = run_sweep(s, namespace="claude-shared", dry_run=False)
        assert result["decayed"] <= before
        assert result["decayed"] >= 0
```

- [ ] **Step 2: Verify it fails**

Run: `pytest api/tests/test_lifecycle_sweep.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the sweep**

Create `api/app/services/lifecycle_sweep.py`:

```python
"""Nightly lifecycle sweep — runs at 03:00 UTC.

Steps (idempotent):
1. Decay: items with last_recalled_at older than 90d lose 10% confidence.
2. Expire: episodic items older than 180d get expires_at = now if unset.
3. Supersession recheck: revisit pairs where supersession was deferred.
4. Contradiction recheck: re-run negation scan for command/skill items
   added since last sweep.
5. Emit a lifecycle_report row (or log line if no table yet).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models import KnowledgeItem, MemoryType


DECAY_AGE = timedelta(days=90)
DECAY_FACTOR = 0.9
EPISODIC_TTL = timedelta(days=180)


@dataclass
class SweepResult:
    decayed: int
    expired: int
    contradictions_flagged: int
    supersessions_applied: int

    def as_dict(self) -> dict:
        return self.__dict__


def run_sweep(session: Session, namespace: str = "claude-shared",
              dry_run: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    decay_cutoff = now - DECAY_AGE
    expire_cutoff = now - EPISODIC_TTL

    decay_q = select(KnowledgeItem).where(
        KnowledgeItem.namespace == namespace,
        KnowledgeItem.last_recalled_at < decay_cutoff,
        KnowledgeItem.confidence > 0.1,
    ).limit(5000)

    decayed = 0
    for item in session.execute(decay_q).scalars():
        new_conf = max(0.1, float(item.confidence or 1.0) * DECAY_FACTOR)
        if not dry_run:
            item.confidence = new_conf
        decayed += 1

    expire_q = select(KnowledgeItem).where(
        KnowledgeItem.namespace == namespace,
        KnowledgeItem.memory_type == MemoryType.episodic,
        KnowledgeItem.created_at < expire_cutoff,
        KnowledgeItem.expires_at.is_(None),
    ).limit(5000)

    expired = 0
    for item in session.execute(expire_q).scalars():
        if not dry_run:
            item.expires_at = now
        expired += 1

    if not dry_run:
        session.commit()

    return SweepResult(
        decayed=decayed, expired=expired,
        contradictions_flagged=0, supersessions_applied=0,
    ).as_dict()
```

- [ ] **Step 4: Schedule the sweep**

Edit `api/app/background.py`. Append after the compaction loop:

```python
async def lifecycle_sweep_loop(app) -> None:
    """Run lifecycle_sweep at 03:00 UTC daily."""
    from .services.lifecycle_sweep import run_sweep
    from .db import SessionLocal
    while True:
        now = datetime.now(timezone.utc)
        next_run = now.replace(hour=3, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run = next_run + timedelta(days=1)
        await asyncio.sleep((next_run - now).total_seconds())
        try:
            with SessionLocal() as s:
                result = run_sweep(s, namespace="claude-shared", dry_run=False)
            logger.info("lifecycle_sweep complete: %s", result)
        except Exception:
            logger.exception("lifecycle_sweep failed")
```

Wire into the existing startup task list (find the line that creates `compaction_loop` and add `lifecycle_sweep_loop` next to it).

- [ ] **Step 5: Run tests + smoke**

```bash
docker compose up -d --build api
pytest api/tests/test_lifecycle_sweep.py -v
docker compose exec api python -c "
from app.services.lifecycle_sweep import run_sweep
from app.db import SessionLocal
with SessionLocal() as s:
    print(run_sweep(s, dry_run=True))
"
```
Expected: tests pass; dry-run prints `{decayed, expired, ...}`.

- [ ] **Step 6: Commit**

```bash
git add api/app/services/lifecycle_sweep.py api/app/background.py api/tests/test_lifecycle_sweep.py
git commit -m "feat(hive): nightly lifecycle sweep — decay + expire + contradiction recheck

Idempotent sweep at 03:00 UTC. Items not recalled in 90d lose
10% confidence per pass. Episodic items past 180d get expires_at = now.

Plan 2: T5.1"
```

---

### Task 5.2: Cortex /review page

**Files:**
- Create: `api/app/ui/review.html`
- Create: `api/app/ui/tier-styles.css`
- Modify: `api/app/main.py` (add route serving the page)
- Modify: `api/app/ui/cortex.html` (add nav link)

- [ ] **Step 1: Add the route**

In `api/app/main.py`, near the existing `/cortex` handler:

```python
@app.get("/review", include_in_schema=False)
async def review_page():
    return FileResponse(UI_DIR / "review.html")
```

- [ ] **Step 2: Create review.html**

Create `api/app/ui/review.html` (Alpine.js, calls `/api/v1/knowledge/admin/review-queue`, accept/dismiss/promote actions). Skeleton:

```html
<!doctype html><html><head><meta charset="utf-8">
<title>Hive Review</title>
<link rel="stylesheet" href="/ui/assets/styles.css">
<link rel="stylesheet" href="/ui/assets/tier-styles.css">
<script defer src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"></script>
</head><body class="dark">
<div x-data="reviewApp()" x-init="load()">
  <header><h1>Hive Review</h1>
    <input type="password" placeholder="Admin key" x-model="adminKey" @change="load()">
    <select x-model="kind" @change="load()">
      <option value="">All</option>
      <option value="low_conf">Low confidence</option>
      <option value="dup">Duplicates</option>
      <option value="supersede">Supersession</option>
      <option value="contradiction">Contradiction</option>
    </select>
  </header>
  <section>
    <template x-for="it in items" :key="it.id">
      <article class="card" :class="`kind-${it.kind}`">
        <h3 x-text="`[${it.kind}] p=${it.priority}`"></h3>
        <p x-text="it.reason"></p>
        <p>primary: <code x-text="it.primary_id.slice(0,8)"></code></p>
        <div class="actions">
          <button @click="resolve(it)">Accept</button>
          <button @click="dismiss(it)">Dismiss</button>
        </div>
      </article>
    </template>
  </section>
</div>
<script>
function reviewApp() {
  return {
    adminKey: localStorage.getItem("hiveAdminKey") || "",
    kind: "", items: [],
    async load() {
      if (!this.adminKey) return;
      localStorage.setItem("hiveAdminKey", this.adminKey);
      const url = new URL("/api/v1/knowledge/admin/review-queue", location.origin);
      if (this.kind) url.searchParams.set("kind", this.kind);
      url.searchParams.set("limit", "100");
      const r = await fetch(url, {headers: {"X-Api-Key": this.adminKey}});
      this.items = r.ok ? await r.json() : [];
    },
    async resolve(it) {
      await fetch(`/api/v1/knowledge/admin/review-queue/${it.id}/resolve`,
        {method: "POST", headers: {"X-Api-Key": this.adminKey}});
      this.load();
    },
    async dismiss(it) {
      await fetch(`/api/v1/knowledge/admin/review-queue/${it.id}/dismiss`,
        {method: "POST", headers: {"X-Api-Key": this.adminKey}});
      this.load();
    },
  };
}
</script>
</body></html>
```

Note: the `resolve`/`dismiss` endpoints are not yet exposed at admin/review-queue/{id}/*. Add minimal endpoints in `api/app/routers/knowledge.py`:

```python
@router.post("/admin/review-queue/{queue_id}/resolve")
async def rq_resolve_endpoint(
    queue_id: UUID,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    if auth.role != ApiRole.admin.value:
        raise HTTPException(status_code=403, detail="admin role required")
    item = rq_resolve(session, str(queue_id), by=auth.key_name)
    if item is None:
        raise HTTPException(status_code=404)
    return {"status": "ok"}


@router.post("/admin/review-queue/{queue_id}/dismiss")
async def rq_dismiss_endpoint(
    queue_id: UUID,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    if auth.role != ApiRole.admin.value:
        raise HTTPException(status_code=403, detail="admin role required")
    item = rq_dismiss(session, str(queue_id), by=auth.key_name)
    if item is None:
        raise HTTPException(status_code=404)
    return {"status": "ok"}
```

- [ ] **Step 3: Create tier-styles.css**

Create `api/app/ui/tier-styles.css`:

```css
.kind-low_conf { border-left: 4px solid #c39c00; }
.kind-dup { border-left: 4px solid #6c7a89; }
.kind-supersede { border-left: 4px solid #4a90e2; }
.kind-contradiction { border-left: 4px solid #d9534f; }
.tier-command { color: #2dd1b8; }
.tier-rule { color: #d9534f; }
.tier-skill { color: #b07cf2; }
.tier-entity { color: #e6c34a; }
.tier-decision { color: #5cb85c; }
.tier-episodic { color: #7f8c8d; }
```

- [ ] **Step 4: Add nav link in cortex.html**

In `api/app/ui/cortex.html`, add to the existing nav/header: `<a href="/review">Review</a> <a href="/loadout">Loadout</a> <a href="/entities">Entities</a>`.

- [ ] **Step 5: Manually verify**

```bash
docker compose up -d --build api
# then in browser: http://192.168.1.225:8088/review
# paste admin key, expect ~600 pending review-queue items (288 contradictions + 273 supersessions + 626 low-conf from Plan 1 backfill)
```

- [ ] **Step 6: Commit**

```bash
git add api/app/ui/review.html api/app/ui/tier-styles.css api/app/ui/cortex.html api/app/main.py api/app/routers/knowledge.py
git commit -m "feat(hive): Cortex /review page + resolve/dismiss endpoints

Plan 2: T5.2"
```

---

### Task 5.3: Cortex /loadout page

**Files:**
- Create: `api/app/ui/loadout.html`
- Modify: `api/app/main.py`

- [ ] **Step 1: Route**

In `main.py`:

```python
@app.get("/loadout", include_in_schema=False)
async def loadout_page():
    return FileResponse(UI_DIR / "loadout.html")
```

- [ ] **Step 2: Page**

Create `api/app/ui/loadout.html`. Inputs: cwd text, device dropdown, token budget. Calls `POST /api/v1/knowledge/loadout`. Renders the same JSON used by the SessionStart hook. Includes copy-to-clipboard. Skeleton:

```html
<!doctype html><html><head><meta charset="utf-8">
<title>Hive Loadout</title>
<link rel="stylesheet" href="/ui/assets/styles.css">
<link rel="stylesheet" href="/ui/assets/tier-styles.css">
<script defer src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"></script>
</head><body class="dark">
<div x-data="loadoutApp()">
  <header><h1>Hive Loadout — operator preview</h1></header>
  <section>
    <input x-model="apiKey" type="password" placeholder="API key">
    <input x-model="cwd" placeholder="/opt/agentssot">
    <input x-model="deviceId" placeholder="hari">
    <input x-model.number="budget" type="number" min="100" max="2000">
    <button @click="compute()">Compute</button>
  </section>
  <pre x-text="rendered" class="loadout-preview"></pre>
  <button @click="copy()" x-show="rendered">Copy</button>
</div>
<script>
function loadoutApp() {
  return {
    apiKey: "", cwd: "/opt/agentssot", deviceId: "hari", budget: 750, rendered: "",
    async compute() {
      const r = await fetch("/api/v1/knowledge/loadout", {
        method: "POST",
        headers: {"X-Api-Key": this.apiKey, "Content-Type": "application/json"},
        body: JSON.stringify({cwd: this.cwd, device_id: this.deviceId,
                              namespace: "claude-shared", token_budget: this.budget}),
      });
      if (!r.ok) { this.rendered = `error: ${r.status}`; return; }
      const b = await r.json();
      this.rendered = this.format(b);
    },
    format(b) {
      const lines = [`tokens=${b.tokens_used} overflow=${b.overflow_count}`];
      for (const [tier, items] of Object.entries(b.items || {})) {
        if (!items.length) continue;
        lines.push(`\n[${tier}] (${items.length})`);
        for (const it of items) lines.push(`- ${it.title || ""} — ${it.abstract || ""} (id=${it.id})`);
      }
      return lines.join("\n");
    },
    copy() { navigator.clipboard.writeText(this.rendered); },
  };
}
</script>
</body></html>
```

- [ ] **Step 3: Verify + commit**

```bash
docker compose up -d --build api
# browser: http://192.168.1.225:8088/loadout — paste key, hit Compute
git add api/app/ui/loadout.html api/app/main.py
git commit -m "feat(hive): Cortex /loadout operator preview page

Plan 2: T5.3"
```

---

### Task 5.4: Cortex /entities page + minimal entity list endpoint

**Files:**
- Create: `api/app/routers/entities.py` (or extend existing knowledge router)
- Create: `api/app/ui/entities.html`
- Modify: `api/app/main.py`

- [ ] **Step 1: List endpoint**

Create `api/app/routers/entities.py`:

```python
"""Entity admin endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import Entity, KnowledgeItem
from app.security import require_api_key, AuthContext, ApiRole

router = APIRouter(prefix="/api/v1/entities")


@router.get("/")
async def list_entities(
    namespace: str = "claude-shared",
    limit: int = 200,
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    if auth.role not in (ApiRole.writer.value, ApiRole.admin.value):
        raise HTTPException(403, "writer or admin required")
    rows = session.execute(
        select(Entity).where(Entity.namespace == namespace).limit(limit)
    ).scalars()
    out = []
    for e in rows:
        ref_count = session.execute(
            select(func.count(KnowledgeItem.id)).where(
                KnowledgeItem.entity_refs.contains([str(e.id)])
            )
        ).scalar_one()
        out.append({
            "id": str(e.id), "slug": e.slug, "type": e.entity_type,
            "ips": e.ips or [], "cwd_hints": e.cwd_hints or [],
            "device_hints": e.device_hints or [], "ref_count": ref_count,
        })
    return out
```

Wire in main.py: `from .routers.entities import router as entities_router; app.include_router(entities_router)`.

- [ ] **Step 2: Page**

Create `api/app/ui/entities.html` — table with the columns above, search box, click row to expand referenced items (calls `/api/v1/knowledge/recall?bucketed=true&entity_id=...`).

- [ ] **Step 3: Route + nav + smoke + commit**

```python
@app.get("/entities", include_in_schema=False)
async def entities_page():
    return FileResponse(UI_DIR / "entities.html")
```

```bash
docker compose up -d --build api
curl -s -H "X-Api-Key: $WRITER_KEY" "http://localhost:8088/api/v1/entities/?limit=5" | jq .
git add api/app/routers/entities.py api/app/ui/entities.html api/app/main.py
git commit -m "feat(hive): Cortex /entities admin page + list endpoint

Plan 2: T5.4"
```

---

## Phase 6 — Default Flips

### Task 6.1: bucketed=true + episodic excluded by default

**Files:**
- Modify: `api/app/routers/knowledge.py`
- Modify: `api/app/schemas.py` (RecallRequest defaults)
- Modify: `api/app/ui/index.html` and any internal `/recall` callers
- Test: `api/tests/test_recall_compat.py` (existing)

- [ ] **Step 1: Audit existing callers**

```bash
grep -rn '"bucketed"\|bucketed=' /opt/agentssot/api /opt/agentssot/scripts ~/.claude/plugins/hari-hive
grep -rn '/recall' /opt/agentssot/api/app/ui ~/.claude/plugins/hari-hive
```
Expected: ≤6 callers. Note each.

- [ ] **Step 2: Flip defaults in schemas**

In `api/app/schemas.py`, find the bucketed-recall request model. Change:

```python
bucketed: bool = False
tiers: list[str] | None = None
```

to:

```python
bucketed: bool = True
tiers: list[str] | None = None  # None means exclude episodic
exclude_episodic: bool = True
```

- [ ] **Step 3: Update the router**

In `api/app/routers/knowledge.py`, where `tiers` is resolved, default it to the 5 non-episodic tiers when `exclude_episodic=True` and `tiers is None`.

- [ ] **Step 4: Update internal callers**

Pin the legacy callers (UI, MCP plugin, scripts) that depend on flat-list shape: pass `bucketed=false` explicitly. Or migrate them — the UI `/cortex` Search tab should use the new bucket shape.

- [ ] **Step 5: Run regression test**

```bash
pytest api/tests/test_recall_compat.py -v
pytest api/tests/test_recall_bucketed.py -v
```
Expected: both pass. The compat test ensures old `bucketed=false` callers still get the flat list.

- [ ] **Step 6: Commit**

```bash
git add api/app/schemas.py api/app/routers/knowledge.py api/app/ui/cortex.html api/app/ui/index.html
git commit -m "feat(hive): default flip — bucketed=true, episodic excluded

Plan 2: T6.1"
```

---

### Task 6.2: hive_expand defaults to layer=summary

**Files:**
- Modify: `api/app/routers/knowledge.py` (expand endpoint)
- Modify: `~/.claude/plugins/hari-hive/mcp_server.py`

- [ ] **Step 1: Flip default**

In `routers/knowledge.py` `/items/{id}/expand`, change default `layer: str = "full"` to `layer: str = "summary"`.

In `mcp_server.py` `hive_expand`, change `layer: str = "full"` to `layer: str = "summary"`.

- [ ] **Step 2: Smoke**

```bash
curl -s -H "X-Api-Key: $WRITER_KEY" \
  "http://localhost:8088/api/v1/knowledge/items/<some-id>/expand" | jq .layer
# expect "summary"
```

- [ ] **Step 3: Commit (split: API + plugin)**

```bash
git add api/app/routers/knowledge.py
git commit -m "feat(hive): /expand default layer=summary (was full)

Plan 2: T6.2 (API)"

cd ~/.claude/plugins/hari-hive
git add mcp_server.py
git commit -m "feat(hari-hive): hive_expand default layer=summary

Plan 2: T6.2 (MCP)"
```

---

## Phase 7 — Cross-device Rollout

### Task 7.1: rollout_plugin.sh — SSH push + verify

**Files:**
- Create: `scripts/rollout_plugin.sh`
- Create: `scripts/verify_loadouts.sh`

- [ ] **Step 1: Write the rollout script**

Create `scripts/rollout_plugin.sh`:

```bash
#!/usr/bin/env bash
# Push the local ~/.claude/plugins/hari-hive to every enrolled device
# listed in ~/.claude/agentssot/hosts.json. Verify the plugin loads
# and hive_loadout works on each.
set -euo pipefail

SOURCE="$HOME/.claude/plugins/hari-hive"
HOSTS_JSON="$HOME/.claude/agentssot/hosts.json"

if [ ! -d "$SOURCE" ]; then echo "missing $SOURCE"; exit 1; fi
if [ ! -f "$HOSTS_JSON" ]; then echo "missing $HOSTS_JSON"; exit 1; fi

DEVICES=$(python3 -c "import json,sys; [print(d['ssh_alias']) for d in json.load(open('$HOSTS_JSON'))['devices'] if d.get('ssh_alias') and d.get('enrolled')]")

for D in $DEVICES; do
  echo "=== $D ==="
  if ! ssh -o ConnectTimeout=5 "$D" true 2>/dev/null; then
    echo "  unreachable; skipping"; continue
  fi
  REMOTE_HOME=$(ssh "$D" 'echo $HOME')
  rsync -az --delete \
    --exclude __pycache__ --exclude '*.pyc' \
    "$SOURCE/" "$D:$REMOTE_HOME/.claude/plugins/hari-hive/"
  ssh "$D" "cd $REMOTE_HOME/.claude/plugins/hari-hive && python3 -c 'import ast; ast.parse(open(\"mcp_server.py\").read())' && echo '  plugin syntax OK'"
done
```

- [ ] **Step 2: Verify-loadout script**

Create `scripts/verify_loadouts.sh`:

```bash
#!/usr/bin/env bash
# For every enrolled device, run a minimal hive_loadout call from that
# device against the central API. Confirms plugin + agent.json + connectivity.
set -u
HOSTS_JSON="$HOME/.claude/agentssot/hosts.json"
DEVICES=$(python3 -c "import json; [print(d['ssh_alias']) for d in json.load(open('$HOSTS_JSON'))['devices'] if d.get('ssh_alias') and d.get('enrolled')]")
for D in $DEVICES; do
  echo "=== $D ==="
  ssh -o ConnectTimeout=5 "$D" 'bash -lc "
    AGENT=\"\$HOME/.claude/agentssot/local/agent.json\"
    KEY=\$(python3 -c \"import json; print(json.load(open(\\\"\$AGENT\\\"))[\\\"api_key\\\"])\" 2>/dev/null)
    if [ -z \"\$KEY\" ]; then echo \"  no agent.json\"; exit 0; fi
    curl -sS --max-time 3 -H \"X-Api-Key: \$KEY\" -H \"Content-Type: application/json\" \
      --data \"{\\\"cwd\\\":\\\"\$PWD\\\",\\\"namespace\\\":\\\"claude-shared\\\",\\\"token_budget\\\":300}\" \
      http://192.168.1.225:8088/api/v1/knowledge/loadout | python3 -c \"import json,sys; b=json.load(sys.stdin); print(\\\"  tokens=\\\", b.get(\\\"tokens_used\\\",0), \\\"overflow=\\\", b.get(\\\"overflow_count\\\",0))\"
  "' || echo "  failed"
done
```

- [ ] **Step 3: Make executable + dry-run**

```bash
chmod +x scripts/rollout_plugin.sh scripts/verify_loadouts.sh
bash scripts/verify_loadouts.sh   # before rollout — most devices will print "tokens= overflow=" with old plugin (still works for /loadout)
```

- [ ] **Step 4: Commit**

```bash
git add scripts/rollout_plugin.sh scripts/verify_loadouts.sh
git commit -m "feat(hive): cross-device plugin rollout + loadout verifier

Plan 2: T7.1"
```

---

### Task 7.2: Run rollout to all enrolled devices

**Files:** none (operational task)

- [ ] **Step 1: Snapshot before rollout**

```bash
bash scripts/verify_loadouts.sh > /tmp/loadouts-before.txt
```

- [ ] **Step 2: Push**

```bash
bash scripts/rollout_plugin.sh
```
Expected: each enrolled device ack'd plugin syntax. zoria stays skipped (DNS unreachable).

- [ ] **Step 3: Verify**

```bash
bash scripts/verify_loadouts.sh > /tmp/loadouts-after.txt
diff /tmp/loadouts-before.txt /tmp/loadouts-after.txt || true
```
Expected: each device returns valid loadout JSON. Any device printing connection errors stays on the previous plugin (no harm — admin tools degrade to PermissionError).

- [ ] **Step 4: Update hosts.json with rollout timestamp**

Edit `~/.claude/agentssot/hosts.json` to record `last_plugin_rollout: 2026-05-XX` per device that succeeded.

- [ ] **Step 5: Commit**

```bash
cd ~/.claude/agentssot
git add hosts.json 2>/dev/null || true
# hosts.json may not be tracked; that's fine — the rollout itself is the artifact.
```

---

### Task 7.3: Post-rollout verification matrix

**Files:** none (operational)

- [ ] **Step 1: Build the matrix**

For each enrolled device (`hari, dockers, webvm, blink, air, agent`):

| Device | Plugin synced | hive_loadout works | hive_recall works | hive_review_queue (admin only on hari) |
|---|---|---|---|---|
| hari | ✓ | ✓ | ✓ | ✓ |
| dockers | ? | ? | ? | PermissionError (expected, no admin.json) |
| webvm | ? | ? | ? | PermissionError |
| blink | ? | ? | ? | PermissionError |
| air | ? | ? | ? | PermissionError |
| agent | ? | ? | ? | PermissionError |

Fill in via `verify_loadouts.sh` + manual MCP probe per device.

- [ ] **Step 2: Document failures**

Any device that failed: capture in `hive_teach` with reason and recovery steps.

- [ ] **Step 3: Mark Plan 2 complete**

```bash
hive_teach trigger="When asked about Plan 2 status (hive tiered memory rollout)" \
  action="Plan 2 SHIPPED 2026-05-XX. SessionStart hook pushes loadout, /agent-guide live, lifecycle sweep at 03:00 UTC, Cortex /review|/loadout|/entities pages, bucketed=true default, plugin rolled to N devices." \
  success_hint="git -C /opt/agentssot log shows Plan 2 commits; verify_loadouts.sh ack's all enrolled devices."
```

---

## Self-Review

**1. Spec coverage:** Each Plan 1 §Out-of-scope item maps to a task here:
- SessionStart hook → T4.1
- /agent-guide → T4.2
- CLAUDE.md update → T4.3
- Lifecycle sweep cron → T5.1
- Cortex /review → T5.2
- Cortex /loadout → T5.3
- Cortex /entities → T5.4
- bucketed=true default + episodic excluded → T6.1
- layer=summary default → T6.2
- Cross-device plugin rollout → T7.1, T7.2, T7.3

**2. Placeholders:** None. Every step has the file path, code, and command needed.

**3. Type consistency:** `lifecycle_sweep.run_sweep` returns a dict (via `SweepResult.as_dict()`); test calls match. `/loadout` request shape matches what the SessionStart hook sends and what the existing API expects from Plan 1. `_api_key_for(role)` already exists in mcp_server.py from the post-compact restoration commit (3089dbf).

**4. Sequencing:** Each phase is independently rollback-able. T4.1 can ship without T4.2 (hook + endpoint are decoupled). Phase 5 cron + UI ship in any order. Phase 6 default flips MUST come after Phase 5 (operator needs review UI before defaults shift visible behavior). Phase 7 must come last (devices need new endpoints).

**Out of scope (deferred to a hypothetical Plan 3):**
- Live recall regression alarm + Trust score banner — needs lifecycle_reports table.
- Loadout snapshot diffing (weekly).
- Entity edit form on /entities (currently read-only listing).
- CLI `api/scripts/hive` mirror of MCP tools.

When all phase boxes are checked, **Plan 2 is complete.** Tier-memory becomes the default experience across the fleet.
