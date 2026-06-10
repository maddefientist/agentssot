# Implementation Plan — Cortex Control Panel + Learning-Loop Repair

**Date:** 2026-06-04
**Author:** Madi (Opus, on-box at `hari:/opt/agentssot`)
**Companion to:** `docs/plans/AUDIT-2026-06-04-cortex-learning-loops.md`
**Workflow:** subagent-driven (chain.sh bricks). Operator opted in to **GPT-5.5** for high-level ops
(`gpt-orchestrate` / `gpt-review`), **MiniMax-M3** (`minimax-bulk`) for bulk reads/digests, with
GLM/Kimi/DeepSeek for the rest.
**Swap mechanism decision:** **Hybrid** — live health dashboard for everything; DB-backed runtime
overrides for the *hot* knobs (synthesis model, reranker model+URL, embedding/classifier endpoints);
all other settings stay `.env`-driven.

---

## 0. Verified ground truth (re-checked on-box, supersedes the remote audit where they differ)

| # | Finding | Status | Evidence (file:line) |
|---|---|---|---|
| ① | **Reranker endpoint is DOWN** — every recall silently degrades to vector-only | **P0 live** | logs: `RerankerProviderError: [Errno 111] Connection refused` ×5/30h via `reranker/ollama_provider.py:44` |
| ①b | `/health` + `/doctor` report reranker **`available`** while it's dead — `is_available` only checks *config present*, not reachability | **P0** | `ollama_provider.py:27`; `main.py:282-347` |
| ② | Synthesis model default is retired `qwen3.5:cloud` | **P1** | `settings.py:60` |
| ②b | Synthesis forms **zero clusters fleet-wide** → never reaches the model. Per-namespace, last-24h, needs ≥2 similar items | **P1 (root)** | `loop.py:166-206`, `clustering.py`, `settings.py:63-64` |
| ③ | Review queue has no drainer; `dup` kind never populated (`semantic_dedup_threshold=0.0`) | **P1** | `services/review_queue.py` (no drain), `knowledge.py:267-306`, `settings.py:106` |
| ④ | Feedback manual-only; `implicit_recalls` signal exists but underfed | **P1** | `reconciler.py:222`; `main.py:696-710` |
| — | Concept "1000 cap" = display limit, not storage | downgraded | `main.py:429`, `crud.py` |
| — | Synapse listener **running** on this host | downgraded | `synapse-listener.service` active |
| — | `madi-self` namespace exists (empty) | downgraded | synthesis logs 03:04 UTC |
| — | Reranker 4B/8B "mismatch" = intended two-tier design | not a bug | `reranker/router.py`, `settings.py:42-51` |

**Migration style:** raw SQL in `db/init/NNN_*.sql` (NOT Alembic). Next free number: **007**.
**Settings:** `get_settings()` is `lru_cache`d → providers built once at startup (`main.py:116-118`).
This is why a restart is currently required to change anything.

---

## 1. Architecture — the Hybrid runtime-config layer

The spine that makes the GUI useful. Everything else hangs off it.

### 1.1 New table — `db/init/007_runtime_config.sql`
```sql
CREATE TABLE IF NOT EXISTS runtime_config (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by  TEXT
);
-- Seed nothing. Absence of a row == "use the .env/Settings default".
```

### 1.2 Override resolver — new `api/app/runtime_config.py`
- `HOT_KEYS: frozenset[str]` — the **only** keys the GUI may override (allow-list, never arbitrary):
  `synthesis_model`, `synthesis_fallback_model`, `synthesis_similarity_threshold`,
  `synthesis_min_cluster_size`, `ollama_reranker_model`, `ollama_reranker_base_url`,
  `ollama_reranker_fast_model`, `ollama_reranker_fast_base_url`, `ollama_embed_model`,
  `ollama_base_url`, `classifier_model`, `classifier_base_url`, `semantic_dedup_threshold`.
- `load_overrides(session) -> dict[str,str]` — read all rows.
- `effective(settings, app_state, key)` — return override if present (typed-cast to the Settings
  field type), else `getattr(settings, key)`. Single chokepoint; **all hot-key reads go through it.**
- `set_override(session, key, value, by)` — validate key ∈ HOT_KEYS, type-check against the
  Settings field, upsert, return new effective value.
- `app.state.runtime_overrides: dict` cached copy, refreshed on every write.

### 1.3 Provider hot-swap — `api/app/providers.py` (or extend `main.py` startup)
- `rebuild_providers(app)` — re-run `build_embedding_provider` / `build_reranker_provider` /
  `build_llm_provider` against **effective** settings and atomically swap `app.state.*_provider`.
  Called after any write that touches a provider-affecting key.
- Synthesis reads `synthesis_model` fresh each run → route those reads through `effective(...)`
  in `loop.py:_run_synthesis_for_namespace` (no rebuild needed for synthesis).

### 1.4 Admin endpoints — new `api/app/routers/admin.py`
- `GET /admin/connections` — **live** probe of each provider:
  for each ollama endpoint, GET `{base_url}/api/tags` (2s timeout) → `{reachable, latency_ms,
  model_present: bool, error}`. This is the truth `/doctor` lies about today.
- `GET /admin/config` — list HOT_KEYS with `{key, effective, default, overridden: bool, updated_at, updated_by}`.
- `POST /admin/config` — body `{key, value}` → `set_override` → `rebuild_providers` if needed →
  return new `/admin/connections` row for the affected provider so the GUI reflects reality immediately.
- `DELETE /admin/config/{key}` — drop override (revert to `.env` default) + rebuild.
- Guard all writes behind the existing admin-key/namespace check used by `/keys`.

### 1.5 Frontend — new `api/app/ui/connections.html` + nav entry
- Add `href="/connections"` to `ui/_nav.html` and a route in `main.py` (`render_with_nav`).
- Panel: one card per provider (embedding / reranker-deep / reranker-fast / llm / classifier /
  synthesis). Each shows: live status dot (green/red from `/admin/connections`), base URL,
  current model (dropdown of models present in `/api/tags`), latency, "model present?" flag.
- Inline edit → `POST /admin/config` → optimistic update → re-poll the affected card.
- Reuse the existing HUD style (`hud.css`, the SSE pattern from the gateway status feeder).
- **Reranker-down alarm**: red banner at top of HUD + `/connections` when any provider
  `reachable=false`. This is the single most valuable pixel in the whole build.

---

## 2. Phase breakdown & subagent routing

> Orchestration model: **Opus writes every brief** (workers can't ask questions). Each phase is a
> chain brick. Parallel code phases run in **git worktrees** (`using-git-worktrees`). Verify after
> every dispatch — read changed files + run tests. Briefs follow the 10-point rubric.

| Phase | Work | Primary brick (model) | Review | Worktree? |
|---|---|---|---|---|
| **P0-arch** | Validate §1 design, resolve open questions, sequence the build | `gpt-orchestrate` (GPT-5.5) | — | no |
| **P1** | Runtime-config table + resolver + hot-swap + admin endpoints | `glm-implement` (GLM-5.1) | `kimi-review` → escalate `gpt-review` | yes |
| **P2** | `/connections` HUD page + nav + SSE wiring | `kimi-design` (Kimi-K2.6) | `kimi-review` | yes |
| **P3** | Restore reranker endpoint (ops) — validated via P2 panel | manual / `glm-quick` | self | no |
| **P4** | Synthesis: route model through `effective()`; clustering-scope rework | `deepseek-plan` → `glm-implement` | `gpt-review` (correctness-critical) | yes |
| **P5** | Review-queue drainer + dup detector + GUI wiring | `glm-implement` | `kimi-review` | yes |
| **P6** | Feedback auto-fire (implicit recall→use signal) | `glm-implement` | `kimi-review` | yes |
| **P7** | Hygiene: `/doctor` live-ping, stat count cap, Last-Active, reranker-name display | `glm-quick` | self | no |
| **P-pre** | Bulk read/digest of any subsystem a phase needs mapped first | `minimax-bulk` (MiniMax-M3) | — | no |

**Escalation rule:** P1 (provider hot-swap touches live request path) and P4 (concept correctness)
escalate review to `gpt-review`. Everything else stays free-tier `kimi-review`.

---

## 3. Worker briefs (10-point rubric — hand these to the chains verbatim)

### BRIEF P1 — Runtime-config layer + admin endpoints  → `glm-implement`
1. **Branch:** `feat/runtime-config-control-plane`
2. **Create:**
   - `/opt/agentssot/db/init/007_runtime_config.sql` (DDL in §1.1)
   - `/opt/agentssot/api/app/runtime_config.py` (§1.2 — `HOT_KEYS`, `load_overrides`, `effective`, `set_override`)
   - `/opt/agentssot/api/app/routers/admin.py` (§1.4 — 4 endpoints)
   - `/opt/agentssot/api/tests/test_runtime_config.py`
3. **Modify:**
   - `api/app/main.py:116-118` — after building providers, `app.state.runtime_overrides = load_overrides(...)`; add `rebuild_providers(app)` helper; `app.include_router(admin_router)`.
   - `api/app/reranker/__init__.py` / `embeddings/__init__.py` — `build_*` must accept effective base_url/model (read via `effective`), not raw settings.
4. **Migration:** 007 (raw SQL, idempotent `IF NOT EXISTS`). Container auto-runs `db/init/*` on fresh DB; for the live DB, apply via `docker compose exec db psql`.
5. **Logic:** §1.2–§1.4 exactly. Type-cast override strings to the Settings field type (bool/int/float/str) before use; reject unknown keys with 422.
6. **Imports/types:** `from .settings import get_settings`; `from .db import SessionLocal`; provider builders from their packages. No new deps.
7. **Acceptance:** (a) `POST /admin/config {synthesis_model: qwen3.5:397b-cloud}` then `GET /admin/config` shows `overridden=true`; (b) `GET /admin/connections` returns live `reachable` per provider; (c) override a bad reranker URL → card flips red within one poll; (d) `DELETE` reverts to `.env`; (e) pytest green.
8. **OUT OF SCOPE / forbidden:** do NOT allow override of any key outside `HOT_KEYS` (no DB URL, no secrets, no ports). Do NOT mutate `.env`. Do NOT remove the `lru_cache` on `get_settings`.
9. **Verify:** paste `curl -s localhost:8088/admin/connections | jq` output + pytest summary as proof.
10. **DoD:** green typecheck + tests; PR title `feat(admin): hybrid runtime-config control plane`.

### BRIEF P2 — `/connections` HUD page  → `kimi-design`
- **Branch:** `feat/hud-connections-panel`. **Create:** `api/app/ui/connections.html`. **Modify:**
  `api/app/ui/_nav.html` (+`href="/connections"`), `api/app/main.py` (route via `render_with_nav`).
- **Tokens/states:** reuse `hud.css` palette; one card/provider; green/red status dot; latency badge;
  model `<select>` populated from `/admin/connections.model_present` + `/api/tags`; loading + error +
  saved states. Red global banner when any `reachable=false`.
- **Data:** poll `/admin/connections` every 5s; `GET/POST/DELETE /admin/config`. No framework — match
  existing vanilla-JS HUD pattern (`hud.js`, `cortex-shell.js`).
- **Acceptance:** page lists all 6 providers with live dots; editing a model persists + re-polls;
  reranker-down shows the banner. **OUT:** no inline secrets, no arbitrary-key field — only HOT_KEYS.

### BRIEF P4 — Synthesis clustering-scope rework  → `deepseek-plan` then `glm-implement`, review `gpt-review`
- **Problem:** per-namespace/last-24h scoping means clusters never reach `min_cluster_size`. Fixing the
  model name alone is inert.
- **Two-part fix:**
  1. **Model:** route `settings.synthesis_model` reads in `loop.py:224` through `effective(...)` so the
     GUI/override (`qwen3.5:397b-cloud`) takes effect; change the **default** in `settings.py:60` too.
  2. **Scope (the real lever) — pick ONE in P0-arch, do not let the worker decide:**
     - **(a) Cross-namespace synthesis pass** for a configurable set of shared namespaces
       (`claude-shared` etc.) so the daily delta is pooled, not fragmented; OR
     - **(b) Rolling window** — widen `since` from 24h to N days (config) AND lower
       `synthesis_min_cluster_size` to 2 with the existing 0.65 threshold; OR
     - **(c) Backlog resynthesis** — one-time `full_resynthesis=True` sweep (`loop.py:162`) to mint
       concepts from the existing 6477 items, then rely on the rolling delta.
  - Recommended: **(c) once** to seed, then **(b)** for steady-state. (a) is a larger redesign — defer.
- **OUT:** do not change decay/promotion logic; do not touch `reconcile_concepts` signatures.
- **Review = `gpt-review`** (concept creation correctness, idempotency of the backlog sweep).

### BRIEF P5 — Curation loops  → `glm-implement`, review `kimi-review`
- **Drainer:** new `drain` path in `services/review_queue.py` + a synthesis-loop hook (or admin
  endpoint) that auto-resolves stale/duplicate `contradiction` rows (dedupe the 5×-queued primaries)
  and exposes `resolve/dismiss` through `/review` UI buttons (wire to existing `routers/review.py`).
- **Dup detector:** set `semantic_dedup_threshold` default >0 (e.g. 0.92) via override, and on ingest
  queue a `ReviewQueueKind.dup` when similarity ∈ [threshold, 1.0) instead of silently skipping.
- **OUT:** do not auto-DELETE knowledge; dup/contradiction resolution is propose→human-confirm via GUI.

### BRIEF P6 — Feedback auto-fire  → `glm-implement`
- When a recalled item is subsequently used as context (recall→ingest within a session window), emit an
  implicit `useful` signal feeding `reconciler.apply_feedback_signals` (`implicit_recalls`, cap 5).
  Raise the 17% feedback rate mechanically. **OUT:** no change to explicit `hive_feedback` semantics.

### BRIEF P7 — Hygiene  → `glm-quick`
- `/doctor`: replace `is_available` with the live-ping from `/admin/connections`; add `reranker_fast_model`.
- Stats: report true concept count (drop the 1000 display cap in the count path).
- Fix `AgentProfile` Last-Active update on recall. Reconcile reranker-name display vs `/api/tags`.

---

## 4. Sequencing & parallelism

```
P0-arch (GPT-5.5) ── decides P4 scope option, signs off §1
   │
   ├─ P1 (worktree) ──┐         providers + admin endpoints  [BLOCKS P2,P3,P4-model]
   │                  │
   ├─ P-pre minimax ──┘ (maps review_queue + ingest path for P5 in parallel)
   │
   ▼ after P1 merges
   ├─ P2 (worktree)  HUD panel        ─┐
   ├─ P3 reranker restore (ops)        ├─ can run concurrently
   ├─ P5 (worktree)  curation          │
   ├─ P6 (worktree)  feedback          ─┘
   ▼
   P4 (worktree)  synthesis  ── seed backlog AFTER reranker is healthy (P3) so embeddings/rerank are sound
   ▼
   P7 hygiene (cleanup pass)
```

**Critical path:** P0-arch → P1 → (P2 ∥ P3) → P4. P3 (reranker) gates P4's backlog sweep — don't
mint concepts on a degraded retrieval stack.

---

## 5. Verification & Definition of Done (whole effort)

- [ ] `/admin/connections` shows **live** reachability; reranker card green after P3.
- [ ] Synthesis run produces ≥1 cluster + ≥1 new concept on a seeded namespace (check `synthesis complete` stats `new>0`).
- [ ] `hive_status` synthesis model reads `qwen3.5:397b-cloud`.
- [ ] Review queue drains from the GUI; dup detector queues a `dup` row on a near-duplicate ingest.
- [ ] Feedback rate climbs above ~17% baseline over a day of normal use.
- [ ] All pytest suites green; no regression in `/recall` latency (vec 70ms / rerank ~17ms restored).
- [ ] Each PR: green typecheck, green tests, verification output pasted.

---

## 6. Risks & guardrails

- **Hot-swap on the live request path (P1):** atomic provider swap; never serve a half-built provider.
  If `rebuild_providers` throws, keep the old provider and surface the error in `/admin/connections`.
- **Override allow-list is the security boundary** — `HOT_KEYS` only. No secrets/DB/ports, ever
  (CLAUDE.md: no hardcoded secrets; validate all inputs). Run `security-reviewer` on `admin.py`.
- **Backlog resynthesis (P4c)** must be idempotent — re-running can't duplicate concepts; reconcile by
  embedding similarity (existing `reconcile_concepts`).
- **Reranker restore (P3)** is the prerequisite for trustworthy recall — do it before the synthesis seed.
- `.env` lives outside the container and in a perms-restricted dir → runtime overrides are DB-backed by
  design; document that overrides win over `.env` until deleted (drift note in `connections.html`).

---

## 7. Open questions for P0-arch (GPT-5.5) to close

1. P4 scope: confirm **(c)-seed-then-(b)-steady** vs full cross-namespace redesign (a).
2. Should overrides also persist back to `.env` on a "make permanent" action, or stay DB-only?
3. Admin auth: reuse `/keys` admin gate, or a dedicated control-plane key?
4. Does the gateway status SSE already carry provider health we can extend, or is a fresh
   `/admin/connections` poll cleaner? (Grep `gateway/feeders.py` + `gateway/service.py`.)
