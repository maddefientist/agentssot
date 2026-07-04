# Hive / Cortex — Architecture Handoff for Fable Review

**Date:** 2026-07-04
**Prepared by:** Madi (Opus, on-box at `hari:/opt/agentssot`)
**Audience:** Fable — asked to critique this system with advanced intelligence and make us better.
**Method:** Ground truth reconstructed by reading the live code (four parallel deep-map passes + direct reads), cross-checked against the design docs, the prior self-audit, hive teachings, and **live production behavior**. Every structural claim is anchored to `file:line`. Where a claim was independently re-verified against source it is tagged **[VERIFIED]**; otherwise it comes from a direct code read with the cited line.

> **Read this first:** The `README.md` describes maybe 40–60% of what is actually built and is wrong in several load-bearing places (auth model, default landing page, the actual recall path, chunking). **Do not trust the README. Trust this document + the code.** Design intent lives in `docs/plans/*` and in hive teachings, not in the README.

---

## 0. Why this handoff exists — the North Star

This is not "review a memory microservice." AgentSSOT / Hive / Cortex was built to become a **self-growing, self-driven, curious learning system** — a *living, on-LAN machine* that:

- **learns how we work** from every session (facts, decisions, preferences, skills, corrections),
- **consolidates** that raw experience into durable concepts and standing doctrine without us asking,
- **injects the right context at the right moment** (`<hive-loadout>` at session start, working-memory reconstruction after compaction),
- and thereby **cuts context-token spend and reduces per-LLM reliance** — the free local/cloud-Ollama tier does the memory work so the expensive frontier models only do the thinking that needs them.

The ambition is a **partner-grade AI infrastructure** that compounds: the more we use it, the more it knows, the less we have to re-explain, the cheaper and faster each session gets. A flywheel.

**The central question for Fable is therefore not just "is this code good?" but "does this architecture actually become that partner — a brain — or is it an accumulating warehouse that only looks like one?"** Our own 2026-06-04 self-audit (`docs/plans/AUDIT-2026-06-04-cortex-learning-loops.md`) landed a one-line verdict we want you to pressure-test:

> *"The cortex is a healthy warehouse, not yet a brain. Ingestion and retrieval — the hard 80% — work well. The loops that make it learn, curate, and grow are dormant, miscalibrated, or pointed at a dead model."*

Most of that audit's operational findings (F1, F3–F10) were repaired by 2026-06-10. **F2 — "there is no self-model; nothing for the system to grow *toward*" — remains open by intent.** That is squarely your territory.

---

## 1. What we are asking Fable to do

Please bring your full intelligence to bear on all of the following. Rank ruthlessly; we would rather have 15 true findings than 60 hedged ones.

1. **Poor habits / anti-patterns.** We have been building this largely with local/cloud chains and self-review; agent-authored commits with self-reported "tests green" are common. Call out the habits that will compound into rot: the copy-paste, the two-of-everything, the swallowed exceptions, the scattered magic numbers, the god-modules.
2. **Poor implementations & surface errors.** Concrete bugs, correctness gaps, race conditions, security holes we overlooked. The risk register in §9 is our own list — verify it, kill the false positives, and find what we missed.
3. **Design gotchas.** Places where the *design* (not just the code) fights the goal — e.g. two parallel recall/ingest stacks, two independent decay systems, a rotation scheme that is date-local while everything else is UTC.
4. **Optimizations & efficiency / anti-bloat.** The reranker is eating 11–28 seconds per recall in production (§8). We want the whole stack lean — is the software footprint justified, or are we carrying subsystems that don't earn their keep? Are we making best use of Postgres+pgvector, Ollama, and FastAPI, or fighting them?
5. **The meta-judgment (most important).** Does this architecture actually deliver the North Star (§0)? What is the shortest path from "warehouse" to "learning partner"? Is the self-model (F2) the right next investment, or is something more foundational broken? Where are we over-engineered vs where are we under-built for the vision?

---

## 2. Deployment topology & the layer cake

```
        ┌─────────────────────────────────────────────────────────────┐
        │  Agents (Claude Code fleet, HUD clients, other LLMs)         │
        │  via MCP stdio plugin  +  SessionStart/SessionEnd shell hooks│
        └───────────────┬───────────────────────────┬─────────────────┘
                        │ hive_* / cortex_* tools    │ WebSocket /gateway/ws
                        ▼                            ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  FastAPI service  (hari :8088, single container)             │
        │                                                              │
        │  CORE      recall / ingest / query / summarize               │
        │  CORTEX    per-agent working memory (task state, deltas)     │
        │  SYNTHESIS nightly Ollama loop: cluster→concept→doctrine     │
        │  CURATION  lifecycle decay, contradiction, review queue      │
        │  LOADOUT   cwd-aware tiered context pack (<hive-loadout>)     │
        │  GATEWAY   HUD nervous system: intent router + executors     │
        │  SYNAPSE   live cross-agent session/collision plane          │
        │  SYNC/WAL  per-device sync cursors + audit log               │
        │  CONTROL   runtime HOT_KEYS override plane                    │
        └───────┬───────────────────────┬──────────────────┬──────────┘
                │ SQLAlchemy (SYNC)      │ httpx            │ httpx
                ▼                        ▼                  ▼
        ┌───────────────┐        ┌──────────────┐   ┌───────────────┐
        │ Postgres 16   │        │ Ollama       │   │ Ollama        │
        │ + pgvector    │        │ embed / chat │   │ reranker 8B/4B│
        │ (data/postgres)│       │ / synthesis  │   │ (the bottleneck)│
        └───────────────┘        └──────────────┘   └───────────────┘
```

- **Single Postgres, single FastAPI container.** `docker-compose.yml`. Data in `./data/postgres`.
- **Everything orchestration-side is Ollama-only** (embed, chat/summarize, synthesis, classify, rerank). **No Anthropic client exists in the service** — [VERIFIED] by absence; the README's "Anthropic models are never called from this service" checks out. This is the per-LLM-reliance-reduction principle enforced at the infrastructure boundary.
- **Live scale (2026-07-04):** 7,439 knowledge items · 1,333 concepts · 122 namespaces · 140 active keys. Agent `device-hari-writer`: 936 recalls / 108 feedback.
- **The `<hive-loadout>` and `<cortex-working-memory>` blocks are emitted by external shell hooks** (installed under `~/.claude/`), not by this repo. The API only serves the JSON; the hooks wrap it. **This is a critical coupling: the repo cannot be understood in isolation from the hooks.**

---

## 3. Data model (the substrate)

SQLAlchemy models in `api/app/models.py` (529 lines); schema bootstrap in `db/init/00{1..4}*.sql`.

### 3.1 Typed-memory taxonomy — `MemoryType` (`models.py:54-77`)
`fact · decision · preference · skill · reference · correction · session_summary` (original) + the tiered additions **`command · rule · entity · episodic · doctrine`**. This taxonomy is the spine of loadout tiering and reranker routing. **It is sound but unenforced** — audit F6 found "weekly status update" filed as `decision`, etc. Discipline, not structure, is the gap.

### 3.2 `KnowledgeItem` (`models.py:192-283`) — the workhorse row
Carries far more than "content + embedding." Notable columns:
- **Neural/strength:** `strength`, `last_recalled_at`, `recall_count`, `positive_feedback`, `negative_feedback`, `status`.
- **Lifecycle (Plan-1 Phase-0):** `expires_at`, `superseded_by`, `confidence` (≥0.5 eligible for loadout — *lifecycle* confidence, distinct from `Concept.confidence`), `loadout_priority`, `last_classified_at`.
- **Tiered content layers (OpenViking-style):** `layer` (abstract L0 / summary L1 / full L2), `abstract`, `summary`, `verbatim` (permanent per-row flag suppressing L0/L1 synthesis for truth-critical memories — quotes, credentials).
- **Graph hints:** `entity_refs`, `rule_refs`, `cwd_hints`, `device_hints` (all JSONB).

### 3.3 `Concept` (`models.py:326-360`) — synthesized knowledge
`type` (mental_model / relationship / principle / skill), `scope` (global/project/device), `evidence_ids[]`, `confidence` (0.5 default, grows/decays), `version` + `parent_id` (version chain), `confirming_agents[]` (cross-agent consensus), skill triple (`trigger`/`action`/`success_hint`). `ConceptLink` (`:363-374`) is the associative graph (weight, co-occurrence).

### 3.4 Feedback & profile
`RecallEvent` (`:383`), `ConceptFeedback` (`:397`, useful/noted/wrong), `AgentProfile` (`:414`, learned strengths/preferences per agent key).

### 3.5 Curation
`DeletionLog` (`:432`, append-only audit), `ReviewQueueItem` (`:471`, kinds: low_conf/dup/supersede/contradiction; `attempts` convergence guard).

### 3.6 Ephemeral planes (not in models.py's main tables)
- **Cortex working memory** — `cortex_working_memory` + `cortex_deltas`, created idempotently in `cortex.py:132`.
- **Synapse** — `synapse_session` + `synapse_event`, both **`UNLOGGED`** (RAM-like, dropped on crash; intentional — live state only). `db/init/003_synapse.sql`. NOTIFY trigger in `004_synapse_notify.sql`.

---

## 4. Subsystem architecture (granular)

### 4.1 Core memory service — `db.py`, `embeddings/`, `chunking.py`, `main.py`, `crud.py`

**Connection pool (`db.py`, 33 lines — the entire pool story):**
- `pool_size=10`, `max_overflow=20` (30 max), `pool_timeout=10`, `pool_pre_ping=True` (`db.py:16-23`). Bounded by commit `565b62a`.
- Server-side `statement_timeout=30000` via libpq `options` (`db.py:14`). `idle_in_transaction_session_timeout` **deliberately not set** (`db.py:10-13`) because synthesis holds one session open across multi-second LLM calls — a GPT-5.5-review-driven revert (`7c7165a`).
- **The engine is fully SYNCHRONOUS.** There is no async DB driver. This is the root of the biggest structural issue (§4.1.4).

**Embeddings (`embeddings/`):** ABC + Ollama/OpenAI/Disabled providers. `embed_texts` is a naive loop over `embed_text` — **no true batching even for OpenAI** (`base.py:20`). Ollama provider can force `num_gpu:0` (CPU-only) to dodge GPU contention (`ollama_provider.py:28`). OpenAI provider never validates model/dim — mismatch only caught downstream.

**Chunking (`chunking.py`):** `chunk_text_semantic(max_chars=800)` — paragraph→sentence→hard split. **Only the legacy `/ingest` path chunks.** The tiered path stores full content as one row (see §4.1.2).

**`main.py` (2,371 lines) — god-module.** App construction + ~55 routes + settings-metadata dicts + type-coercion helpers + **two ~150-line inline bash scripts as f-strings** (`/enroll/bootstrap.sh` `:2054-2237`, `/enroll/install-plugin.sh` `:2246-2348`). Lifespan (`:137-246`) launches 5 background tasks (compaction, synthesis, lifecycle sweep, synapse reaper, synapse listener) with the shutdown `cancel/await/except` boilerplate repeated 5× verbatim.

#### 4.1.1 THE recall path — there are **two**, fully parallel
This is the single most important design fact in the codebase.

- **Legacy flat path:** `POST /recall` (`main.py:1155`, **sync** → threadpool) → `crud.recall` (`crud.py:586`). Weighted ranking in `_recall_knowledge_weighted` (`crud.py:350`): `score = similarity*0.6 + norm_strength*0.3 + recency*0.1` (`:383`), optional **hybrid RRF** fusing pgvector with Postgres FTS (`websearch_to_tsquery`/`ts_rank`, RRF k=60) (`:432-461`), then `_apply_reranker`. Flat response schema.
  - **Score inconsistency [from code read]:** it *orders* by `weighted_score` but *returns* raw cosine distance in the `score` field (`crud.py:385,471`) — the number the client sees is not the number used to rank. Reranker then overrides ordering entirely anyway.
- **Bucketed/tiered path:** `POST /api/v1/knowledge/recall` (`knowledge.py:424`, **async**, `bucketed=True` default) → `_recall_bucketed` (`:540`). Per-tier cosine + two-tier reranker (fast 4B for procedural tiers, deep 8B for nuanced) + per-tier diagnostics. **This is the actual default production path and the one the MCP `hive_recall` tool uses** — yet the README documents only the legacy path.

They share almost no code, have different dedup semantics, different reranker wiring, different response schemas, and even a different candidate multiplier (settings `2` vs inline fallback `3`, `knowledge.py:589`). **A fix to one does not propagate to the other.**

#### 4.1.2 THE ingest path — also two
- **Legacy `crud.ingest_batch` (`:88`):** secret-scan gate → chunk (`:162`) → **per-chunk `SELECT COUNT(*) WHERE content=chunk` exact-dup check (N+1)** (`:170`) → sync embed per chunk.
- **Tiered `ingest_tiered` (`knowledge.py:131`):** embed full content → classify via Ollama → `compute_layers` → **semantic dedup** (cosine ≥0.985 collapses silently; between `semantic_dedup_threshold` and 0.985 inserts + queues `dup` review) → supersession scan → contradiction scan. **Up to 4 separate `session.commit()` per single ingest** (`:304,332,382,411`) — non-atomic; a crash mid-way leaves a half-applied item.

**Three different definitions of "duplicate" live in the codebase:** exact-content (`ingest_batch`), semantic-cosine (`ingest_tiered`), and content-group batch (`dedup_knowledge_items`, `crud.py:1226`).

#### 4.1.3 Reranker — the live performance wound  [VERIFIED, live]
Every production `hive_recall` shows `vec=28–329ms` but **`rerank=11,448–28,201ms`**. The Qwen3-Reranker-8B does an N-candidate cross-encode; earlier work parallelized it (15.7s→8.7s) and cut the multiplier 3→2, but it remains the dominant cost on the hot path of *every session start*. See §8.

#### 4.1.4 Sync-SQLAlchemy-on-the-event-loop  [VERIFIED]
All `async def` routers (`knowledge`, `entities`, `doctrine`, `signals`, `wonder`, `review`, `adherence`) execute **blocking** `session.execute(...)` directly on the event loop — the sync engine has no async offload. The `asyncio.to_thread` wrapping covers only embed/classify calls, **not DB and not the reranker**. Confirmed: `knowledge.py:616` `reranker.rerank(...)` is bare while `:147/:160/:468/:560` embeddings are all `to_thread`-wrapped. So in the default recall path, an 11–28s rerank **blocks the entire event loop**, not just a worker thread — partially defeating the pool-bounding work. (The legacy path is `def`/threadpool, so it's fine there — the inconsistency itself is the smell.)

#### 4.1.5 Unauthenticated data-exposure endpoints  [VERIFIED]
`cortex_data` (`main.py:643`), `dashboard_stats` (`main.py:780`), `cortex_activity` (`:733`), `cortex_system_info`, `cortex_links` take only `Depends(get_session)` — **no auth**. Docstrings literally say *"No auth required"* / *"Public read-only."* They return concept titles/content, **agent keys**, raw recall **`query_text`**, feedback signals, namespace lists, and configured model names to any caller that can reach `:8088`. This directly contradicts the README's "API key authentication on all endpoints except /health." Whole-LAN readable.

### 4.2 Cortex working memory — `cortex.py` (566 lines)
Per-agent task state across session resets (decisions / pending / artifacts / deltas), upserted by `(namespace, agent_key, task_id)`. Injected as `<cortex-working-memory>` at session start; reconstructed budget-aware after compaction.
- **Deliberately weakened auth (`cortex.py:44-55`):** bypasses bcrypt (comment: "~3s per call") because it's hit every conversation turn. **Accepts *any* key starting with `ssot_`** — no DB lookup, no namespace ACL. Comment claims "constant-time comparison" but code uses plain `==` (`:51`) — misleading.
- **Upsert race (`:213-274`):** manual SELECT-then-INSERT/UPDATE, not `ON CONFLICT` — two concurrent turns for a new task can double-insert or collide.
- Table-creation failure is swallowed (`:192`, `logger.warning`); endpoints then 500 at runtime.
- Duplicated `CortexTaskOut` construction blocks (`:333` vs `:508`).

### 4.3 Synthesis / doctrine / loadout — the learning loop
This is where "warehouse" is supposed to become "brain."

**Synthesis loop (`synthesis/loop.py`, 403 lines):** nightly (3:00 UTC), per-namespace, offloaded via `to_thread`.
- **`SYNTHESIS_ENABLED` defaults to `False`** (`settings.py:59`) — the growth loop is *off* unless explicitly enabled (it is ON live, model overridden to `qwen3.6:27b`).
- Pipeline: gather recent embedded KIs → **greedy single-pass clustering** (`clustering.py`, order-dependent, threshold 0.65) → skill fast-track (taught skills bypass min-cluster-size) → per-cluster LLM synthesis in a **SAVEPOINT** (so one failure doesn't poison the txn — a real `InFailedSqlTransaction` fix) → `reconcile_concepts` → auto-complete stale recall sessions (implicit-useful credit) → feedback signals → decay → agent-profile build → **doctrine promotion**.
- **`_run_synthesis_for_namespace` is ~200 lines / 6 concerns** — prime refactor target.
- **Enable flag checked only at startup** (`main.py:161`), not per-cycle — a runtime toggle needs a restart.
- **Preflight resilience (recent work, `synthesis/preflight.py`):** before any run, validates the effective models against Ollama `/api/tags`; degrades (drop to fallback / drop fallback), or skips+alerts if models unavailable/Ollama unreachable; wrapped so preflight can never crash the loop. This is *good* — the defensive pattern we want more of.

**Reconciliation (`synthesis/reconciler.py`, 320 lines):** reinforce (+0.05, cap 1.0, content replaced only if longer) / contradict (old −0.2, tag superseded, null embedding, new versioned child) / new. Feedback signals: useful ×0.15, noted ×0.05, implicit-recall ×0.02 (capped); ≥3 confirming agents → `consensus`; `wrong` → `contested`. Decay: −decay_rate down to floor, then tag `dormant`.
- **Smell (`:57`):** `except (ValueError, Exception)` — `Exception` supersets `ValueError`; swallows *all* DB errors as "no match."
- **All confidence deltas are hardcoded magic numbers** scattered across the module — none configurable.

**Doctrine promotion (`synthesis/promotion.py`):** Concepts with confidence ≥0.8 and type principle/mental_model are mirrored into `memory_type="doctrine"` KnowledgeItems at `loadout_priority=4`. **Loads the entire concept table into memory each run** (`:70`) and filters in Python — O(all concepts) per namespace per night.

**Loadout assembly (`services/loadout.py`, 147 lines):** the context pack that reduces per-session token spend.
- **Rules (priority 5):** ALL rules in namespace, unconditionally, every session.
- **Doctrine (priority 4):** **one item/day**, `rotation_index = sha256(date.today()) % count`, selected via `OFFSET rotation_index LIMIT 1` over `created_at`.
  - **Two gotchas [from code read]:** (a) `date.today()` is **server-local**, mixing local-date rotation with UTC everywhere else; (b) OFFSET-over-`created_at` means inserting/removing any doctrine item **shifts every index** — it's not a stable per-item schedule.
- **Other tiers** only surface when cwd/entity refs match. cwd-hint match is **prefix OR substring** (`h in cwd_norm`, `:40`) — the substring branch can false-match unrelated paths.
- Greedy token-budget packing (`_TOKEN_PER_CHAR=0.27` magic constant); priority-5 rules pack before priority-4 doctrine, so a large rule set crowds everything else out.

### 4.4 Curation loops — `services/` (lifecycle, contradiction, review_queue)
- **Two independent decay systems** [design gotcha]: (1) *Concept* decay in synthesis (updated_at-based, −0.02, floor 0.15); (2) *KnowledgeItem* decay in `lifecycle_sweep.py` (last_recalled_at-based, ×0.9, floor 0.1, 90-day age). Different entities, different math, different floors — trivially confused.
- **`lifecycle_sweep` steps 3 & 4 (contradiction/supersession recheck) are documented stubs returning 0** (`:6-7,75`). Advertised, not implemented. And the sweep is **hardcoded to namespace `claude-shared`** (`background.py:106`) — other namespaces never decay or expire.
- **Contradiction detection (`contradiction.py`) is purely lexical** — regex negation patterns (`never/don't/forbidden/...`). "never" inside a benign sentence false-fires. No semantics.
- **Review queue (`review_queue.py`, 338 lines)** is the human/agent curation surface. Documents real incidents in prose: the "3-day-loop bug" (rows re-queued forever, fixed by `attempts` convergence), the backfill duplicate-queue (same pair queued 5×). `audit_supersede` reverses false-positive supersessions via embedding cosine ≥0.80 (undoing the ×0.3 confidence penalty with ÷0.3). Heavy raw-SQL + ORM mix.

### 4.5 Gateway / HUD — the "nervous system" (`gateway/`, ~900 lines)
One WebSocket command channel (`/gateway/ws`) + one SSE status stream (`/gateway/sse/status`), a hybrid intent router, and a registry of swappable "brain-region" executors.
- **Flow:** WS message → load hive-backed session history → classify (explicit intent → regex rules → Ollama classifier `deepseek-v4-flash:cloud`, 4s timeout, `format:json` → default `chat-local`) → dispatch to executor → stream Events back → persist reply. Session state lives in Postgres `gateway_session`, **deliberately outside the knowledge graph** so chatter never pollutes recall.
- **Executors:** `chat-local`, `orchestrate` (falls up a ladder opus→deepseek-pro→deepseek-flash→local, emits visible `fallover` events — "the reliability heart"), `hive-tool`, `dispatch`, `briefing`.
- **Half-finished by intent [from code read]:** `teach_fn=None` hardwired in prod (`wiring.py:269`) → HUD "remember that X" hits a dead-end stub even though the router *routes* teach phrasings to it; `DeferredBriefingExecutor` is a placeholder; `DispatchExecutor` v1 can't reach `chain.sh` in-container and returns an apology string; `chains=None` in the status snapshot (`.chain/` not mounted). A `kind=="chain"` orchestrate branch (`wiring.py:118`) is dead (no ladder rung defines it).
- **"Degrade fast" change (`2f9e3de`):** classifier timeout 8s→4s (env `MADI_CLASSIFIER_TIMEOUT`); on timeout the bare `except` swallows to `chat-local`. Caveat: httpx `timeout=4.0` is per-phase, not total wall-clock — no `asyncio.wait_for` around the whole classify, so the "bounds worst case to ~4s" comment is optimistic.
- **Blocking DB on the loop** here too: `SqlBackend` uses sync sessions inside `async def`.

### 4.6 Synapse — live cross-agent awareness plane (`synapse/`, NOT in README)
Answers "which agents are active right now, where, and are two about to edit the same file?" (**collision detection**). Separate from the durable knowledge graph. Both tables `UNLOGGED`.
- REST: `/session`, `/heartbeat`, `/event`, `/active`, `/collisions`, SSE `/stream`.
- **Listener** = Postgres LISTEN/NOTIFY fan-out to per-worker subscriber queues (drop-oldest + `{"kind":"overflow"}` marker on full queue); exponential reconnect backoff.
- **Critical hidden coupling:** `POST /event` never calls NOTIFY — the notification comes from a **Postgres trigger** `synapse_event_notify()` in `db/init/004_synapse_notify.sql` (cwd truncated to 200 chars to stay under the 8000-byte NOTIFY limit). **If `004` isn't applied, SSE `/stream` silently emits nothing while REST still works.** Python and SQL must deploy together.
- **10-minute TTL is triplicated** (reaper SQL, wiring `_synapse_activity`, `/active` default) — change one, the HUD "active" count and the reaper diverge.
- **`_handle_notification` uses `asyncio.get_event_loop()`** (deprecated) and doesn't retain the `create_task` result (`listener.py:182-183`) — the fan-out task can be GC'd mid-flight (asyncio footgun).

### 4.7 Sync / WAL — `sync.py` (391), `wal.py` (138)
- **`sync.py` is NOT replication.** It is a passive pull-model sync-cursor + duplicate detector; clients resolve conflicts themselves. Off by default (`SYNC_TRACKING_ENABLED=false`). No push, no transport, no merge.
  - **Semantic bug [from code read]:** `_detect_conflicts` groups by `source` (a path/URL) but returns it as `device_ids` — "conflict" really means "same content from two sources," not cross-device.
  - **Dead code:** `limit=0` "count trick" (`:361`) then an unused `_get_checkpoint` (`:363`) with a comment describing unimplemented intent.
  - `sync_conflict_window_hours` setting exists but is **never used** — the detector hardcodes 24.
  - Reads a module-import `settings` snapshot (`:29`) — diverges from the live-mutated Settings singleton used everywhere else; also not in HOT_KEYS.
- **`wal.py` is a best-effort audit log, explicitly NOT durability.** Fire-and-forget JSONL, daily-rotated. **Silently drops events** on any dir/serialize/append failure (warning only) while the user write proceeds — so gaps ≠ "no writes happened." Key-**name**-based redaction only (a secret under a non-listed key, or in free-form `content`, is written verbatim modulo 2000-char truncation). `threading.Lock` only guards within one process; multi-worker appends interleave. `prune_older_than` uses mtime, not the filename date.

### 4.8 MCP tool surface — `plugin/mcp_server.py` (1,144 lines)
Thin stdio FastMCP proxy (`hari-hive`) — 28 tools, formatting + HTTP only. **This is the primary agent-facing API.** Core: `hive_recall/query/ingest/stats/summarize`. Cortex: `cortex_state/reconstruct/update`. Learning: `hive_feedback/teach/profile/session_end`. Tier: `hive_expand/loadout/supersede/expire/promote`. Admin: `hive_create_namespace/key`, `hive_delete_items/dedup/review_queue`. Ops: `hive_status/guide/doctor`.
- **Security smells [from code read]:** silent privilege fallback to the *writer* key when `admin.json` is missing (`:72`) — admin ops then 403 opaquely; `admin_api_key` used as the *default* key for all tools (`:33`) → if present, even plain recall runs as admin; **`hive_create_key` returns the plaintext key in tool output** (`:621`) → transcript/log leakage.
- `hive_review_queue` silently prints only the first 20 of `limit`(50). Hardcoded LAN IP default `http://192.168.1.225:8088`.

### 4.9 Security, gates, control plane
- **Auth (`security.py`):** bcrypt-hashed keys, **linear O(N) bcrypt scan** on cache miss (`_lookup_api_key:108`), backed by a 1h / 1024-entry TTL cache keyed by `sha256(plaintext)`.
  - **Revocation gap [from code read]:** deactivating/re-scoping a key does **not** evict the cache — a revoked key keeps working up to 3600s unless every key-mutation endpoint calls `clear_auth_cache()`. The primitive exists; it relies on discipline. **Verify the call sites.**
  - **Negative-auth DoS amplification:** invalid-key misses are never cached → every request with a bad key forces a full O(N) bcrypt scan. No rate limiting.
- **Ingest secret-scan gate (`secret_scanner.py`, ~30 patterns):** rejects likely secrets with a 422 that names patterns, never values (good). **False-positive magnets:** `base58_private_key {87,88}` and `private_key_hex 0x[a-f0-9]{64,}` will reject legit long tokens / hashes / data-URIs and block the *whole* batch.
- **Read-side sanitizer (`output_sanitizer.py`):** neutralizes prompt-injection in recalled content before it reaches a model (imperative-prefix patterns, chat-template tokens, zero-width/bidi strip, remote-markdown-image defang). Good defensive posture.
  - **Coverage inconsistency [VERIFIED, corrected]:** the legacy `/recall` path sanitizes **only the `snippet` field** (`main.py:1176`, default `snippet_keys=("snippet",)`); the bucketed path sanitizes `content/abstract/summary/full_content` (`knowledge.py:530`). The MCP `hive_recall` uses the bucketed path, so it *is* covered — this is **not** a live exploit (an earlier draft overstated it), but the two paths having different sanitization coverage is a real inconsistency and a latent gap if a client renders a non-`snippet` field off the legacy path.
- **Runtime control plane (`runtime_config.py`):** 17 HOT_KEYS (synthesis/reranker/ollama/classifier/dedup/supersession tuning) persistable to the `runtime_config` table, validated (URL scheme, numeric ranges), applied by mutating the `@lru_cache`d Settings singleton in place via `object.__setattr__`. **Two parallel validation systems** exist for `/admin/settings` (HOT_KEYS via runtime_config vs `_coerce_setting`/`_SETTING_RANGES`).

---

## 5. The learning-loop honest assessment (warehouse → brain)

Mapping the North Star (§0) onto reality, updated from the 2026-06-04 audit:

```
INGEST     ✅ alive (two paths, but alive)
STORE      ✅ 7,439 KIs / 1,333 concepts, pgvector
RECALL     ✅ returns good results ... ⚠️ but 11–28s dominated by rerank
───────────────────────────────────────────────────────────────
FEEDBACK   ⚠️  wired, was ~17% fed; auto-complete-stale added to reduce
               reliance on agent discipline — needs re-measurement
SYNTHESIS  ✅  repaired (was pointed at retired model); ON, qwen3.6:27b,
               now preflight-guarded. But enable-flag needs restart to toggle.
CURATE     ⚠️  review queue converges now; contradiction is lexical-only;
               sweep steps 3&4 are stubs; sweep is single-namespace
ASSOCIATE  ⚠️  synapse exists (collision plane) but concept-graph edges only
               form for matched concepts, never for brand-new ones
SELF-MODEL 🔴  still does not exist (F2) — nothing for the loop to grow toward
```

**The honest read:** the plumbing that repaired the audit's P0/P1 findings is largely in place, and the preflight/alerting resilience work is genuinely good. But the system still **accumulates faster than it curates**, curation is partly lexical/stubbed, and — most importantly for the vision — **there is still no representation of the system itself**: no identity, no behavioral doctrine it maintains about *how it should work with us*, no improvement goals it revises. It learns *facts about our projects*; it does not yet learn *how to be a better partner*. That is the gap between "warehouse" and "brain," and it is a design question, not a bug.

---

## 6. Config drift & doc drift register

| Claim (README/design doc) | Reality (code) |
|---|---|
| `/` serves Browse/Search/Admin dashboard | `/` serves the Madi HUD; legacy dashboard moved to `/classic` (`main.py:537`) |
| "Auth on all endpoints except /health" | `cortex_*`, `dashboard/stats`, enrollment scripts are unauthenticated [VERIFIED] |
| API ref documents `/recall`,`/query`,`/ingest` | The default path is the undocumented `/api/v1/knowledge/*` bucketed surface |
| "Auto-chunked to ~800 chars" | Only the legacy `/ingest` path chunks; tiered stores full content |
| `RERANKER_CANDIDATE_MULTIPLIER` default 3 | Code default 2 (`settings.py:40`); bucketed path falls back to 3 inline |
| Design doc: `qwen3:30b-a3b`, `vector(4096)` | Live synthesis `qwen3.6:27b`; embedding dim differs (nomic-embed-text) |
| `.env` model values | DB runtime overrides mask them — correct until a clean DB wipe silently reverts to broken defaults |
| Synthesis model (settings default) `qwen3.5:397b-cloud` | Live override `qwen3.6:27b` |

**Import-style inconsistency:** some routers use absolute `from app.db`, others relative `from ..db` — mixed across one package.

---

## 7. Poor-habit patterns (the compounding rot)

For Fable to weigh — these recur across subsystems and will compound:

1. **Two-of-everything.** Two recall stacks, two ingest stacks, three dedup definitions, two decay systems, two settings-validation systems. Almost certainly the biggest maintenance liability.
2. **Swallow-and-continue.** Broad `except Exception: pass|return None|[]` in dozens of places (embeddings→None silently makes an item un-recallable; FTS failure; rerank failure; synapse fan-out; sync bootstrap; WAL; MCP grant). Many log only at `debug`. Failures are invisible with no metric/WAL entry — debugging a silently-degraded system will be brutal.
3. **Scattered magic numbers.** Confidence deltas (0.05/0.2/0.15/0.02), thresholds (0.6/0.8/0.985), TTLs (10min triplicated), token-per-char 0.27, caps (2/2/5), consensus ≥3 — none centralized despite `config.py`'s stated purpose.
4. **God-modules & copy-paste.** `main.py` 2,371 lines with embedded bash; `crud.recall` ~240 lines / 5 near-duplicate branches; `ingest_tiered` ~290 lines / 6 concerns; 5× repeated shutdown boilerplate; duplicated `CortexTaskOut` blocks.
5. **N+1 everywhere.** Per-entity ref-count (also namespace-unscoped), per-ancestor concept-history walk, per-chunk dup COUNT, per-id delete loops, O(N) bcrypt loops in auth *and* enrollment.
6. **Undocumented debt.** Zero `TODO/FIXME/HACK` markers anywhere — the debt lives in behavior and prose docstrings, not annotations. Don't expect a trail.
7. **Self-reported correctness.** Much was agent-authored with self-reported "tests green" (e.g. the whole gateway landed by Madi; commit author "Tycho Latency-Cap"). Treat "verified" claims in git history skeptically.

---

## 8. Live performance story (measured, not theorized)

- **Reranker is 97%+ of recall latency.** `vec 28–329ms · rerank 11,448–28,201ms` in production today. On the hot path of every session start.
- **It blocks the event loop** in the default async recall path (§4.1.4) — under concurrent load, recalls serialize on the reranker.
- **Candidate width × per-candidate cost** is the driver: an N-candidate cross-encoder over an 8B model. Prior mitigations: ThreadPoolExecutor parallelization, multiplier 3→2, two-tier (4B fast / 8B deep) routing. Pending idea in working memory: a single batched logit forward-pass (one `/api/generate` instead of N calls) and/or raising `OLLAMA_NUM_PARALLEL`.
- **Open question for Fable:** is a cross-encoder reranker the right tool at this corpus size at all, given hybrid RRF already fuses keyword+vector? What is the precision gain actually buying us vs. 11–28s of latency and event-loop blocking on *every* session start? This is the sharpest efficiency-vs-value call in the system.

---

## 9. Consolidated risk register (our list — verify, prune, extend)

Ranked by our estimate of severity. **Please challenge the ranking.**

| # | Severity | Finding | Anchor | Status |
|---|---|---|---|---|
| 1 | High | Reranker 11–28s dominates every recall; blocks event loop in async path | `knowledge.py:616` | [VERIFIED] |
| 2 | High | Unauthenticated endpoints leak concept content, agent keys, recall query_text | `main.py:643,733,780` | [VERIFIED] |
| 3 | High | Auth cache not invalidated on key revocation → up to 1h window | `security.py:24,108` | code read |
| 4 | High | Two parallel recall + two ingest stacks; fixes don't propagate | `crud.py` vs `knowledge.py` | code read |
| 5 | Med-High | Cortex auth trusts any `ssot_`-prefixed key, no ACL | `cortex.py:53` | code read |
| 6 | Med-High | MCP: plaintext key in tool output; silent admin→writer fallback; admin-as-default | `mcp_server.py:33,72,621` | code read |
| 7 | Med | Sync-SQLAlchemy on the event loop across all async routers | `db.py`; `knowledge.py` etc. | [VERIFIED] |
| 8 | Med | Non-atomic tiered ingest (up to 4 commits) | `knowledge.py:304,332,382,411` | code read |
| 9 | Med | Doctrine rotation: server-local date + unstable OFFSET index | `loadout.py:87` | code read |
| 10 | Med | lifecycle sweep hardcoded to `claude-shared`; steps 3&4 are stubs | `background.py:106`, `lifecycle_sweep.py:75` | code read |
| 11 | Med | Negative-auth DoS: bad keys force full O(N) bcrypt scan, uncached | `security.py:108` | code read |
| 12 | Med | Synapse SSE silently dead if `004_notify.sql` not applied | `db/init/004` | code read |
| 13 | Med | secret_scanner false-positive magnets block legit batches | `secret_scanner.py:152,194` | code read |
| 14 | Low-Med | Two independent decay systems, easily conflated | `reconciler.py` vs `lifecycle_sweep.py` | code read |
| 15 | Low-Med | Sanitizer coverage differs between recall paths (latent, not live) | `main.py:1176` vs `knowledge.py:530` | [VERIFIED] |
| 16 | Low-Med | Contradiction detection is lexical-only (regex negations) | `contradiction.py` | code read |
| 17 | Low-Med | reconciler `except (ValueError, Exception)` swallows all DB errors | `reconciler.py:57` | code read |
| 18 | Low | WAL drops events silently; key-name-only redaction | `wal.py` | code read |
| 19 | Low | sync.py device/source conflation, dead count-trick, unused window setting | `sync.py:264,361` | code read |
| 20 | Low | Score returned ≠ score ranked on (legacy recall) | `crud.py:385` | code read |

**Cross-cutting:** swallowed exceptions (§7.2), scattered magic numbers (§7.3), N+1s (§7.5), README drift (§6).

---

## 10. Open design questions for Fable (the ones we most want answered)

1. **Warehouse → brain: what is the shortest path?** Is the self-model (F2) the right next build, or is something more foundational (curation convergence, taxonomy enforcement) the real blocker?
2. **The self-model (F2).** Should the system hold a *first-class* representation of itself — identity, behavioral doctrine, improvement goals — that the synthesis loop reads and updates? Or is that over-reach that will accrete noise? If yes, what's the minimal viable shape?
3. **Does the loadout actually save context spend?** We inject rules (all, every session) + one rotating doctrine + cwd-matched tiers. Is this the right economy, or are we spending tokens on low-value rules while the good doctrine rotates past only 1/day?
4. **Is the reranker earning its 11–28s?** (§8) Cross-encoder vs. hybrid-RRF-only vs. a cheaper/batched rerank.
5. **Consolidate the two-of-everything?** Is unifying the recall/ingest stacks worth the migration risk, or do the two paths serve genuinely different needs?
6. **Is the taxonomy enforceable?** It's sound but unenforced (audit F6). Can classification discipline be made mechanical without a heavy per-ingest LLM cost?
7. **Efficiency / anti-bloat.** Are we carrying subsystems that don't earn their keep (sync? WAL? parts of the gateway that are stubbed)? Where is the software stack fighting us vs. serving us?
8. **The meta-judgment.** Given the North Star of a living, learning, partner-grade infrastructure — where would *you* invest next, and what would you rip out?

---

## 11. Madi's strategic assessment (for Fable to adjudicate, not just accept)

The on-box agent's own read of the system, offered as a position for Fable to pressure-test — agree, refute, or sharpen. **Please tell us where this is wrong.**

- **Impressive ≠ serving.** As an engineering artifact the system is substantial; as a *partner that makes us faster and cheaper*, we have no evidence it's net-positive. An 11–28s reranker on every session start is the literal opposite of "reduce context spend / reduce per-LLM reliance." **Nobody has measured whether the flywheel actually turns.** That's the question under all the others.
- **The disease is accretion, not any single bug.** Two recall stacks, two ingest stacks, three "duplicate" definitions, two decay systems, two validators, dead/stubbed branches — the signature of many parallel agent sessions each solving a local problem with no custodian of the whole. The risk register (§9) is symptoms; this is the cause. It's the failure mode the cheap-parallel-worker delegation model optimizes *for* and doesn't defend *against*.
- **Self-model (F2) is premature.** It reads from a corpus that accumulates faster than it curates. Give the system a sense of self on an unclean substrate and it accretes confident noise *about itself* — worse than noise about projects, because it acts on it. Earn curation trust first.
- **Firing a bug-hunt at it now risks deepening the habit.** If findings go back through the same parallel-worker pipeline, they get fixed *beside* the old code, spending frontier tokens to worsen the accretion. The handoff's real value is forcing the decision: **consolidate, or keep accreting.**
- **Proposed sequence:** (1) measure the flywheel for one honest week — is loadout cutting spend? are recalls useful? is rerank buying precision over RRF-only? (2) pick the reranker fight first (hot path of everything); (3) a custodian pass to collapse the two-of-everything *before* adding anything; (4) *then* build.

**Reranker-specific note for Fable + the landscape research feeding this section:** the current Qwen3-Reranker-8B was chosen against the model landscape *at the time*. That landscape moves monthly. Attached research surveys what's available *now* (cross-encoder vs late-interaction/ColBERT vs LLM-rerank vs stronger-embedding-only, on Ollama/cloud, with latency/VRAM/quality tradeoffs). We want Fable's judgment on: **is a heavyweight cross-encoder reranker still the right tool at this corpus size given hybrid RRF already runs — or has the landscape moved enough to change the answer?**

---

## Appendix B — 2026 reranker/retrieval landscape research (input for the reranker verdict)

Commissioned research (mid-2026 sources) on whether the Qwen3-Reranker-8B cross-encoder is still the right tool. **Bottleneck is already confirmed as the reranker** (live: vec 28–329ms vs rerank 11–28s), so the research's "profile first to find the bottleneck" caveat is answered — the reranker *is* the target.

**Model landscape:** Qwen3-Reranker comes in 0.6B / 4B / 8B — **8B is likely overkill**; 0.6B trades ~4–5% nDCG for 2–3× speed; 4B is the practical midpoint. Jina-Reranker-v3 (0.6B) hits top-tier quality at ~188ms p50. BGE-reranker-v2-m3 (600M), mxbai-rerank are local-friendly. Cohere/Voyage are API-only (off-table for Ollama-only).

**Alternatives:** ColBERT / late-interaction ≈2.2× faster than cross-encoders (22.6ms vs 49.9ms/query) but needs per-token doc embeddings (~2–3× storage) and a less mature ecosystem. LLM-as-reranker (RankLLM) is slower (500ms–2s local) — reserve for reasoning-heavy queries. **Stronger first-stage embedding + skip rerank** is increasingly viable; 2024–2026 evidence shows reranking helps *weak* embeddings but can *hurt* strong ones on short-target tasks.

**Rerank-on-top-of-RRF gain:** benchmarks show +17% Recall@5 in general, but at ~7500 items with hybrid RRF already present, realistic marginal gain is **~5–10%** — not nothing, but weigh it against 11–28s.

**Sub-2s techniques:** INT8 quantization (≈2× throughput, <1 nDCG loss), batching, doc truncation to ~256 tokens, ONNX/TEI/FlashRank. Confirmed sub-1s local rerank is achievable with a 0.6B INT8 model + batching.

**Embeddings:** current `nomic-embed-text` could upgrade to BGE-M3 (dense+sparse+multi-vector, ~1.2GB, immediate hybrid lift) or Qwen3-Embedding-8B (#1 MTEB, needs VRAM). Matryoshka truncation (512d ≈ 94–98% quality) cuts pgvector cost.

**Three ranked paths (all ~$0, Ollama-local):**
- **Path A (fastest, ~1.3s, −~10% precision):** 8B → 0.6B (Jina-v3 or Qwen3-0.6B) + INT8 + batch/truncate.
- **Path B (balanced, ~1.7s, −~2%):** 8B → 4B or BGE-v2-m3 + INT8 + batch/truncate.
- **Path C (skip rerank, ~400ms, −5% to +3%):** upgrade first-stage embedding (BGE-M3 / Qwen3-8B), strengthen + tune RRF (try K=30 vs 60), drop the reranker; **test on our corpus before committing.**

*Full sourced version in the research transcript; ~20 2025–2026 citations. Confidence: HIGH on A/B latency feasibility, MEDIUM on marginal-gain estimates at our corpus size (corpus-specific — must A/B test).*

---

## Appendix A — where the truth lives (for Fable's own reading)
- **Code (authoritative):** `api/app/` — `main.py`, `crud.py`, `models.py`, `settings.py`, `db.py`, `cortex.py`, `synthesis/`, `services/`, `gateway/`, `synapse/`, `routers/`, `security.py`, `runtime_config.py`, `plugin/mcp_server.py`.
- **Design intent:** `docs/plans/2026-02-19-hive-cortex-design.md` (original synthesis vision), `docs/plans/2026-04-24-hive-tiered-memory-design.md` (tiered memory), `docs/plans/2026-05-31-madi-hud-gateway-design.md` (gateway), `docs/superpowers/specs/2026-07-02-preflight-model-validation-and-alerting-design.md` (latest resilience).
- **Prior self-audit (read this):** `docs/plans/AUDIT-2026-06-04-cortex-learning-loops.md` + `PLAN-2026-06-04-cortex-control-panel-and-learning-loops.md`.
- **Do NOT trust:** `README.md` (≈40% drifted — see §6).
- **External but essential:** the SessionStart/SessionEnd shell hooks under `~/.claude/` emit `<hive-loadout>` / `<cortex-working-memory>` and own the graceful-degrade-on-timeout behavior — the API only serves the JSON.
