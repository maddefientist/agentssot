# AgentSSOT Onboarding For LLM Agents

Last updated: 2026-06-10

This is a short, copy/paste-friendly guide you can provide to any LLM agent so it can use AgentSSOT as its primary long-term memory system.

## What's New Since 2026-02 (Cortex Layer)

**Typed memory taxonomy**: Knowledge items now have explicit `memory_type` (rule, doctrine, entity, command, episodic, etc.) to support selective loadout and promotion workflows.

**Tiered loadout system**: At session start, agents receive a `<hive-loadout>` block containing the most relevant rules (priority 5, always included) and rotating doctrine items (priority 4). Use `hive_recall` and `hive_teach` to query and store knowledge in the shared hive.

**Synthesis & promotion**: Daily jobs cluster recent knowledge, reconcile Concepts, and promote high-confidence doctrine back into knowledge items so it enters the loadout. Feedback signals (` POST /feedback`) train the synthesis pipeline.

**Review queue and dedup**: New `/admin/feedback` and `/admin/dedup` endpoints manage editorial workflows — contested items, duplicates, low-confidence entries. `POST /admin/feedback/complete-sessions` summarizes multi-turn interactions.

**Runtime control plane**: Operators can now hot-swap Ollama models, adjust synthesis thresholds, and repair failing providers without restarting (`GET /admin/config`, `POST /admin/config`). Live provider health visible at `GET /admin/connections`.

**Gateway/HUD interface**: Realtime command dispatch via `WebSocket /gateway/ws` and status streaming via `GET /gateway/sse/status`. The Madi HUD (`GET /hud`) provides an Obsidian Terminal surface for interactive workflows.

Endpoint summary for new features:
- `POST /feedback` — Signal usefulness, errors, or classifications.
- `GET  /admin/config` — Read runtime overrides.
- `POST /admin/config` — Set an override (hot model swap, threshold tuning).
- `DELETE /admin/config/{key}` — Revert to default.
- `GET  /admin/connections` — Provider health snapshot.
- `POST /admin/feedback/complete-sessions` — Summarize session interactions.
- `POST /admin/dedup` — Deduplicate items by embedding similarity.

## Connection

- Base URL: `http://your-host:8088`
- Auth header: `X-API-Key: <your-key>`
- Web GUI: `/`
- Docs: `/docs`

Note: `GET /onboarding` returns a plaintext onboarding guide tailored to the API key (role + allowed namespaces). It still requires auth.

## The One Rule That Matters

**Always specify the namespace.** Namespaces are the privacy boundary.

Examples:
- Shared: `team-shared`
- Private: `device-laptop-private`, `finance-private`

If you query the wrong namespace, you either leak data or miss the relevant context.

## Recommended Agent Loop

1. Start every task:
   - `GET /query` (quick scan, keyword search)
   - `POST /recall` (semantic search) in the relevant namespace(s)
2. During the task:
   - Append durable progress as `events` (decisions/directives/results)
3. End of task:
   - Write durable facts as `knowledge_items` (atomic, tagged)
   - If available, summarize/clear the session with `POST /summarize_clear` (archives verbose events and stores a summary)

## Endpoints (Minimal)

- `GET  /health` (no auth)
- `GET  /query` (auth)
- `POST /recall` (auth)
- `POST /ingest` (writer/admin)
- `POST /summarize_clear` (writer/admin)

Admin only:
- `POST /admin/namespaces`
- `POST /admin/api-keys`
- `GET  /admin/api-keys`
- `POST /admin/backfill-embeddings`

## Token Efficiency Rules

- Prefer Top-K recall (default 5).
- Keep knowledge items atomic.
- Don't dump transcripts as knowledge unless explicitly requested.
- Let compaction produce "summary" knowledge for long sessions.

## Embeddings

- If `EMBEDDING_PROVIDER=ollama|openai`, the server can embed query text for recall and can embed ingested items when embedding is not provided.
- If `EMBEDDING_PROVIDER=none`, clients must provide embeddings when required (especially recall query embeddings).

## Suggested Data Shapes

- **knowledge_items**: facts, configs, stable decisions, canonical project context.
- **events**: decisions/actions/results per work session; use `session_id`.
- **requirements**: goals/backlog items with status/priority.
