# hari-hive (AgentSSOT) Onboarding For LLM Agents

This is a short, copy/paste-friendly guide you can provide to any LLM agent so it can use hari-hive as its primary long-term memory system.

## Connection

- Base URL: `http://192.168.1.225:8088`
- Auth header: `X-API-Key: <your-key>`
- Web GUI: `/`
- Docs: `/docs`

Note: `GET /onboarding` returns a plaintext onboarding guide tailored to the API key (role + allowed namespaces). It still requires auth.

## The One Rule That Matters

**Always specify the namespace.** Namespaces are the privacy boundary.

Examples:
- Shared: `hari-hive` (or `team-shared`)
- Private: `device-macbook-private`, `finance-private`

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
- Don’t dump transcripts as knowledge unless explicitly requested.
- Let compaction produce “summary” knowledge for long sessions.

## Embeddings

- If `EMBEDDING_PROVIDER=ollama|openai`, the server can embed query text for recall and can embed ingested items when embedding is not provided.
- If `EMBEDDING_PROVIDER=none`, clients must provide embeddings when required (especially recall query embeddings).

## Suggested Data Shapes

- **knowledge_items**: facts, configs, stable decisions, canonical project context.
- **events**: decisions/actions/results per work session; use `session_id`.
- **requirements**: goals/backlog items with status/priority.

