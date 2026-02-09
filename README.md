# AgentSSOT (hari-hive)

AgentSSOT is the software project. **hari-hive** is your cross-LLM shared memory deployment name on your network.

## Stack

- PostgreSQL 16 + pgvector
- FastAPI API service
- Daily `pg_dump` backups to `./backups`
- Optional pgAdmin profile

## Quick Start

1. Copy env template and set secrets:
   - `cp .env.example .env`
   - Edit `.env` and set at least `POSTGRES_PASSWORD`.
2. Build and start:
   - `docker compose up -d --build`
3. Check health:
   - `curl http://localhost:${API_PORT:-8088}/health`
4. Open dashboard:
   - `http://localhost:${API_PORT:-8088}/`
   - Swagger docs remain at `http://localhost:${API_PORT:-8088}/docs`
5. LLM-friendly onboarding page (requires API key):
   - `GET /onboarding`
6. Repo handoff notes (for future operators/LLMs):
   - `HANDOFF.md`
   - `STATUS.md`
   - `docs/ONBOARDING_FOR_LLMS.md`

## Auth bootstrap

On first API startup:

- If `namespaces` is empty, `default` is created.
- If `api_keys` is empty, a bootstrap admin API key is generated and logged once.
- Only the bcrypt hash is stored in DB.

Capture that key from API container logs immediately:

- `docker compose logs api | rg BOOTSTRAP_ADMIN_API_KEY`

Use it as header:

- `X-API-Key: <plaintext-key>`

## Endpoints

Public:

- `GET /health`

Authenticated:

- `POST /ingest` (writer/admin)
- `GET /query` (reader/writer/admin)
- `POST /recall` (reader/writer/admin)
- `POST /summarize_clear` (writer/admin)

Admin:

- `POST /admin/namespaces`
- `POST /admin/api-keys`
- `GET /admin/api-keys`

## Built-in Web GUI

The API serves a built-in control panel at `/` with:

- Connection/API key management
- Ingest batch JSON submission
- Query and vector recall tools
- Manual summarize/clear trigger
- Admin namespace and API key operations

It calls the same API endpoints as external clients and still requires `X-API-Key` for protected routes.

## LLM-Friendly Onboarding

Agents can fetch a plaintext onboarding guide (includes their role/namespaces) via:

- `GET /onboarding` with `X-API-Key`

## Memory Scoping Pattern

For privacy-conscious multi-agent setups, use namespace isolation by default:

- Create one namespace per private context (e.g., `agent-a-private`, `finance-private`).
- Create one or more explicit shared namespaces (e.g., `team-shared`).
- Issue API keys with only the minimum allowed namespaces.
- Ingest/query/recall against a specific namespace each time.

This creates siloed memory with explicit opt-in sharing instead of global recall across all data.

## Future: "Cousins" (Off-Field Remote Agents)

Goal: run agents on other home networks (parents/sister) with **limited** access that still allows agent-to-agent communication.

Recommended pattern (works with current system):
- Create a dedicated shared communication namespace (example: `cousins-shared`).
- Issue each remote agent its own API key with `reader`/`writer` access to **only** `cousins-shared` (and optionally its own private namespace like `cousin-parents-private`).
- Do not grant those keys access to your internal namespaces.

Network exposure options (in order of preference):
1. **VPN (WireGuard/Tailscale)**: remote devices join your private network and call hari-hive over LAN/private IP.
2. **Secure tunnel / reverse proxy**: expose only the API (`:8088`) behind TLS, with strict auth; consider IP allowlists and rate limiting.
3. **Separate infrastructure**: deploy a separate AgentSSOT instance per household and build a relay/federation layer (more work, but strongest isolation/offline independence).

If you mainly want a "social network" for the remote agents, option (1) or (2) plus a dedicated namespace is typically enough. Choose (3) only if you need independent admin control, offline operation, or hard isolation between households.

## Provider configuration

### Embeddings

- `EMBEDDING_PROVIDER=none|openai|ollama`
- If `none`, clients must send embeddings when needed (for recall query embeddings and vector ingest).
- If configured but missing credentials, API still starts; embedding calls fail with clear errors.

If you have Ollama on the host and want **server-side embeddings** (recommended for token optimization):

1. In `.env`:
   - `EMBEDDING_PROVIDER=ollama`
   - `OLLAMA_BASE_URL=http://host.docker.internal:11434`
   - `OLLAMA_EMBED_MODEL=qwen3-embedding:latest` (or your preferred embedding model)
2. Restart:
   - `docker compose up -d --build`
3. Backfill embeddings for existing rows (admin-only endpoint):
   - `POST /admin/backfill-embeddings` with body like:
     - `{"namespace":"claude-shared","scope":"knowledge","limit":5000,"batch_size":50,"dry_run":false}`

Note: `EMBEDDING_DIM` must match the embedding model output dimension. If you change models and the dimension differs, you may need to re-initialize vector columns.

### Summarizer

- `LLM_PROVIDER=none|openai|ollama`
- If `none`, background compaction is disabled automatically.
- Manual `/summarize_clear` remains available but returns a clear configuration error.

## Database port exposure

Postgres is **not** exposed by default.

If you explicitly need host access, set `EXPOSE_DB_PORT=true` and run with profile:

- `docker compose --profile db-port up -d`

## Optional pgAdmin

- `docker compose --profile pgadmin up -d`
- Connect to host `db`, port `5432`, database/user from `.env`.

## Notes

- `db/init/001_init.sql` is idempotent schema bootstrap.
- `db/init/002_optional_hnsw.sql` installs a safe helper function; API executes it only when `ENABLE_HNSW_INDEX=true`.
- Knowledge content is automatically chunked to max 800 chars per row.
