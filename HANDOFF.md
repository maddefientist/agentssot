# Handoff: AgentSSOT (hari-hive)

Date: 2026-02-08

This repository is a production-ish deployment of **AgentSSOT** (software project) running as **hari-hive** (your cross-LLM shared memory service) on the host **hari**.

## Current Deployment

- Host: `hari` (LAN IP observed: `192.168.1.225`)
- API: `http://192.168.1.225:8088`
- Web GUI: `http://192.168.1.225:8088/`
- Swagger/OpenAPI: `http://192.168.1.225:8088/docs`
- Public health: `GET /health` (no auth)

Compose stack:
- `pgvector/pgvector:pg16` (Postgres 16 + pgvector)
- `agentssot-api` (FastAPI + SQLAlchemy + psycopg + pgvector)
- `agentssot-backup` (daily `pg_dump` to `./backups`)
- Optional profiles:
  - `pgadmin` (disabled unless `--profile pgadmin`)
  - `db-port` (disabled unless `--profile db-port`)

## What Is Implemented

Core requirements implemented:
- X-API-Key auth on all endpoints except `/health`
- API keys stored as **bcrypt hashes** in DB (no plaintext in DB)
- RBAC-lite: `admin | writer | reader`
- Namespaces (multi-tenant scoping) enforced at request time
- Batch ingest: entities/requirements/knowledge_items/events
- Query (light SQL search) and Recall (vector cosine distance) with token clipping
- Knowledge item semantic chunking: max ~800 chars per row (server-side)
- Session compaction loop:
  - enabled only if `COMPACTION_ENABLED=true` **and** `LLM_PROVIDER != none`
  - produces summary `knowledge_item` tagged `summary` and archives events
- Embedding provider plugin:
  - `EMBEDDING_PROVIDER=none|openai|ollama`
  - if `none`, clients must provide embeddings where needed
- LLM provider plugin:
  - `LLM_PROVIDER=none|openai|ollama`
  - if `none`, compaction auto-disabled; manual summarize endpoint returns clear error
- Optional HNSW index creation:
  - controlled via `ENABLE_HNSW_INDEX=true|false`
  - if creation fails, logs and continues

UX:
- Built-in web GUI at `/` for admin + ingest/query/recall utilities
- LLM-friendly onboarding page at `/onboarding` (plaintext; requires a key)

## Key Operational Files

- Compose: `docker-compose.yml`
- Env template: `.env.example`
- DB init (idempotent): `db/init/001_init.sql`
- Optional HNSW helper func: `db/init/002_optional_hnsw.sql`
- API code: `api/app/`

## Embeddings: Ollama + qwen3-embedding

You confirmed on host `hari`:
- `qwen3-embedding:latest` is present and ready in Ollama
- (future) `dengcao/Qwen3-Reranker-8B:Q8_0` is present and ready (not yet integrated)

Important fact discovered:
- `qwen3-embedding:latest` outputs **4096-dim** vectors.

To enable server-side embeddings:
1. In `/opt/agentssot/.env` set:
   - `EMBEDDING_PROVIDER=ollama`
   - `EMBEDDING_DIM=4096`
   - `OLLAMA_BASE_URL=http://host.docker.internal:11434`
   - `OLLAMA_EMBED_MODEL=qwen3-embedding:latest`
2. Restart:
   - `docker compose up -d --build`
3. Backfill embeddings for existing rows (admin-only endpoint):
   - `POST /admin/backfill-embeddings`

Docker-to-host networking:
- `docker-compose.yml` includes `extra_hosts: host.docker.internal:host-gateway` to reach Ollama on the host.
- Ollama must listen on a non-loopback interface for containers to reach it. If Ollama is bound only to `127.0.0.1:11434`, container calls will fail.

Vector column dimension:
- `db/init/001_init.sql` initially creates `VECTOR(1536)` columns.
- `api/app/startup.py` now attempts to migrate `embedding` columns to `EMBEDDING_DIM` at startup.
  - If embeddings already exist with a different dimension, Postgres may reject the conversion.
  - In that case, easiest fix is to clear embeddings or re-init DB.

## Key Management / Distribution (Claude + Codex)

This repo does not ship plaintext keys. Keys are issued via:
- Admin endpoint: `POST /admin/api-keys` (returns plaintext key once)
- Bootstrap admin key: printed once on first startup in API logs as `BOOTSTRAP_ADMIN_API_KEY=...`

Automation pack exists in the user’s home directory (not in this repo):
- `~/.claude/agentssot/` with scripts for provisioning + migrating Claude memories + pushing keys to devices.
- Secrets are stored in `~/.claude/agentssot/local/` and are **gitignored** (do not sync).

Important ops note:
- Key distribution via SSH should be run from the “controller” machine that has SSH access to the fleet (often the MacBook).
- This session was running *on host hari*, so it may not have SSH keys/config to reach the other devices.

Scripts (on controller):
- Provision namespaces and per-device writer keys from `hosts.json`:
  - `python3 ~/.claude/agentssot/scripts/provision_from_hosts.py`
- Push keys to SSH-reachable devices:
  - `~/.claude/agentssot/scripts/push_keys_ssh.sh`

## Security Model (Intended Usage)

- “hari-hive” is shared memory, but **namespaces are the privacy boundary**.
- Recommended pattern:
  - One explicit shared namespace (ex: `hari-hive` or `team-shared`)
  - Many private namespaces (per device / per project / per domain)
  - Most agents get `writer` keys limited to only the namespaces they should see.
  - Reserve `admin` keys for a single provisioning agent/operator.

## Known Gaps / Next Work

High-priority:
- Configure `.env` on host to enable Ollama embeddings (4096 dim) and run backfill.
- Verify API container can reach Ollama (depends on Ollama bind address).
- Optionally implement reranking (use the available Qwen3 reranker) after vector recall.

Medium:
- Add rate limiting / basic abuse controls (since keys are bearer tokens).
- Add “read-only onboarding” docs page that does not require auth (requires relaxing requirement; currently only `/health` is public).

Low:
- More importers for project context files into `knowledge_items`.
- Add more structured UI workflows for namespaces + scoped key issuance.

## Quick Verify Commands

- Health:
  - `curl http://192.168.1.225:8088/health`
- Local compose status:
  - `cd /opt/agentssot && docker compose ps`
- API logs (bootstrap key on first run):
  - `cd /opt/agentssot && docker compose logs api | rg BOOTSTRAP_ADMIN_API_KEY`

