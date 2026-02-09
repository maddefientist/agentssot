# Status (hari-hive / AgentSSOT)

As of: 2026-02-08

## Running

- Host: `hari` (`192.168.1.225`)
- API: `:8088`
- Web GUI: `/`
- Docs: `/docs`

## Implemented

- Namespaces + role enforcement (`reader|writer|admin`)
- X-API-Key auth (only `/health` is public)
- Postgres 16 + pgvector + daily backups
- Ingest/query/recall endpoints
- Session compaction (auto-disabled unless `LLM_PROVIDER != none`)
- Embedding provider plugin (`none|openai|ollama`)
- Optional HNSW index creation toggle
- Admin endpoints for namespaces + API keys
- Admin embeddings backfill endpoint: `POST /admin/backfill-embeddings`

## Embeddings (Important)

- Ollama model `qwen3-embedding:latest` produces **4096-dim** vectors.
- Set:
  - `EMBEDDING_PROVIDER=ollama`
  - `EMBEDDING_DIM=4096`
  - `OLLAMA_EMBED_MODEL=qwen3-embedding:latest`

## Key Distribution

- Best practice: generate per-device/per-agent writer keys scoped to only needed namespaces.
- Use the controller machine (MacBook) to run:
  - `python3 ~/.claude/agentssot/scripts/provision_from_hosts.py`
  - `~/.claude/agentssot/scripts/push_keys_ssh.sh`

## Next Steps

1. Enable Ollama embeddings in `/opt/agentssot/.env` and restart compose.
2. Backfill embeddings for existing namespaces with `/admin/backfill-embeddings`.
3. (Optional) Add reranking with `dengcao/Qwen3-Reranker-8B:Q8_0`.

