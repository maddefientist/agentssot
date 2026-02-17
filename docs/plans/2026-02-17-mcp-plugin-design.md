# hari-hive Claude Code Plugin — Design

**Date:** 2026-02-17
**Status:** Approved
**Approach:** HTTP Proxy MCP Server as Claude Code Plugin

## Goal

Replace the current shell-script-based SessionStart/SessionEnd hooks with a Claude Code plugin that exposes AgentSSOT as native MCP tools. Token optimization through lazy (on-demand) retrieval instead of eager context loading.

## Architecture

```
Claude Code ──stdio──> mcp_server.py ──HTTP──> AgentSSOT REST API (port 8088)
                                                      │
                                                 pgvector DB
```

The MCP server is a thin translation layer. All business logic (embeddings, reranking, auth, compaction) stays in the existing API container. No container changes required.

## Plugin Structure

```
~/.claude/plugins/hari-hive/
├── plugin.json            # Plugin manifest
├── .mcp.json              # MCP server declaration (stdio transport)
├── mcp_server.py          # MCP server (Python, mcp SDK)
├── skills/
│   └── hive/SKILL.md      # /hive slash command
└── hooks/
    ├── session-start.md    # SessionStart prompt hook (replaces shell script)
    └── session-end.md      # SessionEnd prompt hook (replaces shell script)
```

## MCP Tools (10)

### Core Retrieval
| Tool | Endpoint | Description |
|------|----------|-------------|
| `hive_recall` | `POST /recall` | Semantic vector search across knowledge/requirements/events |
| `hive_query` | `GET /query` | Text search + tag filtering |
| `hive_ingest` | `POST /ingest` | Store knowledge items, events, requirements |

### Browse & Stats
| Tool | Endpoint | Description |
|------|----------|-------------|
| `hive_stats` | `GET /admin/stats` | Namespace item counts and embedding coverage |
| `hive_summarize` | `POST /summarize_clear` | Summarize + archive session events |

### Admin
| Tool | Endpoint | Description |
|------|----------|-------------|
| `hive_create_namespace` | `POST /admin/namespaces` | Create a new namespace |
| `hive_list_keys` | `GET /admin/api-keys` | List API keys |
| `hive_create_key` | `POST /admin/api-keys` | Create new API key |
| `hive_delete_items` | `POST /admin/delete-items` | Delete specific items by ID |
| `hive_dedup` | `POST /admin/dedup` | Find and remove duplicate items |

## Plugin Hooks

### SessionStart (replaces hive-session-start.sh)
Prompt-based hook. Injects a minimal hint (~50 tokens):
```
You have access to hari-hive (AgentSSOT) via MCP tools.
Use hive_recall for semantic search, hive_query for text search.
Current project: {project_name}. Namespace: claude-shared.
Fetch context on demand — do not pre-load.
```

### SessionEnd (replaces hive-session-end.sh + extract_and_ingest.py)
Prompt-based hook. Instructs Claude to extract key facts from the session and ingest them via `hive_ingest`.

## Auth

- MCP server reads `~/.claude/agentssot/local/agent.json` at startup
- Passes `X-API-Key` header on every HTTP request
- Admin tools require admin-role key
- No new auth mechanisms

## Token Savings

| Scenario | Current | With Plugin |
|----------|---------|-------------|
| Session start | 500-2000 tokens | ~50 tokens |
| Per hive lookup | N/A (pre-loaded) | ~200 tokens |
| Session with 0 lookups | 500-2000 wasted | 50 tokens |
| Session with 3 lookups | 500-2000 | ~650 tokens |
| Session end | ~300 tokens | ~100 tokens |

Net savings: 60-90% for typical sessions.

## Error Handling

- API unreachable: tools return clear error, don't crash
- Auth failure: surface "unauthorized" message
- Namespace missing: surface API error
- Graceful degradation: plugin works even if container is down

## Dependencies

- Python `mcp` SDK (pip installable)
- `httpx` for async HTTP client
- Existing `agent.json` credentials
