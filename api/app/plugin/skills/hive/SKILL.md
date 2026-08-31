---
name: hive
description: Query and manage AgentSSOT unified memory. Use when asked to search memory, recall context, store knowledge, or manage the knowledge base.
allowed-tools: Read, Bash, Grep, Glob, AskUserQuestion
---

# /hive — AgentSSOT Knowledge Base

Interact with AgentSSOT unified memory via MCP tools.

## Commands

When the user runs `/hive`, determine intent from arguments:

| Usage | Action |
|-------|--------|
| `/hive [query]` | Run `hive_recall` with the query text |
| `/hive search [text]` | Run `hive_query` for exact text match |
| `/hive store [content]` | Run `hive_ingest` with the content |
| `/hive stats` | Run `hive_stats` for namespace overview |
| `/hive dedup` | Run `hive_dedup` (dry run first, confirm before executing) |
| `/hive keys` | Run `hive_list_keys` to show API keys |
| `/hive` (no args) | Show this help |

## Default Behavior

- Namespace: use the agent's configured namespace unless specified
- Scope: `all` unless specified (knowledge + synthesized concepts)
- Recall: one non-reranked pass, 5 total results
- Query: 10 exact/keyword results
- Deep tier sweep: explicit only; it is slower and uses a per-tier result budget

## Recall Decision

Use Hive only when prior cross-session or cross-agent context could materially
change the answer. Skip it when the prompt/repository is self-contained or live
state should be verified directly. A weak or empty match means
`NO_MEMORY_NEEDED`; do not force unrelated memory into the task.

Rate only the exact ID returned by recall. Do not use fuzzy feedback when an
item ID is available.

## Tags Convention

When ingesting, always include:
- `device-{hostname}` — source device
- Project name if in a project directory
- `session-extract` for end-of-session facts
- `cross-llm` for info that should be available to all AI tools
