---
name: hive
description: Query and manage hari-hive (AgentSSOT) unified memory. Use when asked to search memory, recall context, store knowledge, or manage the knowledge base.
allowed-tools: Read, Bash, Grep, Glob, AskUserQuestion
---

# /hive — AgentSSOT Knowledge Base

Interact with hari-hive unified memory via MCP tools.

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

- Namespace: `claude-shared` unless specified
- Scope: `knowledge` unless specified
- Top-k: 5 for recall, 10 for query

## Tags Convention

When ingesting, always include:
- `device-{hostname}` — source device
- Project name if in a project directory
- `session-extract` for end-of-session facts
- `cross-llm` for info that should be available to all AI tools
