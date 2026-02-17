# hari-hive Claude Code Plugin — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Claude Code plugin that exposes AgentSSOT as native MCP tools, replacing shell-script hooks with on-demand retrieval for 60-90% token savings.

**Architecture:** HTTP proxy pattern — a Python MCP server (stdio transport, run via `uv run`) translates MCP tool calls into HTTP requests to the existing AgentSSOT REST API at `http://YOUR_HOST:8088`. Plugin hooks replace the current `hive-session-start.sh` and `hive-session-end.sh` shell scripts.

**Tech Stack:** Python 3.12, `mcp` SDK, `httpx` for async HTTP, `uv` for dependency management (inline script deps)

---

### Task 1: Scaffold Plugin Structure

**Files:**
- Create: `~/.claude/plugins/hari-hive/plugin.json`
- Create: `~/.claude/plugins/hari-hive/.mcp.json`

**Step 1: Create plugin directory**

```bash
mkdir -p ~/.claude/plugins/hari-hive/{hooks,skills/hive}
```

**Step 2: Write plugin.json**

Create `~/.claude/plugins/hari-hive/plugin.json`:

```json
{
  "name": "hari-hive",
  "version": "1.0.0",
  "description": "AgentSSOT (hari-hive) unified memory — MCP tools for recall, query, ingest, and admin",
  "author": "MadDefientist",
  "hooks": ["SessionStart", "SessionEnd"],
  "commands": ["hive"],
  "dependencies": {
    "uv": "required"
  },
  "settings": {
    "baseUrl": {
      "type": "string",
      "default": "http://YOUR_HOST:8088",
      "description": "AgentSSOT API base URL"
    },
    "defaultNamespace": {
      "type": "string",
      "default": "claude-shared",
      "description": "Default namespace for recall/query/ingest"
    }
  }
}
```

**Step 3: Write .mcp.json**

Create `~/.claude/plugins/hari-hive/.mcp.json`:

```json
{
  "mcpServers": {
    "hari-hive": {
      "command": "uv",
      "args": ["run", "${CLAUDE_PLUGIN_ROOT}/mcp_server.py"],
      "env": {
        "HIVE_AGENT_JSON": "${HOME}/.claude/agentssot/local/agent.json"
      }
    }
  }
}
```

**Step 4: Verify plugin is detected**

```bash
ls -la ~/.claude/plugins/hari-hive/
# Should show: plugin.json, .mcp.json, hooks/, skills/
```

**Step 5: Commit**

```bash
cd /opt/agentssot
git add -A
git commit -m "feat: scaffold hari-hive plugin structure"
```

---

### Task 2: Build MCP Server — Core Retrieval Tools

**Files:**
- Create: `~/.claude/plugins/hari-hive/mcp_server.py`

This is the main MCP server. Uses inline `uv` script dependencies so no venv is needed.

**Step 1: Write MCP server with core tools (recall, query, ingest)**

Create `~/.claude/plugins/hari-hive/mcp_server.py`:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp[cli]>=1.0.0",
#     "httpx>=0.27",
# ]
# ///
"""hari-hive MCP server — proxy to AgentSSOT REST API."""

import json
import os
import sys
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

# --- Config ---

def load_config() -> tuple[str, str]:
    """Load base_url and api_key from agent.json."""
    agent_json = os.environ.get(
        "HIVE_AGENT_JSON",
        str(Path.home() / ".claude/agentssot/local/agent.json"),
    )
    try:
        data = json.loads(Path(agent_json).read_text())
        base_url = data["base_url"].rstrip("/")
        api_key = data.get("api_key") or data.get("admin_api_key") or ""
        if not api_key:
            print("WARN: no api_key in agent.json", file=sys.stderr)
        return base_url, api_key
    except Exception as e:
        print(f"ERROR loading agent.json: {e}", file=sys.stderr)
        sys.exit(1)


BASE_URL, API_KEY = load_config()

mcp = FastMCP(
    "hari-hive",
    instructions=(
        "hari-hive is a knowledge base (AgentSSOT). "
        "Use hive_recall for semantic search, hive_query for text/tag search, "
        "hive_ingest to store new knowledge. Default namespace: claude-shared."
    ),
)

# --- HTTP helpers ---

def _headers() -> dict[str, str]:
    return {"Content-Type": "application/json", "X-API-Key": API_KEY}


async def _post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{BASE_URL}{path}", json=payload, headers=_headers())
        r.raise_for_status()
        return r.json()


async def _get(path: str, params: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{BASE_URL}{path}", params=params, headers=_headers())
        r.raise_for_status()
        return r.json()


# --- Core Retrieval Tools ---

@mcp.tool()
async def hive_recall(
    query: str,
    namespace: str = "claude-shared",
    scope: str = "knowledge",
    top_k: int = 5,
) -> str:
    """Semantic vector search across the knowledge base.

    Args:
        query: Natural language search query
        namespace: Namespace to search (default: claude-shared)
        scope: What to search — knowledge, requirements, or events
        top_k: Number of results to return (default: 5)
    """
    result = await _post("/recall", {
        "query_text": query,
        "namespace": namespace,
        "scope": scope,
        "top_k": top_k,
    })
    items = result.get("items", [])
    if not items:
        return f"No results found for '{query}' in {namespace}/{scope}"
    lines = []
    for item in items:
        score = item.get("reranker_score") or item.get("score", 0)
        tags = ", ".join(item.get("tags", []))
        lines.append(f"[score={score:.2f}] ({tags})\n{item['snippet']}")
    return f"Found {len(items)} results:\n\n" + "\n\n---\n\n".join(lines)


@mcp.tool()
async def hive_query(
    q: str,
    namespace: str = "claude-shared",
    limit: int = 10,
) -> str:
    """Text search across all items. Good for exact matches and tag filtering.

    Args:
        q: Search text (matches against title, content, tags)
        namespace: Namespace to search (default: claude-shared)
        limit: Max results (default: 10)
    """
    result = await _get("/query", {"q": q, "namespace": namespace, "limit": limit})
    total = result.get("total", 0)
    results = result.get("results", [])
    if not results:
        return f"No results for '{q}' in {namespace} (0 total)"
    lines = []
    for r in results:
        tags = ", ".join(r.get("tags", []))
        lines.append(f"**{r['title']}** ({tags})\n{r['snippet']}")
    return f"Found {total} total, showing {len(results)}:\n\n" + "\n\n---\n\n".join(lines)


@mcp.tool()
async def hive_ingest(
    content: str,
    tags: list[str] | None = None,
    namespace: str = "claude-shared",
    source: str | None = None,
) -> str:
    """Store a knowledge item in the hive.

    Args:
        content: The text content to store
        tags: Tags for categorization (e.g. ["project-context", "my-project"])
        namespace: Target namespace (default: claude-shared)
        source: Optional source reference (e.g. file path, URL)
    """
    item = {"content": content, "tags": tags or [], "source": source}
    result = await _post("/ingest", {
        "namespace": namespace,
        "knowledge_items": [item],
    })
    counts = result.get("counts", {})
    return f"Ingested into {namespace}: {counts}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

**Step 2: Test MCP server starts without errors**

```bash
cd ~/.claude/plugins/hari-hive
uv run mcp_server.py --help 2>&1 || echo "Check errors above"
```

Expected: Server help or no errors (it will hang waiting for stdio input, Ctrl+C to stop).

**Step 3: Test with MCP inspector (optional)**

```bash
uv run mcp dev ~/.claude/plugins/hari-hive/mcp_server.py
```

Verify tools `hive_recall`, `hive_query`, `hive_ingest` appear. Test `hive_recall` with query "infrastructure" to confirm HTTP proxy works.

**Step 4: Commit**

```bash
cd /opt/agentssot
git add -A
git commit -m "feat: MCP server with core retrieval tools (recall, query, ingest)"
```

---

### Task 3: Add Browse & Admin Tools to MCP Server

**Files:**
- Modify: `~/.claude/plugins/hari-hive/mcp_server.py`

**Step 1: Add stats, summarize, and admin tools**

Append these tools to `mcp_server.py` (before the `if __name__` block):

```python
# --- Browse & Stats Tools ---

@mcp.tool()
async def hive_stats(namespace: str = "claude-shared") -> str:
    """Get item counts and embedding coverage for a namespace.

    Args:
        namespace: Namespace to check (default: claude-shared)
    """
    result = await _get("/admin/stats", {"namespace": namespace})
    ki = result.get("knowledge_items", {})
    req = result.get("requirements", {})
    ev = result.get("events", {})
    return (
        f"Namespace: {result['namespace']}\n"
        f"Entities: {result.get('entities', 0)}\n"
        f"Knowledge items: {ki.get('total', 0)} ({ki.get('embedded', 0)} embedded)\n"
        f"Requirements: {req.get('total', 0)} ({req.get('embedded', 0)} embedded)\n"
        f"Events: {ev.get('total', 0)} ({ev.get('embedded', 0)} embedded)"
    )


@mcp.tool()
async def hive_summarize(
    session_id: str,
    namespace: str = "claude-shared",
    project_slug: str | None = None,
    max_events: int = 500,
) -> str:
    """Summarize and archive session events into a knowledge item.

    Args:
        session_id: Session ID to summarize
        namespace: Namespace (default: claude-shared)
        project_slug: Optional project filter
        max_events: Max events to process (default: 500)
    """
    result = await _post("/summarize_clear", {
        "namespace": namespace,
        "session_id": session_id,
        "project_slug": project_slug,
        "max_events": max_events,
    })
    return (
        f"Summarized session {result['session_id']}:\n"
        f"  Archived events: {result['archived_events']}\n"
        f"  Summary item ID: {result['summary_knowledge_item_id']}"
    )


# --- Admin Tools ---

@mcp.tool()
async def hive_create_namespace(name: str) -> str:
    """Create a new namespace.

    Args:
        name: Namespace name (lowercase, hyphens allowed)
    """
    result = await _post("/admin/namespaces", {"name": name})
    return f"Created namespace '{result['name']}' at {result['created_at']}"


@mcp.tool()
async def hive_list_keys() -> str:
    """List all API keys (requires admin role)."""
    keys = await _get("/admin/api-keys")
    if not keys:
        return "No API keys found"
    lines = []
    for k in keys:
        ns = ", ".join(k.get("namespaces", []))
        lines.append(
            f"  {k['name']} [{k['role']}] ns=[{ns}] "
            f"active={k['is_active']} preview={k['key_preview']}"
        )
    return f"API Keys ({len(keys)}):\n" + "\n".join(lines)


@mcp.tool()
async def hive_create_key(
    name: str,
    role: str = "writer",
    namespaces: list[str] | None = None,
) -> str:
    """Create a new API key (requires admin role).

    Args:
        name: Key name (e.g. 'device-myhost-writer')
        role: reader, writer, or admin
        namespaces: List of namespace access (default: ['claude-shared'])
    """
    result = await _post("/admin/api-keys", {
        "name": name,
        "role": role,
        "namespaces": namespaces or ["claude-shared"],
    })
    return (
        f"Created key '{result['name']}' [{result['role']}]\n"
        f"  API Key: {result['api_key']}\n"
        f"  Namespaces: {result['namespaces']}"
    )


@mcp.tool()
async def hive_delete_items(
    ids: list[str],
    namespace: str = "claude-shared",
) -> str:
    """Delete specific items by ID.

    Args:
        ids: List of item UUIDs to delete (max 100)
        namespace: Namespace (default: claude-shared)
    """
    result = await _post("/admin/delete-items", {
        "namespace": namespace,
        "ids": ids,
    })
    return f"Deleted {result['deleted']} items from {result['namespace']}"


@mcp.tool()
async def hive_dedup(
    namespace: str = "claude-shared",
    dry_run: bool = True,
) -> str:
    """Find and remove duplicate items.

    Args:
        namespace: Namespace to dedup (default: claude-shared)
        dry_run: If True, only report duplicates without deleting (default: True)
    """
    result = await _post("/admin/dedup", {
        "namespace": namespace,
        "dry_run": dry_run,
    })
    mode = "DRY RUN" if result["dry_run"] else "EXECUTED"
    return (
        f"Dedup [{mode}] in {result['namespace']}:\n"
        f"  Duplicate groups: {result['duplicate_groups']}\n"
        f"  {'Would delete' if result['dry_run'] else 'Deleted'}: {result['deleted']}"
    )
```

**Step 2: Test that all 10 tools register**

```bash
cd ~/.claude/plugins/hari-hive
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | uv run mcp_server.py 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{len(d['result']['tools'])} tools:\"); [print(f\"  - {t['name']}\") for t in d['result']['tools']]"
```

Expected: 10 tools listed.

**Step 3: Commit**

```bash
cd /opt/agentssot
git add -A
git commit -m "feat: add browse/stats and admin tools to MCP server"
```

---

### Task 4: Write Plugin Hooks

**Files:**
- Create: `~/.claude/plugins/hari-hive/hooks/SessionStart.md`
- Create: `~/.claude/plugins/hari-hive/hooks/SessionEnd.md`

**Step 1: Write SessionStart hook**

Create `~/.claude/plugins/hari-hive/hooks/SessionStart.md`:

```markdown
---
name: SessionStart
description: Inject lightweight hive context hint on session start
enabled: true
---

# hari-hive Session Start

```bash
#!/bin/bash
PROJECT_NAME=$(basename "$(pwd)")

echo "<hive-available>"
echo "You have access to hari-hive (AgentSSOT) via MCP tools."
echo "Use hive_recall for semantic search, hive_query for text/tag search."
echo "Use hive_ingest to store knowledge. Use hive_stats for namespace info."
echo "Current project: ${PROJECT_NAME}. Default namespace: claude-shared."
echo "Fetch context on demand — do NOT pre-load everything."
echo "</hive-available>"
```
```

**Step 2: Write SessionEnd hook**

Create `~/.claude/plugins/hari-hive/hooks/SessionEnd.md`:

```markdown
---
name: SessionEnd
description: Prompt fact extraction and ingest on session end
enabled: true
---

# hari-hive Session End

```bash
#!/bin/bash
PROJECT_NAME=$(basename "$(pwd)")
DEVICE_NAME=$(hostname -s 2>/dev/null || echo "unknown")

echo "<hive-session-end>"
echo "Before ending, extract 3-5 key facts from this session and ingest them."
echo "Use hive_ingest with tags: [\"session-extract\", \"device-${DEVICE_NAME}\", \"${PROJECT_NAME}\"]"
echo "Focus on: decisions made, bugs fixed, patterns learned, architecture changes."
echo "Skip: routine file reads, obvious actions, session-specific details."
echo "</hive-session-end>"
```
```

**Step 3: Verify hook files**

```bash
ls -la ~/.claude/plugins/hari-hive/hooks/
# Should show: SessionStart.md, SessionEnd.md
head -5 ~/.claude/plugins/hari-hive/hooks/SessionStart.md
# Should show YAML frontmatter
```

**Step 4: Commit**

```bash
cd /opt/agentssot
git add -A
git commit -m "feat: add SessionStart/SessionEnd plugin hooks"
```

---

### Task 5: Write /hive Skill

**Files:**
- Create: `~/.claude/plugins/hari-hive/skills/hive/SKILL.md`

**Step 1: Write SKILL.md**

Create `~/.claude/plugins/hari-hive/skills/hive/SKILL.md`:

```markdown
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
```

**Step 2: Verify skill file**

```bash
head -5 ~/.claude/plugins/hari-hive/skills/hive/SKILL.md
# Should show YAML frontmatter with name: hive
```

**Step 3: Commit**

```bash
cd /opt/agentssot
git add -A
git commit -m "feat: add /hive slash command skill"
```

---

### Task 6: Disable Old Shell Hooks

**Files:**
- Modify: `~/.claude/settings.json` (remove old hive hook entries)
- Note: Do NOT delete old scripts — keep as backup in `~/.claude/agentssot/scripts/`

**Step 1: Read current settings.json hook config**

```bash
cat ~/.claude/settings.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
hooks = d.get('hooks', {})
for event, config in hooks.items():
    print(f'{event}:')
    if isinstance(config, dict):
        for h in config.get('hooks', []):
            print(f'  - {h.get(\"command\", \"?\")[:80]}')
    elif isinstance(config, list):
        for h in config:
            print(f'  - {h}')
"
```

**Step 2: Remove hive-session-start and extract_and_ingest hooks from settings.json**

Edit `~/.claude/settings.json`:
- Remove the SessionStart hook entry that runs `hive-session-start.sh`
- Remove the SessionEnd hook entry that runs `extract_and_ingest.py`
- Keep any non-hive hooks intact

**Step 3: Verify settings.json is valid JSON**

```bash
python3 -c "import json; json.load(open('$HOME/.claude/settings.json')); print('valid')"
```

**Step 4: Commit**

```bash
cd /opt/agentssot
git add -A
git commit -m "feat: disable old shell hooks in favor of plugin hooks"
```

---

### Task 7: Integration Test

**Step 1: Verify plugin is detected by Claude Code**

Restart Claude Code and check that:
- `hari-hive` appears in plugin list
- MCP server starts (check for `hari-hive` in MCP server list)
- SessionStart hook fires (should see `<hive-available>` message)

**Step 2: Test each MCP tool manually**

In a Claude Code session:
1. Ask: "Use hive_recall to search for 'infrastructure'" — should return results
2. Ask: "Use hive_query to search for 'agentssot'" — should return results
3. Ask: "Use hive_stats" — should show namespace counts
4. Ask: "Use hive_list_keys" — should show API keys
5. Ask: "Use hive_ingest to store 'MCP plugin integration test successful' with tags ['test', 'mcp-plugin']" — should confirm ingestion
6. Ask: "Use hive_recall to search for 'MCP plugin integration test'" — should find the item just ingested
7. Ask: "Use hive_delete_items to remove the test item" — should confirm deletion

**Step 3: Verify token savings**

Compare session start token usage:
- Old: ~500-2000 tokens of eagerly loaded context
- New: ~50 tokens (just the `<hive-available>` hint)

**Step 4: Commit any fixes**

```bash
cd /opt/agentssot
git add -A
git commit -m "test: integration test pass for hari-hive MCP plugin"
```

---

### Task 8: Update Documentation

**Files:**
- Modify: `/opt/agentssot/README.md` (add MCP plugin section)
- Modify: `~/.claude/projects/-opt-agentssot/memory/MEMORY.md` (update operational memory)

**Step 1: Add plugin section to README**

Add a "Claude Code Plugin" section to the README documenting:
- How to install (it's already in `~/.claude/plugins/`)
- Available MCP tools (10 tools with descriptions)
- `/hive` command usage
- How hooks work (SessionStart hint, SessionEnd extraction)

**Step 2: Update MEMORY.md**

Add plugin info to operational memory:
- Plugin location: `~/.claude/plugins/hari-hive/`
- MCP server: `mcp_server.py` (uv run, stdio transport)
- Tools: 10 (recall, query, ingest, stats, summarize, create_namespace, list_keys, create_key, delete_items, dedup)
- Old hooks disabled, plugin hooks active

**Step 3: Commit**

```bash
cd /opt/agentssot
git add -A
git commit -m "docs: add MCP plugin documentation and update operational memory"
```
