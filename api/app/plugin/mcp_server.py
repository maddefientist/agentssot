# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcp[cli]>=1.0.0",
#   "httpx>=0.27",
# ]
# ///
"""hari-hive MCP server -- proxies tool calls to the AgentSSOT REST API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_agent_path = Path(
    os.environ.get("HIVE_AGENT_JSON", Path.home() / ".claude/agentssot/local/agent.json")
)
try:
    _cfg = json.loads(_agent_path.read_text())
except Exception as exc:
    raise SystemExit(f"Cannot read agent config at {_agent_path}: {exc}") from exc

BASE_URL: str = _cfg.get("base_url", "http://YOUR_HOST:8088")
API_KEY: str = _cfg.get("admin_api_key") or _cfg.get("api_key", "")
DEFAULT_NS: str = _cfg.get("default_namespace", "claude-shared")

HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
TIMEOUT = httpx.Timeout(30.0, connect=10.0)

mcp = FastMCP("hari-hive")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=BASE_URL, headers=HEADERS, timeout=TIMEOUT)


def _fmt_recall_item(item: dict, idx: int) -> str:
    score = item.get("score", 0)
    rs = item.get("reranker_score")
    tags = ", ".join(item.get("tags", []))
    score_str = f"vec={score:.3f}"
    if rs is not None:
        score_str += f" rerank={rs:.3f}"
    snippet = item.get("snippet", "")
    created = item.get("created_at", "")
    return f"[{idx}] ({score_str}) [{tags}] {created}\n{snippet}"


def _fmt_query_result(item: dict, idx: int) -> str:
    title = item.get("title", "(untitled)")
    snippet = item.get("snippet", "")
    tags = ", ".join(item.get("tags", []))
    return f"[{idx}] {title} [{tags}]\n{snippet}"


async def _api_error(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        detail = body.get("detail", resp.text)
    except Exception:
        detail = resp.text
    return f"API error {resp.status_code}: {detail}"


# ---------------------------------------------------------------------------
# Core tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def hive_recall(
    query: str,
    namespace: str = "",
    scope: str = "knowledge",
    top_k: int = 5,
) -> str:
    """Semantic (vector) recall from hari-hive memory. Returns the most relevant items by embedding similarity.

    Args:
        query: Natural-language search query.
        namespace: Namespace to search (default: claude-shared).
        scope: Scope filter -- knowledge, requirements, events, or all.
        top_k: Max results to return.
    """
    ns = namespace or DEFAULT_NS
    body = {"query_text": query, "namespace": ns, "scope": scope, "top_k": top_k}
    try:
        async with await _client() as c:
            resp = await c.post("/recall", json=body)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    data = resp.json()
    items = data.get("items", [])
    if not items:
        return f"No results for '{query}' in {ns}/{scope}."
    lines = [f"Recall: {len(items)} results in {ns}/{scope}\n"]
    for i, item in enumerate(items, 1):
        lines.append(_fmt_recall_item(item, i))
    return "\n\n".join(lines)


@mcp.tool()
async def hive_query(
    q: str,
    namespace: str = "",
    limit: int = 20,
) -> str:
    """Full-text search across hari-hive items. Faster than recall for exact matches.

    Args:
        q: Text search query.
        namespace: Namespace to search (default: claude-shared).
        limit: Max results.
    """
    ns = namespace or DEFAULT_NS
    params: dict[str, Any] = {"q": q, "namespace": ns, "limit": limit}
    try:
        async with await _client() as c:
            resp = await c.get("/query", params=params)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    data = resp.json()
    total = data.get("total", 0)
    results = data.get("results", [])
    if not results:
        return f"No results for '{q}' in {ns}."
    lines = [f"Query: {total} total matches in {ns} (showing {len(results)})\n"]
    for i, item in enumerate(results, 1):
        lines.append(_fmt_query_result(item, i))
    return "\n\n".join(lines)


@mcp.tool()
async def hive_ingest(
    content: str,
    tags: list[str] | None = None,
    source: str | None = None,
    namespace: str = "",
) -> str:
    """Ingest a single knowledge item into hari-hive memory.

    Args:
        content: The text content to store.
        tags: Optional list of tags for categorization.
        source: Optional source identifier (e.g. file path, URL).
        namespace: Target namespace (default: claude-shared).
    """
    ns = namespace or DEFAULT_NS
    item: dict[str, Any] = {"content": content, "tags": tags or []}
    if source:
        item["source"] = source
    body = {"namespace": ns, "knowledge_items": [item]}
    try:
        async with await _client() as c:
            resp = await c.post("/ingest", json=body)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    data = resp.json()
    counts = data.get("counts", {})
    return f"Ingested into {ns}: {json.dumps(counts)}"


# ---------------------------------------------------------------------------
# Browse / Stats tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def hive_stats(namespace: str = "") -> str:
    """Get item counts and stats for a namespace.

    Args:
        namespace: Namespace to check (default: claude-shared).
    """
    ns = namespace or DEFAULT_NS
    params = {"namespace": ns}
    try:
        async with await _client() as c:
            resp = await c.get("/admin/stats", params=params)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    data = resp.json()
    lines = [f"Stats for {ns}:"]
    lines.append(f"  Entities: {data.get('entities', '?')}")
    for section in ("knowledge_items", "requirements", "events"):
        s = data.get(section, {})
        if isinstance(s, dict):
            lines.append(f"  {section}: total={s.get('total', 0)}, embedded={s.get('embedded', 0)}")
        else:
            lines.append(f"  {section}: {s}")
    return "\n".join(lines)


@mcp.tool()
async def hive_summarize(
    session_id: str,
    namespace: str = "",
    project_slug: str | None = None,
    max_events: int = 50,
) -> str:
    """Summarize and clear session events. Call at end of session to compact memory.

    Args:
        session_id: Unique session identifier.
        namespace: Namespace (default: claude-shared).
        project_slug: Optional project slug for scoping.
        max_events: Max events to summarize at once.
    """
    ns = namespace or DEFAULT_NS
    body: dict[str, Any] = {
        "namespace": ns,
        "session_id": session_id,
        "max_events": max_events,
    }
    if project_slug:
        body["project_slug"] = project_slug
    try:
        async with await _client() as c:
            resp = await c.post("/summarize_clear", json=body)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    data = resp.json()
    return f"Summarize result: {json.dumps(data, indent=2)}"


# ---------------------------------------------------------------------------
# Admin tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def hive_create_namespace(name: str) -> str:
    """Create a new namespace in hari-hive.

    Args:
        name: Namespace name (lowercase, hyphens allowed).
    """
    try:
        async with await _client() as c:
            resp = await c.post("/admin/namespaces", json={"name": name})
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code not in (200, 201):
        return await _api_error(resp)
    return f"Namespace '{name}' created."


@mcp.tool()
async def hive_list_keys() -> str:
    """List all API keys (admin only)."""
    try:
        async with await _client() as c:
            resp = await c.get("/admin/api-keys")
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    keys = resp.json()
    if not keys:
        return "No API keys found."
    lines = [f"API keys ({len(keys)}):"]
    for k in keys:
        name = k.get("name", "?")
        role = k.get("role", "?")
        ns_list = ", ".join(k.get("namespaces", []))
        prefix = k.get("key_prefix", "")
        lines.append(f"  {name} (role={role}) ns=[{ns_list}] prefix={prefix}")
    return "\n".join(lines)


@mcp.tool()
async def hive_create_key(
    name: str,
    role: str = "writer",
    namespaces: list[str] | None = None,
) -> str:
    """Create a new API key (admin only).

    Args:
        name: Descriptive key name.
        role: Role -- reader, writer, or admin.
        namespaces: List of namespaces this key can access.
    """
    body: dict[str, Any] = {"name": name, "role": role}
    if namespaces:
        body["namespaces"] = namespaces
    try:
        async with await _client() as c:
            resp = await c.post("/admin/api-keys", json=body)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code not in (200, 201):
        return await _api_error(resp)
    data = resp.json()
    key_val = data.get("api_key") or data.get("key", "")
    return f"Key created: name={name} role={role}\nKey value: {key_val}\n(Store this -- it cannot be retrieved again.)"


@mcp.tool()
async def hive_delete_items(
    ids: list[str],
    namespace: str = "",
) -> str:
    """Delete specific items by ID from hari-hive.

    Args:
        ids: List of item UUIDs to delete.
        namespace: Namespace (default: claude-shared).
    """
    ns = namespace or DEFAULT_NS
    body = {"namespace": ns, "ids": ids}
    try:
        async with await _client() as c:
            resp = await c.post("/admin/delete-items", json=body)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    data = resp.json()
    deleted = data.get("deleted", 0)
    return f"Deleted {deleted} items from {ns}."


@mcp.tool()
async def hive_dedup(
    namespace: str = "",
    dry_run: bool = True,
) -> str:
    """Find and optionally remove duplicate items in a namespace.

    Args:
        namespace: Namespace to deduplicate (default: claude-shared).
        dry_run: If True, only report duplicates without deleting (default: True).
    """
    ns = namespace or DEFAULT_NS
    body = {"namespace": ns, "dry_run": dry_run}
    try:
        async with await _client() as c:
            resp = await c.post("/admin/dedup", json=body)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    data = resp.json()
    mode = "dry-run" if dry_run else "live"
    return f"Dedup ({mode}) in {ns}: {json.dumps(data, indent=2)}"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
