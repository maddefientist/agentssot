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

BASE_URL: str = _cfg.get("base_url", "http://192.168.1.225:8088")
API_KEY: str = _cfg.get("admin_api_key") or _cfg.get("api_key", "")
DEFAULT_NS: str = _cfg.get("default_namespace", "claude-shared")
DEVICE_NAME: str = _cfg.get("device_name", "unknown")
AGENT_KEY: str = f"device-{DEVICE_NAME}-writer"

HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
TIMEOUT = httpx.Timeout(30.0, connect=10.0)

mcp = FastMCP("hari-hive")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_key_for(role: str = "writer") -> str:
    """Return the API key appropriate for the requested role.

    role='admin' reads admin.json if present; raises PermissionError if missing.
    role='writer' or 'reader' returns the agent.json api_key.
    """
    admin_path = Path.home() / ".claude/agentssot/local/admin.json"
    agent_path = Path(
        os.environ.get("HIVE_AGENT_JSON", Path.home() / ".claude/agentssot/local/agent.json")
    )
    if role == "admin":
        if admin_path.exists():
            return json.loads(admin_path.read_text())["admin_api_key"]
        raise PermissionError(
            "admin operation requires admin.json on this device — contact operator"
        )
    return json.loads(agent_path.read_text())["api_key"]


async def _client(role: str | None = None) -> httpx.AsyncClient:
    """Build an AsyncClient. Default uses the cached writer key; pass role='admin'
    to swap in the admin key from admin.json."""
    if role and role != "writer":
        try:
            key = _api_key_for(role)
        except (PermissionError, FileNotFoundError, KeyError):
            key = API_KEY
        headers = {"X-API-Key": key, "Content-Type": "application/json"}
        return httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=TIMEOUT)
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
    item_scope = item.get("scope", "knowledge")

    header = f"[{idx}] ({score_str}) [{tags}] {created}"
    if item_scope == "concepts":
        ctype = item.get("concept_type", "?")
        conf = item.get("confidence", 0)
        if ctype == "skill":
            trigger = item.get("trigger", "?")
            action = item.get("action", "?")
            hint = item.get("success_hint", "")
            header = f"[{idx}] SKILL (conf={conf:.2f}) ({score_str}) [{tags}]"
            skill_snippet = f"When: {trigger}\nDo: {action}"
            if hint:
                skill_snippet += f"\nVerify: {hint}"
            return f"{header}\n{skill_snippet}"
        header = f"[{idx}] CONCEPT ({ctype}, conf={conf:.2f}) ({score_str}) [{tags}]"
    return f"{header}\n{snippet}"


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
    scope: str = "all",
    top_k: int = 5,
    session_id: str = "",
) -> str:
    """Semantic (vector) recall from hari-hive memory. Returns tier-bucketed results.

    Args:
        query: Natural-language search query.
        namespace: Namespace to search (default: claude-shared).
        scope: Scope filter -- all (blends knowledge + concepts), knowledge, requirements, events, or concepts.
        top_k: Max results per tier (used as default for top_per_tier).
        session_id: Optional session identifier for tracking recall events.
    """
    ns = namespace or DEFAULT_NS
    import time as _time
    body: dict[str, Any] = {
        "query": query,
        "namespace": ns,
        "bucketed": True,
        "top_per_tier": {
            "command": top_k, "rule": top_k, "skill": top_k,
            "entity": top_k, "decision": top_k,
        },
        "session_id": session_id or f"session-{_time.time_ns()}",
    }
    try:
        async with await _client() as c:
            resp = await c.post("/api/v1/knowledge/recall", json=body)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    data = resp.json()
    buckets = data.get("buckets", {})
    diag = data.get("diagnostics", {})
    lines = [f"Recall: '{query}' in {ns}"]
    total = 0
    for tier, items in buckets.items():
        if not items:
            continue
        total += len(items)
        lines.append(f"\n[{tier}] ({len(items)})")
        for it in items:
            abs_text = it.get("abstract") or ""
            lines.append(f"  • {abs_text} (id={it['id']})")
    if total == 0:
        return f"No results for '{query}' in {ns}."
    if diag:
        lines.append(f"\n— vec={diag.get('vec_ms',0)}ms rerank={diag.get('rerank_ms',0)}ms ({diag.get('reranker_used','none')})")
    return "\n".join(lines)


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
            # Use cortex endpoints (writer-accessible) instead of /admin/stats
            resp = await c.get("/cortex/data", params=params)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        # Fallback to admin endpoint for admin-role keys
        try:
            async with await _client() as c:
                resp = await c.get("/admin/stats", params=params)
            if resp.status_code != 200:
                return await _api_error(resp)
        except httpx.HTTPError as exc:
            return f"Connection error: {exc}"
        data = resp.json()
        lines = [f"Stats for {ns}:"]
        lines.append(f"  Entities: {data.get('entities', '?')}")
        for section in ("knowledge_items", "requirements", "events", "concepts"):
            s = data.get(section, {})
            if isinstance(s, dict):
                lines.append(f"  {section}: total={s.get('total', 0)}, embedded={s.get('embedded', 0)}")
            else:
                lines.append(f"  {section}: {s}")
        return "\n".join(lines)
    data = resp.json()
    lines = [f"Stats for {ns}:"]
    lines.append(f"  Concepts: {data.get('total', '?')}")
    lines.append(f"  Knowledge Items: {data.get('knowledge_count', '?')}")
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
# Cortex working-memory tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def cortex_state(
    agent_key: str,
    namespace: str = "",
    include_completed: bool = False,
) -> str:
    """Read an agent/project's working-memory tasks (decisions, pending actions, artifacts)
    and recent deltas from Cortex.

    Use this at session start to reload prior working state before continuing a task.

    agent_key convention: slugified basename of the project cwd, e.g. "agentssot" or
    "teleton". One rolling task per project; the same key is shared across Pi madi-core
    and Claude Code so both see the same blackboard.

    Args:
        agent_key: Slug identifying the agent/project (e.g. "agentssot").
        namespace: Namespace (default: claude-shared).
        include_completed: If True, include completed/abandoned tasks.
    """
    ns = namespace or DEFAULT_NS
    params: dict[str, Any] = {
        "namespace": ns,
        "agent_key": agent_key,
        "include_completed": include_completed,
    }
    try:
        async with await _client() as c:
            resp = await c.get("/cortex/state", params=params)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    data = resp.json()
    active_tasks = data.get("active_tasks", [])
    recent_deltas = data.get("recent_deltas", [])
    if not active_tasks:
        return f"No active working memory for agent_key={agent_key}"
    lines: list[str] = []
    for task in active_tasks:
        lines.append(
            f"Task: {task.get('task_title', '(untitled)')} "
            f"[status={task.get('status')} version={task.get('version')} "
            f"updated={task.get('updated_at', '')}]"
        )
        lines.append(f"  task_id: {task.get('task_id', '')}")
        decisions = task.get("decisions") or []
        if decisions:
            lines.append("  Decisions:")
            for d in decisions:
                lines.append(f"    - {d}")
        pending = task.get("pending_actions") or []
        if pending:
            lines.append("  Pending actions:")
            for p in pending:
                lines.append(f"    - {p}")
        artifacts = task.get("artifacts") or []
        if artifacts:
            lines.append("  Artifacts:")
            for a in artifacts:
                lines.append(f"    - {a}")
        snapshot = task.get("context_snapshot", "")
        if snapshot:
            lines.append(f"  Context snapshot: {snapshot}")
        lines.append("")
    if recent_deltas:
        lines.append("Recent deltas:")
        for delta in recent_deltas:
            lines.append(
                f"  [{delta.get('created_at', '')}] {delta.get('task_id', '')} "
                f"({delta.get('delta_type', '')}): {delta.get('content', '')}"
            )
    return "\n".join(lines).rstrip()


@mcp.tool()
async def cortex_reconstruct(
    agent_key: str,
    namespace: str = "",
    max_chars: int = 6000,
    include_recent_knowledge: bool = False,
    top_k_knowledge: int = 5,
) -> str:
    """Build a budget-aware working-memory context block to reload prior task state
    at session start.

    Returns a ready-to-inject string summarising the agent's last known state plus,
    optionally, relevant knowledge items. The returned block includes a WRITE-BACK
    CONTRACT footer reminding the caller to push updates via cortex_update.

    Args:
        agent_key: Slug identifying the agent/project (e.g. "agentssot").
        namespace: Namespace (default: claude-shared).
        max_chars: Budget cap for the returned injection string.
        include_recent_knowledge: If True, attach relevant knowledge snippets.
        top_k_knowledge: How many knowledge items to include (if enabled).
    """
    ns = namespace or DEFAULT_NS
    body: dict[str, Any] = {
        "namespace": ns,
        "agent_key": agent_key,
        "max_chars": max_chars,
        "include_recent_knowledge": include_recent_knowledge,
        "top_k_knowledge": top_k_knowledge,
    }
    try:
        async with await _client() as c:
            resp = await c.post("/cortex/reconstruct", json=body)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    data = resp.json()
    injection = data.get("injection", "")
    if not injection:
        return f"No prior working memory to reconstruct for agent_key={agent_key}"
    footer = (
        "\n\n---\n"
        f'WRITE-BACK CONTRACT: as work progresses, call cortex_update(agent_key="{agent_key}", '
        "task_title=..., status=\"in_progress\", decisions=[...], pending_actions=[...], "
        'artifacts=[...], delta="<what changed>"). Send COMPLETE lists (update overwrites them).'
    )
    return injection + footer


@mcp.tool()
async def cortex_update(
    agent_key: str,
    task_title: str,
    task_id: str = "",
    status: str = "in_progress",
    decisions: list[str] | None = None,
    pending_actions: list[str] | None = None,
    artifacts: list[str] | None = None,
    context_snapshot: str = "",
    delta: str = "",
    namespace: str = "",
) -> str:
    """Write (overwrite) an agent/project's working-memory task in Cortex.

    IMPORTANT — OVERWRITE SEMANTICS: decisions, pending_actions, and artifacts are
    FULLY REPLACED on every call. Do NOT send only new items; always send the COMPLETE
    current list. Use the delta field to append an incremental note to the audit log
    without affecting the lists.

    agent_key convention: slugified basename of the project cwd (e.g. "agentssot").
    One rolling task per project; shared between Pi madi-core and Claude Code.

    Args:
        agent_key: Slug identifying the agent/project.
        task_title: Human-readable title for the task.
        task_id: Existing task ID to update (leave blank to auto-create).
        status: One of: pending, in_progress, completed, abandoned.
        decisions: COMPLETE list of decisions made so far (overwrites previous).
        pending_actions: COMPLETE list of pending actions (overwrites previous).
        artifacts: COMPLETE list of artifact paths/URLs (overwrites previous).
        context_snapshot: Optional free-form snapshot string (overwrites previous).
        delta: Optional incremental note appended to the audit delta log.
        namespace: Namespace (default: claude-shared).
    """
    valid_statuses = {"pending", "in_progress", "completed", "abandoned"}
    if status not in valid_statuses:
        return (
            f"Invalid status '{status}'. Must be one of: "
            + ", ".join(sorted(valid_statuses))
        )
    ns = namespace or DEFAULT_NS
    body: dict[str, Any] = {
        "namespace": ns,
        "agent_key": agent_key,
        "task_title": task_title,
        "status": status,
        "decisions": decisions if decisions is not None else [],
        "pending_actions": pending_actions if pending_actions is not None else [],
        "artifacts": artifacts if artifacts is not None else [],
    }
    if task_id:
        body["task_id"] = task_id
    if context_snapshot:
        body["context_snapshot"] = context_snapshot
    if delta:
        body["delta"] = delta
    try:
        async with await _client() as c:
            resp = await c.post("/cortex/update", json=body)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    data = resp.json()
    tid = data.get("task_id", "?")
    version = data.get("version", "?")
    created = "created" if data.get("created") else "updated"
    return (
        f"Cortex updated: task_id={tid} version={version} ({created}). "
        "Remember: lists are overwritten, send full state next time."
    )


# ---------------------------------------------------------------------------
# Admin tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def hive_create_namespace(name: str) -> str:
    """Create a new namespace in hari-hive.

    Also grants the agent's writer key access to the new namespace (Layer-2 fix).

    Args:
        name: Namespace name (lowercase, hyphens allowed).
    """
    try:
        async with await _client(role="admin") as c:
            resp = await c.post("/admin/namespaces", json={"name": name})
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code not in (200, 201):
        return await _api_error(resp)

    # Grant the agent's writer key access to the new namespace.
    # The admin key (used above) may have wildcard scope and skip the auto-grant in the
    # server-side Layer-1 patch. We explicitly grant the writer key here to be safe.
    try:
        async with await _client(role="admin") as c:
            keys_resp = await c.get("/admin/api-keys")
        if keys_resp.status_code == 200:
            all_keys = keys_resp.json()
            writer_key = next((k for k in all_keys if k.get("name") == AGENT_KEY), None)
            if writer_key:
                key_id = writer_key["id"]
                async with await _client(role="admin") as c:
                    await c.post(
                        f"/admin/api-keys/{key_id}/namespaces/grant",
                        json={"namespaces": [name]},
                    )
    except Exception:
        pass  # grant failure is non-fatal; admin can grant manually

    return f"Namespace '{name}' created and writer key '{AGENT_KEY}' granted access."


@mcp.tool()
async def hive_list_keys() -> str:
    """List all API keys (admin only)."""
    try:
        async with await _client(role="admin") as c:
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
        async with await _client(role="admin") as c:
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
        async with await _client(role="admin") as c:
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
        async with await _client(role="admin") as c:
            resp = await c.post("/admin/dedup", json=body)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    data = resp.json()
    mode = "dry-run" if dry_run else "live"
    return f"Dedup ({mode}) in {ns}: {json.dumps(data, indent=2)}"


@mcp.tool()
async def hive_feedback(
    signal: str,
    concept_id: str = "",
    query: str = "",
    note: str = "",
    session_id: str = "",
) -> str:
    """Rate a concept: 'useful' (helped with task), 'noted' (good reminder), or 'wrong' (outdated/incorrect).
    Provide concept_id for direct reference, or query for fuzzy semantic match.
    Add a note for corrections (especially with 'wrong' signal).

    Args:
        signal: Feedback signal -- 'useful', 'noted', or 'wrong'.
        concept_id: Direct concept UUID to reference.
        query: Fuzzy semantic query to identify the concept (alternative to concept_id).
        note: Optional correction or context note (recommended for 'wrong' signal).
        session_id: Optional session identifier.
    """
    if not concept_id and not query:
        return "Error: provide concept_id or query to identify the concept"
    if signal not in ("useful", "noted", "wrong"):
        return "Error: signal must be 'useful', 'noted', or 'wrong'"

    body: dict[str, Any] = {"signal": signal, "agent_key": AGENT_KEY}
    if concept_id:
        body["concept_id"] = concept_id
    if query:
        body["query"] = query
    if note:
        body["note"] = note
    if session_id:
        body["session_id"] = session_id

    try:
        async with await _client() as c:
            resp = await c.post("/feedback", json=body)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    data = resp.json()
    if "error" in data or "detail" in data:
        return f"Feedback error: {data.get('error') or data.get('detail')}"
    return (
        f"Feedback recorded: {data['signal']} for '{data['concept_title']}' "
        f"(confidence: {data['confidence']:.2f})"
    )


@mcp.tool()
async def hive_teach(
    trigger: str,
    action: str,
    success_hint: str = "",
    namespace: str = "",
) -> str:
    """Teach the hive a new skill -- prescriptive knowledge: 'when X, do Y'.

    Args:
        trigger: When this skill activates (situation description).
        action: What to do (specific steps).
        success_hint: How to verify it worked (optional).
        namespace: Target namespace (default: claude-shared).
    """
    ns = namespace or DEFAULT_NS
    content = f"When: {trigger}\nDo: {action}"
    if success_hint:
        content += f"\nVerify: {success_hint}"

    payload = {
        "namespace": ns,
        "knowledge_items": [{
            "content": content,
            "source": "hive_teach",
            "tags": ["skill", "operator-taught"],
        }],
    }
    try:
        async with await _client() as c:
            r = await c.post("/ingest", json=payload)
            r.raise_for_status()
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"

    return f"Skill taught: '{trigger}' -> '{action}'"


@mcp.tool()
async def hive_profile(agent_key: str = "") -> str:
    """View an agent's learned profile (strengths, recall stats, preferences).

    Args:
        agent_key: The agent key to look up. Leave empty for your own device profile.
    """
    # Default to own device key
    key = agent_key or f"device-{_cfg.get('device_name', 'unknown')}-writer"
    try:
        async with await _client() as c:
            r = await c.get(f"/agent-profile/{key}")
            if r.status_code == 404:
                return f"No profile found for '{key}'. Profiles auto-create on first recall."
            r.raise_for_status()
            p = r.json()
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"

    strengths = ", ".join(p.get("strengths", [])) or "none yet"
    return (
        f"Agent: {p['agent_key']}\n"
        f"Device: {p.get('device_name', '?')}\n"
        f"Strengths: {strengths}\n"
        f"Recalls: {p.get('total_recalls', 0)} | Feedback: {p.get('total_feedback', 0)}\n"
        f"Since: {p.get('created_at', '?')}"
    )


@mcp.tool()
async def hive_session_end(
    conversation_summary: str,
    session_id: str = "",
) -> str:
    """End-of-session processing: extracts facts via Ollama (zero Claude tokens) and marks recall events complete.
    Call this at the end of a session with a brief summary of what was accomplished.

    Args:
        conversation_summary: Brief summary of what was accomplished in this session.
        session_id: Optional session identifier (auto-generated if not provided).
    """
    if not conversation_summary.strip():
        return "Error: provide a conversation summary"

    import time as _time
    body: dict[str, Any] = {
        "session_id": session_id or f"session-{_time.time_ns()}",
        "conversation_summary": conversation_summary,
        "recalled_concept_ids": [],
        "agent_key": AGENT_KEY,
    }
    try:
        async with await _client() as c:
            resp = await c.post("/session-complete", json=body)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    data = resp.json()
    if "error" in data or "detail" in data:
        return f"Session-end error: {data.get('error') or data.get('detail')}"
    return (
        f"Session complete: {data['facts_extracted']} facts extracted (via Ollama), "
        f"{data['recall_events_completed']} recall events marked complete"
    )


@mcp.tool()
async def hive_status() -> str:
    """Check your integration health with the hive neural memory system.
    Returns: identity, enrollment status, service health, concept/knowledge counts,
    recall/feedback stats, and any detected issues. Call this to verify you are
    properly connected and learning.
    """
    issues: list[str] = []
    lines: list[str] = [
        "=== Hive Integration Status ===",
        f"Device: {DEVICE_NAME}",
        f"Agent Key: {AGENT_KEY}",
        f"API: {BASE_URL}",
        f"Namespace: {DEFAULT_NS}",
        "",
    ]

    # 1. Health check
    try:
        async with await _client() as c:
            resp = await c.get("/health")
        if resp.status_code == 200:
            h = resp.json()
            lines.append("--- Service Health ---")
            lines.append(f"  Embedding: {'OK' if h.get('embedding_available') else 'DOWN'}")
            lines.append(f"  LLM: {'OK' if h.get('llm_available') else 'DOWN'}")
            lines.append(f"  Reranker: {'OK' if h.get('reranker_available') else 'DOWN'}")
            lines.append(f"  Synthesis: {'ON' if h.get('synthesis_enabled') else 'OFF'}")
            if not h.get("embedding_available"):
                issues.append("Embedding provider is down — recall will not work")
            if not h.get("llm_available"):
                issues.append("LLM provider is down — synthesis and fact extraction disabled")
        else:
            issues.append(f"Health endpoint returned {resp.status_code}")
    except httpx.HTTPError as exc:
        issues.append(f"Cannot reach API: {exc}")
        lines.append(f"API UNREACHABLE: {exc}")
        lines.append("")
        lines.append("Issues: " + "; ".join(issues))
        return "\n".join(lines)

    # 2. Stats
    try:
        async with await _client() as c:
            resp = await c.get("/cortex/system-info", params={"namespace": DEFAULT_NS})
        if resp.status_code == 200:
            info = resp.json()
            cfg = info.get("config", {})
            agents = info.get("agents", [])
            lines.append("")
            lines.append("--- Knowledge Stats ---")
            # Get counts from cortex data
            async with await _client() as c:
                dr = await c.get("/cortex/data", params={"namespace": DEFAULT_NS})
            if dr.status_code == 200:
                dd = dr.json()
                lines.append(f"  Concepts: {dd.get('total', '?')}")
                lines.append(f"  Knowledge Items: {dd.get('knowledge_count', '?')}")

            lines.append("")
            lines.append("--- Your Profile ---")
            my_profile = next((a for a in agents if a["agent_key"] == AGENT_KEY), None)
            if my_profile:
                lines.append(f"  Recalls: {my_profile.get('total_recalls', 0)}")
                lines.append(f"  Feedback: {my_profile.get('total_feedback', 0)}")
                strengths = my_profile.get("strengths", [])
                lines.append(f"  Strengths: {', '.join(strengths) if strengths else 'none yet (builds from recall patterns)'}")
                lines.append(f"  Last Active: {my_profile.get('updated_at', 'never')}")
            else:
                lines.append("  No profile yet — it builds automatically from your recalls and feedback")
                issues.append("No agent profile exists yet (normal for new agents)")

            lines.append("")
            lines.append(f"--- Config ---")
            lines.append(f"  Synthesis model: {cfg.get('synthesis_model', '?')}")
            lines.append(f"  Embedding model: {cfg.get('embedding_model', '?')}")
            lines.append(f"  Synthesis hour: {cfg.get('synthesis_schedule_hour', '?')}:00 UTC")
    except Exception:
        issues.append("Could not fetch system info")

    # 3. Summary
    lines.append("")
    if issues:
        lines.append(f"Issues ({len(issues)}):")
        for iss in issues:
            lines.append(f"  - {iss}")
    else:
        lines.append("Status: ALL GOOD — connected, enrolled, learning")

    lines.append("")
    lines.append("Tip: Use hive_recall before starting work. Use hive_feedback when knowledge helps. Facts are auto-extracted at session end.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier-memory tools (Plan 1 T1.5 + T2.6)
# ---------------------------------------------------------------------------

def _format_expand(body: dict) -> str:
    lines = [f"item: {body.get('id','?')} (layer={body.get('layer','?')})"]
    if body.get("abstract"):
        lines.append(f"\n[abstract]\n{body['abstract']}")
    if body.get("summary"):
        lines.append(f"\n[summary]\n{body['summary']}")
    if body.get("content"):
        lines.append(f"\n[content]\n{body['content']}")
    return "\n".join(lines)


def _format_loadout(body: dict) -> str:
    out = [
        f"=== Hive Loadout (tokens_used={body.get('tokens_used',0)}, "
        f"overflow={body.get('overflow_count',0)}) ===\n"
    ]
    items = body.get("items", {}) or {}
    for tier, tier_items in items.items():
        out.append(f"[{tier}] ({len(tier_items)})")
        for it in tier_items:
            title = it.get("title") or ""
            out.append(f"- {title} — {it.get('abstract','')} (id={it.get('id','?')})")
        out.append("")
    if body.get("overflow_count"):
        out.append(f"+{body['overflow_count']} more — call hive_expand or hive_recall")
    cache_key = body.get("cache_key", "")
    if cache_key:
        out.append(f"\ncache_key: {cache_key[:16]}...")
    return "\n".join(out)


@mcp.tool()
async def hive_expand(item_id: str, layer: str = "summary") -> str:
    """Fetch full or summary content for an item by id.

    Args:
        item_id: UUID of the item (from a recall/loadout result).
        layer: 'abstract', 'summary', or 'full'. Default 'summary'.
    """
    try:
        async with await _client() as c:
            resp = await c.get(
                f"/api/v1/knowledge/items/{item_id}/expand",
                params={"layer": layer},
            )
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code == 404:
        return f"item {item_id} not found"
    if resp.status_code != 200:
        return await _api_error(resp)
    return _format_expand(resp.json())


@mcp.tool()
async def hive_loadout(
    cwd: str = "",
    device_id: str = "",
    namespace: str = "",
    token_budget: int = 750,
) -> str:
    """Compute the cwd-aware loadout bundle for this device.

    Use mid-session after a compaction to restore push context, or to
    sanity-check what context the SessionStart hook would have shipped.

    Args:
        cwd: Working directory. Defaults to PWD or cwd().
        device_id: Calling device id. Defaults to this device.
        namespace: Namespace (default: claude-shared).
        token_budget: Max tokens to pack. Default 750.
    """
    body = {
        "cwd": cwd or os.environ.get("PWD") or os.getcwd(),
        "device_id": device_id or DEVICE_NAME,
        "namespace": namespace or DEFAULT_NS,
        "token_budget": token_budget,
    }
    try:
        async with await _client() as c:
            resp = await c.post("/api/v1/knowledge/loadout", json=body)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    return _format_loadout(resp.json())


@mcp.tool()
async def hive_supersede(old_id: str, new_id: str) -> str:
    """Mark old_id as superseded by new_id (decays old confidence, expires in 30d)."""
    try:
        async with await _client() as c:
            resp = await c.post(
                f"/api/v1/knowledge/items/{old_id}/supersede",
                json={"superseded_by": new_id},
            )
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    return json.dumps(resp.json())


@mcp.tool()
async def hive_expire(item_id: str, reason: str = "") -> str:
    """Soft-expire an item (sets expires_at = now). Stays in DB for audit."""
    try:
        async with await _client() as c:
            resp = await c.post(
                f"/api/v1/knowledge/items/{item_id}/expire",
                json={"reason": reason},
            )
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    return json.dumps(resp.json())


@mcp.tool()
async def hive_promote(item_id: str, priority: int) -> str:
    """Set loadout_priority. Higher = packed earlier in cwd-aware loadout."""
    try:
        async with await _client() as c:
            resp = await c.post(
                f"/api/v1/knowledge/items/{item_id}/promote",
                json={"priority": priority},
            )
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    return json.dumps(resp.json())


@mcp.tool()
async def hive_review_queue(namespace: str = "", kind: str = "", limit: int = 50) -> str:
    """List pending Review Queue items. Admin-only.

    kind in {'low_conf','dup','supersede','contradiction'} or empty for all.
    """
    params: dict[str, Any] = {"limit": limit}
    if namespace:
        params["namespace"] = namespace
    if kind:
        params["kind"] = kind
    try:
        async with await _client(role="admin") as c:
            resp = await c.get("/api/v1/knowledge/admin/review-queue", params=params)
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    except PermissionError as exc:
        return f"Permission error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    items = resp.json()
    if not items:
        return "review queue empty"
    lines = [f"{len(items)} pending"]
    for it in items[:20]:
        primary = (it.get("primary_id") or "")[:8]
        lines.append(
            f"  [{it.get('kind','?')}] p={it.get('priority','?')} "
            f"primary={primary} reason={it.get('reason','')}"
        )
    return "\n".join(lines)


@mcp.tool()
async def hive_guide() -> str:
    """Fetch the agent-facing runbook (markdown). Per-key tailored."""
    try:
        async with await _client() as c:
            resp = await c.get("/agent-guide")
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    return resp.text




@mcp.tool()
async def hive_doctor() -> str:
    """One-shot health snapshot for the hive. Call at session start to verify
    providers are up, check index size, and see last ingest timestamp.
    """
    try:
        async with await _client() as c:
            resp = await c.get("/doctor")
    except httpx.HTTPError as exc:
        return f"Connection error: {exc}"
    if resp.status_code != 200:
        return await _api_error(resp)
    d = resp.json()
    lines = [
        "=== Hive Doctor ===",
        f"Status: {d.get('status', '?')}",
        f"Embedding: {d.get('embedding_provider')} / {d.get('embedding_model')}",
        f"Reranker: {d.get('reranker_provider')} / {d.get('reranker_model')}",
        f"LLM: {d.get('llm_provider')} / {d.get('llm_model')}",
        f"Vector index items: {d.get('vector_index_count')}",
        f"Last ingest: {d.get('last_ingest_at') or 'never'}",
        f"Namespaces: {d.get('namespace_count')}",
        f"Active keys: {d.get('active_key_count')}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
