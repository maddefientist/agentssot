"""Gateway configuration — the parts that are *policy*, not code.

The fallback ladder and model names live here so swapping a brain region (e.g.
dropping Opus to deepseek, or pointing the local model at a different tag) never
touches executor or service logic. Environment variables override the defaults
so deployment can retune without a code change.

Deployment note: the gateway runs inside the agentssot-api container, which
reaches Ollama at ``host.docker.internal:11434`` (same as the rest of the app).
Ollama proxies cloud models (``deepseek-v4-pro:cloud`` etc.), so the whole
fallback ladder is served through one endpoint — no chain.sh / pi needed in the
container.
"""
from __future__ import annotations

import os

# --- Local model endpoint (chat + intent classification + ladder) ---
# Prefer an explicit gateway override, else reuse the app's OLLAMA_BASE_URL,
# else the in-container default.
OLLAMA_URL: str = (
    os.environ.get("MADI_OLLAMA_URL")
    or os.environ.get("OLLAMA_BASE_URL")
    or "http://host.docker.internal:11434"
)
LOCAL_MODEL: str = os.environ.get("MADI_LOCAL_MODEL", "qwen3.5:4b")

# --- Memory ---
HIVE_NAMESPACE: str = os.environ.get("MADI_HIVE_NAMESPACE", "claude-shared")

# --- Intent vocabulary ---
VALID_INTENTS: frozenset[str] = frozenset(
    {"chat-local", "hive-tool", "orchestrate", "dispatch", "briefing"}
)
DEFAULT_INTENT: str = "chat-local"

# --- Orchestrate fallback ladder (the user's requested shape) ---
# Tried top-to-bottom; ``kind`` tells the runner how to invoke a rung.
#   anthropic — Claude API (Opus). Lights up only where ANTHROPIC_API_KEY + SDK
#               exist (e.g. a host deploy); skips/falls over in the container.
#   ollama    — served via the Ollama endpoint, INCLUDING cloud models. This is
#               how deepseek-v4-pro/flash are reached from the container.
#   chain     — ~/.claude/scripts/chain.sh brick (host deploys only).
ORCHESTRATE_LADDER: list[dict[str, str]] = [
    {"name": "opus", "kind": "anthropic", "model": "claude-opus-4-8"},
    {"name": "deepseek-v4-pro", "kind": "ollama", "model": "deepseek-v4-pro:cloud"},
    {"name": "deepseek-v4-flash", "kind": "ollama", "model": "deepseek-v4-flash:cloud"},
    {"name": "local", "kind": "ollama", "model": LOCAL_MODEL},
]
