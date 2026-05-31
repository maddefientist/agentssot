"""Gateway configuration — the parts that are *policy*, not code.

The fallback ladder and model names live here so swapping a brain region (e.g.
dropping Opus to deepseek, or pointing the local model at a different tag) never
touches executor or service logic. Environment variables override the defaults
so deployment can retune without a code change.
"""
from __future__ import annotations

import os

# --- Local model (chat + intent classification) ---
OLLAMA_URL: str = os.environ.get("MADI_OLLAMA_URL", "http://192.168.1.225:11434")
LOCAL_MODEL: str = os.environ.get("MADI_LOCAL_MODEL", "qwen3.5:4b")

# --- Memory ---
# Namespace the gateway recalls from / teaches into by default.
HIVE_NAMESPACE: str = os.environ.get("MADI_HIVE_NAMESPACE", "claude-shared")

# --- Intent vocabulary ---
# The complete set of intents the router may emit and the registry must map.
VALID_INTENTS: frozenset[str] = frozenset(
    {"chat-local", "hive-tool", "orchestrate", "dispatch", "briefing"}
)

# Intent used when nothing else matches (cheap, instant, local).
DEFAULT_INTENT: str = "chat-local"

# --- Orchestrate fallback ladder ---
# Tried top-to-bottom. ``kind`` tells the injected runner how to invoke a rung;
# the runner owns the actual API/CLI call so this stays declarative.
#   anthropic — Claude API (Opus)
#   chain     — a ~/.claude/chains/<chain>.json brick via chain.sh
#   ollama    — local model
ORCHESTRATE_LADDER: list[dict[str, str]] = [
    {"name": "opus", "kind": "anthropic", "model": "claude-opus-4-8"},
    {"name": "deepseek-v4-pro", "kind": "chain", "chain": "deepseek-plan"},
    {"name": "glm-flash", "kind": "chain", "chain": "glm-quick"},
    {"name": "local", "kind": "ollama", "model": LOCAL_MODEL},
]
