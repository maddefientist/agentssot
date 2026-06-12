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
# The HUD's casual-chat + intent-classification model. A *cloud* model proxied
# through Ollama (no local VRAM, so it never gets evicted and cold-starts —
# the on-box qwen3.5:4b used to hang up to 60s on the first command after idle).
# Reasoning models like deepseek emit chain-of-thought, so both call sites pass
# ``think: false`` to keep HUD replies and classifier output clean.
LOCAL_MODEL: str = os.environ.get("MADI_LOCAL_MODEL", "deepseek-v4-flash:cloud")

# Intent classification runs on every rules-miss before generation, so a slow
# classify is pure dead time in front of the HUD. Cloud latency is spiky (the
# flash model occasionally takes ~9s), so we cap the classify and let a timeout
# degrade to DEFAULT_INTENT (chat-local) — the correct default for the
# conversational input that dominates the HUD anyway. Bounds the worst case to
# ~CLASSIFIER_TIMEOUT_S + the chat call instead of 8s+ of opaque wait.
CLASSIFIER_TIMEOUT_S: float = float(os.environ.get("MADI_CLASSIFIER_TIMEOUT", "4.0"))

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
