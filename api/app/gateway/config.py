"""Optional gateway policy and model configuration.

The public default is local-only. Broader execution is separately disabled at
the application boundary and deployment-specific provider ladders are an
operator decision.
"""
from __future__ import annotations

import os

# --- Local model endpoint (chat + intent classification + ladder) ---
# Prefer an explicit gateway override, else reuse the app's OLLAMA_BASE_URL,
# else the in-container default.
OLLAMA_URL: str = (
    os.environ.get("GATEWAY_OLLAMA_URL")
    or os.environ.get("OLLAMA_BASE_URL")
    or "http://host.docker.internal:11434"
)
LOCAL_MODEL: str = (
    os.environ.get("GATEWAY_LOCAL_MODEL")
    or os.environ.get("OLLAMA_CHAT_MODEL")
    or "llama3.1"
)

# Intent classification runs on every rules-miss before generation, so a slow
# classify is pure dead time in front of the HUD. Cloud latency is spiky (the
# flash model occasionally takes ~9s), so we cap the classify and let a timeout
# degrade to DEFAULT_INTENT (chat-local) — the correct default for the
# conversational input that dominates the HUD anyway. Bounds the worst case to
# ~CLASSIFIER_TIMEOUT_S + the chat call instead of 8s+ of opaque wait.
CLASSIFIER_TIMEOUT_S: float = float(
    os.environ.get("GATEWAY_CLASSIFIER_TIMEOUT", "4.0")
)

# --- Memory ---
HIVE_NAMESPACE: str = (
    os.environ.get("GATEWAY_NAMESPACE")
    or "default"
)

# --- Intent vocabulary ---
VALID_INTENTS: frozenset[str] = frozenset(
    {"chat-local", "hive-tool", "orchestrate", "dispatch", "briefing"}
)
DEFAULT_INTENT: str = "chat-local"

# The public ladder does not silently select a cloud provider. Operators can
# supply a deployment-specific registry when they explicitly enable execution.
ORCHESTRATE_LADDER: list[dict[str, str]] = [
    {"name": "local", "kind": "ollama", "model": LOCAL_MODEL},
]
