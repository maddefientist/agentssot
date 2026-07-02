"""Live model-existence checks against an Ollama /api/tags endpoint.

Single source of truth for "does this model exist". /api/tags lists both local
and cloud (…:cloud) models, so one call covers qwen3.6:27b and
qwen3.5:397b-cloud alike.
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("agentssot.llm.model_validation")

TAGS_TIMEOUT_SECONDS = 5.0


class ModelListUnavailable(RuntimeError):
    """Raised when the Ollama model list cannot be fetched."""


def list_available_models(base_url: str, timeout: float = TAGS_TIMEOUT_SECONDS) -> set[str]:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        resp = httpx.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return {m.get("name") for m in data.get("models", []) if m.get("name")}
    except Exception as exc:  # noqa: BLE001
        raise ModelListUnavailable(f"cannot fetch/parse {url}: {exc}") from exc


def validate_models(
    base_url: str, required: list[str], timeout: float = TAGS_TIMEOUT_SECONDS
) -> tuple[set[str], set[str]]:
    """Return (present, missing) for the required model names."""
    available = list_available_models(base_url, timeout=timeout)
    req = {r for r in required if r}
    return (req & available, req - available)