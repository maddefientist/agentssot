"""Two-tier reranker router. Selects the 4B fast model for procedural-only
queries and the 8B deep model for queries that include nuanced tiers.

Procedural tiers are configured in settings.procedural_tiers (default:
command, rule, entity). When the query targets only those, the 4B reranker
is used (~80–150ms). Otherwise the 8B reranker (~300–500ms) handles the
candidate pool.
"""
from __future__ import annotations

from app.settings import get_settings
from app.reranker.base import RerankerProvider
from app.reranker.ollama_provider import OllamaRerankerProvider


def select_reranker_model(tiers: list[str]) -> str:
    """Return 'fast' or 'deep' based on the requested tier set."""
    settings = get_settings()
    procedural = set(settings.procedural_tiers)
    if not tiers:
        return "deep"
    requested = set(tiers)
    return "fast" if requested.issubset(procedural) else "deep"


def build_reranker_pair(settings) -> tuple[RerankerProvider, RerankerProvider]:
    """Build (fast, deep) reranker providers. Either may be a Disabled stub
    if the corresponding model isn't available — caller falls back to the
    other or to vector-only ranking."""
    if settings.reranker_provider != "ollama":
        from app.reranker import DisabledRerankerProvider
        stub = DisabledRerankerProvider(reason="RERANKER_PROVIDER=none")
        return stub, stub

    deep_url = settings.ollama_reranker_base_url or settings.ollama_base_url
    fast_url = settings.ollama_reranker_fast_base_url or deep_url

    deep = OllamaRerankerProvider(
        base_url=deep_url,
        model=settings.ollama_reranker_model,
    )
    fast = OllamaRerankerProvider(
        base_url=fast_url,
        model=settings.ollama_reranker_fast_model,
    )
    return fast, deep


def pick_reranker(tiers: list[str], fast: RerankerProvider, deep: RerankerProvider) -> tuple[str, RerankerProvider]:
    """Pick the model name + provider instance for the given tier set."""
    choice = select_reranker_model(tiers)
    if choice == "fast" and fast.is_available:
        return "qwen3-reranker-4b", fast
    if deep.is_available:
        return "qwen3-reranker-8b", deep
    if fast.is_available:
        return "qwen3-reranker-4b", fast
    return "none", deep   # disabled stub; caller will skip rerank
