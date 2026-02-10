from .base import RerankerProvider, RerankerProviderError
from .ollama_provider import OllamaRerankerProvider


class DisabledRerankerProvider(RerankerProvider):
    def __init__(self, reason: str = "Reranker provider disabled"):
        super().__init__(provider_name="none", is_available=False, unavailable_reason=reason)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        raise RerankerProviderError(self.unavailable_reason or "Reranker provider disabled")


def build_reranker_provider(settings) -> RerankerProvider:
    if settings.reranker_provider == "ollama":
        return OllamaRerankerProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_reranker_model,
        )
    return DisabledRerankerProvider(reason="RERANKER_PROVIDER=none")


__all__ = [
    "RerankerProvider",
    "RerankerProviderError",
    "DisabledRerankerProvider",
    "build_reranker_provider",
]
