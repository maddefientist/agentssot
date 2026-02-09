from .base import EmbeddingProvider, EmbeddingProviderError
from .ollama_provider import OllamaEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider


class DisabledEmbeddingProvider(EmbeddingProvider):
    def __init__(self, reason: str = "Embedding provider disabled"):
        super().__init__(provider_name="none", is_available=False, unavailable_reason=reason)

    def embed_text(self, text: str) -> list[float]:
        raise EmbeddingProviderError(self.unavailable_reason or "Embedding provider disabled")


def build_embedding_provider(settings) -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(api_key=settings.openai_api_key, model=settings.openai_embed_model)
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddingProvider(base_url=settings.ollama_base_url, model=settings.ollama_embed_model)
    return DisabledEmbeddingProvider(reason="EMBEDDING_PROVIDER=none")


__all__ = [
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "DisabledEmbeddingProvider",
    "build_embedding_provider",
]
