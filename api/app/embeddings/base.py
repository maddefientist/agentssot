from abc import ABC, abstractmethod


class EmbeddingProviderError(Exception):
    pass


class EmbeddingProvider(ABC):
    provider_name: str

    def __init__(self, provider_name: str, is_available: bool, unavailable_reason: str | None = None):
        self.provider_name = provider_name
        self.is_available = is_available
        self.unavailable_reason = unavailable_reason

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]
