from abc import ABC, abstractmethod


class RerankerProviderError(Exception):
    pass


class RerankerProvider(ABC):
    provider_name: str

    def __init__(self, provider_name: str, is_available: bool, unavailable_reason: str | None = None):
        self.provider_name = provider_name
        self.is_available = is_available
        self.unavailable_reason = unavailable_reason

    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score each document against the query.

        Returns a list of relevance scores (0.0-1.0) in the same order as the
        input documents.  Higher means more relevant.
        """
        raise NotImplementedError
