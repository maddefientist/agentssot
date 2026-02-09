from abc import ABC, abstractmethod


class LLMProviderError(Exception):
    pass


class LLMProvider(ABC):
    provider_name: str

    def __init__(self, provider_name: str, is_available: bool, unavailable_reason: str | None = None):
        self.provider_name = provider_name
        self.is_available = is_available
        self.unavailable_reason = unavailable_reason

    @abstractmethod
    def summarize(self, transcript: str) -> str:
        raise NotImplementedError
