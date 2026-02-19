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

    def synthesize_concepts(self, facts: str, existing_concepts: str, model_override: str | None = None, fallback_model: str | None = None) -> str:
        """Synthesize conceptual knowledge from facts. Uses model_override if set,
        falls back to fallback_model if primary fails, then to self.model."""
        raise NotImplementedError
