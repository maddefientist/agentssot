from .base import LLMProvider, LLMProviderError
from .ollama_provider import OllamaLLMProvider
from .openai_provider import OpenAILLMProvider


class DisabledLLMProvider(LLMProvider):
    def __init__(self, reason: str = "LLM provider disabled"):
        super().__init__(provider_name="none", is_available=False, unavailable_reason=reason)

    def summarize(self, transcript: str) -> str:
        raise LLMProviderError(self.unavailable_reason or "LLM provider disabled")


def build_llm_provider(settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        return OpenAILLMProvider(api_key=settings.openai_api_key, model=settings.openai_chat_model)
    if settings.llm_provider == "ollama":
        return OllamaLLMProvider(base_url=settings.ollama_base_url, model=settings.ollama_chat_model)
    return DisabledLLMProvider(reason="LLM_PROVIDER=none")


__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "DisabledLLMProvider",
    "build_llm_provider",
]
