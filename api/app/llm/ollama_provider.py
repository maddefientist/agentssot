import httpx

from .base import LLMProvider, LLMProviderError


class OllamaLLMProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 45):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        super().__init__(
            provider_name="ollama",
            is_available=bool(base_url and model),
            unavailable_reason=None if (base_url and model) else "OLLAMA_BASE_URL or OLLAMA_CHAT_MODEL missing",
        )

    def summarize(self, transcript: str) -> str:
        if not self.is_available:
            raise LLMProviderError(self.unavailable_reason or "Ollama LLM provider unavailable")

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Summarize the agent session into key decisions and next steps. "
                        "Keep it concise and actionable."
                    ),
                },
                {"role": "user", "content": transcript},
            ],
        }

        url = f"{self.base_url}/api/chat"

        try:
            response = httpx.post(url, json=payload, timeout=self.timeout_seconds)
        except Exception as exc:
            raise LLMProviderError(f"Ollama chat request failed: {exc}") from exc

        if response.status_code >= 400:
            raise LLMProviderError(f"Ollama chat request failed with {response.status_code}: {response.text[:400]}")

        data = response.json()
        message = data.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMProviderError("Ollama chat response format was unexpected")
        return content.strip()
