import httpx

from .base import LLMProvider, LLMProviderError


class OpenAILLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, timeout_seconds: int = 45):
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        super().__init__(
            provider_name="openai",
            is_available=bool(api_key),
            unavailable_reason=None if api_key else "OPENAI_API_KEY is not configured",
        )

    def summarize(self, transcript: str) -> str:
        if not self.is_available:
            raise LLMProviderError(self.unavailable_reason or "OpenAI LLM provider unavailable")

        system_prompt = (
            "You are summarizing an autonomous agent session. "
            "Produce a concise distillation with key decisions and concrete next steps."
        )

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise LLMProviderError(f"OpenAI chat request failed: {exc}") from exc

        if response.status_code >= 400:
            raise LLMProviderError(f"OpenAI chat request failed with {response.status_code}: {response.text[:400]}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("OpenAI chat response format was unexpected") from exc
