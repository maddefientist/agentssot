import httpx

from .base import EmbeddingProvider, EmbeddingProviderError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str, timeout_seconds: int = 30):
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        super().__init__(
            provider_name="openai",
            is_available=bool(api_key),
            unavailable_reason=None if api_key else "OPENAI_API_KEY is not configured",
        )

    def embed_text(self, text: str) -> list[float]:
        if not self.is_available:
            raise EmbeddingProviderError(self.unavailable_reason or "OpenAI embedding provider unavailable")

        payload = {
            "model": self.model,
            "input": text,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise EmbeddingProviderError(f"OpenAI embedding request failed: {exc}") from exc

        if response.status_code >= 400:
            raise EmbeddingProviderError(
                f"OpenAI embedding request failed with {response.status_code}: {response.text[:400]}"
            )

        data = response.json()
        try:
            return data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EmbeddingProviderError("OpenAI embedding response format was unexpected") from exc
