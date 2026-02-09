import httpx

from .base import EmbeddingProvider, EmbeddingProviderError


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        super().__init__(
            provider_name="ollama",
            is_available=bool(base_url and model),
            unavailable_reason=None if (base_url and model) else "OLLAMA_BASE_URL or OLLAMA_EMBED_MODEL missing",
        )

    def embed_text(self, text: str) -> list[float]:
        if not self.is_available:
            raise EmbeddingProviderError(self.unavailable_reason or "Ollama embedding provider unavailable")

        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text,
        }

        try:
            response = httpx.post(url, json=payload, timeout=self.timeout_seconds)
        except Exception as exc:
            raise EmbeddingProviderError(f"Ollama embedding request failed: {exc}") from exc

        if response.status_code >= 400:
            raise EmbeddingProviderError(
                f"Ollama embedding request failed with {response.status_code}: {response.text[:400]}"
            )

        data = response.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise EmbeddingProviderError("Ollama embedding response format was unexpected")
        return embedding
