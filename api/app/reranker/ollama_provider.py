import logging
import re

import httpx

from .base import RerankerProvider, RerankerProviderError

logger = logging.getLogger("agentssot.reranker.ollama")

_PROMPT_TEMPLATE = (
    "Given a query and a document, determine if the document is relevant.\n\n"
    "Query: {query}\n"
    "Document: {document}\n\n"
    "Relevance score (0-1):"
)

_SCORE_RE = re.compile(r"([01](?:\.\d+)?)")


class OllamaRerankerProvider(RerankerProvider):
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        super().__init__(
            provider_name="ollama",
            is_available=bool(base_url and model),
            unavailable_reason=None if (base_url and model) else "OLLAMA_BASE_URL or OLLAMA_RERANKER_MODEL missing",
        )

    def _score_single(self, query: str, document: str) -> float:
        prompt = _PROMPT_TEMPLATE.format(query=query, document=document)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 5, "temperature": 0},
        }
        url = f"{self.base_url}/api/generate"

        try:
            response = httpx.post(url, json=payload, timeout=self.timeout_seconds)
        except Exception as exc:
            raise RerankerProviderError(f"Ollama reranker request failed: {exc}") from exc

        if response.status_code >= 400:
            raise RerankerProviderError(
                f"Ollama reranker request failed with {response.status_code}: {response.text[:400]}"
            )

        data = response.json()
        raw = data.get("response", "").strip()

        match = _SCORE_RE.search(raw)
        if match:
            return min(max(float(match.group(1)), 0.0), 1.0)

        logger.warning("Could not parse reranker score from response: %r", raw)
        return 0.0

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not self.is_available:
            raise RerankerProviderError(self.unavailable_reason or "Ollama reranker provider unavailable")

        return [self._score_single(query, doc) for doc in documents]
