import logging
import re
from concurrent.futures import ThreadPoolExecutor

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
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 30, max_concurrency: int = 8):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        # Candidates are scored with one Ollama /api/generate call each. Run them
        # concurrently (bounded) instead of sequentially — a 30-candidate rerank
        # drops from ~15s to ~order-of-one-call. Effective parallelism is also
        # gated server-side by OLLAMA_NUM_PARALLEL.
        self.max_concurrency = max(1, max_concurrency)
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

        if not documents:
            return []

        # Single document: no pool overhead.
        if len(documents) == 1:
            return [self._score_single(query, documents[0])]

        workers = min(self.max_concurrency, len(documents))
        # ex.map preserves input order; a RerankerProviderError raised in any
        # worker propagates here when results are consumed, so the caller's
        # vector-score fallback still triggers exactly as before.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rerank") as ex:
            return list(ex.map(lambda doc: self._score_single(query, doc), documents))
