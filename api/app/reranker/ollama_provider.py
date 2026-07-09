import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor

import httpx

from .base import RerankerProvider, RerankerProviderError

logger = logging.getLogger("agentssot.reranker.ollama")

# --- "generate" scoring (legacy): ask for a 0-1 float, parse the text ---------
_PROMPT_TEMPLATE = (
    "Given a query and a document, determine if the document is relevant.\n\n"
    "Query: {query}\n"
    "Document: {document}\n\n"
    "Relevance score (0-1):"
)

_SCORE_RE = re.compile(r"([01](?:\.\d+)?)")

# --- "logit" scoring: official Qwen3-Reranker template, read P(yes) vs P(no) --
# Qwen3-Reranker is a cross-encoder trained to answer "yes"/"no" for relevance;
# the relevance score is the softmax of the yes/no logits at the answer position.
# We reproduce its documented chat template (raw mode, so Ollama does not re-wrap
# it) and read the first token's top_logprobs.
_QWEN3_RERANK_SYS = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
_QWEN3_RERANK_INSTRUCT = "Given a query, retrieve memory items relevant to answering it"

# Token variants (post strip().lower()) that count as the yes / no mass.
_YES_TOKENS = frozenset({"yes"})
_NO_TOKENS = frozenset({"no"})


def _build_qwen3_prompt(query: str, document: str) -> str:
    user = f"<Instruct>: {_QWEN3_RERANK_INSTRUCT}\n<Query>: {query}\n<Document>: {document}"
    return (
        f"<|im_start|>system\n{_QWEN3_RERANK_SYS}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


class OllamaRerankerProvider(RerankerProvider):
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 30,
        max_concurrency: int = 8,
        scoring_mode: str = "generate",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        # "generate" = legacy text-parse; "logit" = Qwen3 template + logprobs.
        self.scoring_mode = scoring_mode if scoring_mode in ("generate", "logit") else "generate"
        # Candidates are scored with one Ollama /api/generate call each. Run them
        # concurrently (bounded) instead of sequentially. Effective parallelism is
        # gated server-side by OLLAMA_NUM_PARALLEL (note: throughput on this GPU is
        # candidate-count-bound, not concurrency-bound — see the 2026-07-09 probe).
        self.max_concurrency = max(1, max_concurrency)
        super().__init__(
            provider_name="ollama",
            is_available=bool(base_url and model),
            unavailable_reason=None if (base_url and model) else "OLLAMA_BASE_URL or OLLAMA_RERANKER_MODEL missing",
        )

    def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}/api/generate"
        try:
            response = httpx.post(url, json=payload, timeout=self.timeout_seconds)
        except Exception as exc:
            raise RerankerProviderError(f"Ollama reranker request failed: {exc}") from exc
        if response.status_code >= 400:
            raise RerankerProviderError(
                f"Ollama reranker request failed with {response.status_code}: {response.text[:400]}"
            )
        return response.json()

    def _score_single_generate(self, query: str, document: str) -> float:
        payload = {
            "model": self.model,
            "prompt": _PROMPT_TEMPLATE.format(query=query, document=document),
            "stream": False,
            "options": {"num_predict": 5, "temperature": 0},
        }
        raw = self._post(payload).get("response", "").strip()
        match = _SCORE_RE.search(raw)
        if match:
            return min(max(float(match.group(1)), 0.0), 1.0)
        logger.warning("Could not parse reranker score from response: %r", raw)
        return 0.0

    def _score_single_logit(self, query: str, document: str) -> float | None:
        """Return P(yes) in [0,1] from the yes/no logits, or None if the endpoint
        returned no logprobs (older Ollama) so the caller can fall back."""
        payload = {
            "model": self.model,
            "prompt": _build_qwen3_prompt(query, document),
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 20,
            "options": {"num_predict": 1, "temperature": 0},
        }
        data = self._post(payload)
        entries = data.get("logprobs") or []
        if not entries:
            return None  # logprobs unsupported/absent — signal fallback
        top = entries[0].get("top_logprobs") or []
        # Sum probability mass across yes-variants and no-variants, then normalize.
        p_yes = 0.0
        p_no = 0.0
        for t in top:
            tok = str(t.get("token", "")).strip().lower()
            lp = t.get("logprob")
            if lp is None:
                continue
            if tok in _YES_TOKENS:
                p_yes += math.exp(lp)
            elif tok in _NO_TOKENS:
                p_no += math.exp(lp)
        if p_yes == 0.0 and p_no == 0.0:
            return 0.0  # neither yes nor no surfaced — treat as irrelevant
        return p_yes / (p_yes + p_no)

    def _score_single(self, query: str, document: str) -> float:
        if self.scoring_mode == "logit":
            score = self._score_single_logit(query, document)
            if score is not None:
                return score
            logger.warning(
                "reranker logit mode: no logprobs returned (Ollama <0.20?); "
                "falling back to generate scoring"
            )
        return self._score_single_generate(query, document)

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
