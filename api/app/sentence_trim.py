"""Query-time extractive sentence trimming (lift #3 from ArcRift review).

ArcRift returns only the sentences within a chunk that match the query
("surgical trimming"), rather than the whole chunk, targeting >75% token
compression. AgentSSOT returns full 800-char snippets. For loadout/recall under
a tight token budget, trimming a snippet down to its query-relevant sentences
recovers headroom without dropping whole items.

This is a lexical (embedding-free) extractive trimmer: deterministic, zero extra
API calls, and safe — if nothing matches it returns the original snippet rather
than blanking it. Gated behind RECALL_SENTENCE_TRIM (default: off).
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z0-9_]+")
_STOPWORDS = frozenset(
    "the a an and or of to in is are was were be been being for on at by with "
    "this that these those it its as from we you they i he she them his her our "
    "your their do does did how what why when where which who whom can could "
    "should would will not no yes if then than into out up down over under".split()
)


def _terms(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)} - _STOPWORDS


def trim_to_query(snippet: str, query_text: str | None, max_sentences: int = 4) -> str:
    """Return only the sentences of `snippet` most relevant to `query_text`.

    - Keeps at most `max_sentences`, in their original document order.
    - Scores sentences by overlap with query terms (stopwords removed).
    - Returns the original snippet unchanged when there's no query, the snippet
      is already short, or no sentence overlaps the query (fail-open).
    """
    if not snippet or not query_text:
        return snippet
    q_terms = _terms(query_text)
    if not q_terms:
        return snippet

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(snippet.strip()) if s.strip()]
    if len(sentences) <= max_sentences:
        return snippet

    scored: list[tuple[int, int, str]] = []  # (score, original_index, sentence)
    for idx, sent in enumerate(sentences):
        overlap = len(_terms(sent) & q_terms)
        if overlap:
            scored.append((overlap, idx, sent))

    if not scored:
        return snippet  # fail-open — never return an empty snippet

    # Pick the top-scoring sentences, then restore document order.
    scored.sort(key=lambda t: (-t[0], t[1]))
    keep = sorted(scored[:max_sentences], key=lambda t: t[1])
    trimmed = " ".join(s for _, _, s in keep)

    # If a contiguous gap was removed, mark the elision so the reader knows.
    kept_indices = [i for _, i, _ in keep]
    if kept_indices and (kept_indices[0] > 0 or kept_indices[-1] < len(sentences) - 1
                         or any(b - a > 1 for a, b in zip(kept_indices, kept_indices[1:]))):
        trimmed = trimmed + " […]"
    return trimmed


def trim_recall_items(items: list[dict], query_text: str | None, max_sentences: int = 4,
                      snippet_key: str = "snippet") -> int:
    """Trim the snippet field of recall result dicts in place. Returns count trimmed."""
    if not query_text:
        return 0
    trimmed = 0
    for item in items:
        original = item.get(snippet_key)
        if not isinstance(original, str) or not original:
            continue
        new = trim_to_query(original, query_text, max_sentences)
        if new != original:
            item[snippet_key] = new
            item["trimmed"] = True
            trimmed += 1
    return trimmed
