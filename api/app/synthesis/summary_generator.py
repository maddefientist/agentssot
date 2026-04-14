"""Summary generation service for L0/L1 tiered knowledge."""
import logging

from ..llm import LLMProvider, LLMProviderError

logger = logging.getLogger("agentssot.synthesis.summary_generator")


async def generate_tiered_summaries(
    content: str,
    llm_provider: LLMProvider | None,
) -> tuple[str | None, str | None]:
    """Generate L0 (abstract) and L1 (summary) from full content using LLM.

    Returns (abstract, summary) tuple.
    Abstract is ~50 tokens, summary is ~500 tokens.
    """
    if not llm_provider or not llm_provider.is_available:
        logger.warning("LLM provider unavailable, skipping summary generation")
        return None, None

    # L0: Abstract (~50 tokens)
    abstract_prompt = f"""Summarize the following in exactly 1-2 sentences (max 50 words). Be concise and capture only the core insight:

{content[:2000]}"""

    # L1: Summary (~500 tokens)
    summary_prompt = f"""Summarize the following in 3-5 paragraphs (max 500 words). Include:
1. Main topic/subject
2. Key details and specifics
3. Important relationships or dependencies
4. Actionable insights if any

Content:
{content[:4000]}"""

    abstract = None
    summary = None

    try:
        # Generate abstract (L0)
        abstract_resp = llm_provider.summarize(abstract_prompt)
        abstract = abstract_resp.strip() if abstract_resp else None
    except LLMProviderError as e:
        logger.warning("Failed to generate L0 abstract: %s", e)

    try:
        # Generate summary (L1)
        summary_resp = llm_provider.summarize(summary_prompt)
        summary = summary_resp.strip() if summary_resp else None
    except LLMProviderError as e:
        logger.warning("Failed to generate L1 summary: %s", e)

    return abstract, summary
