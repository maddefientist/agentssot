import json
import logging

from ..llm import LLMProvider, LLMProviderError

logger = logging.getLogger("agentssot.synthesis.synthesizer")


def _format_facts(items: list[dict]) -> str:
    lines = []
    for i, item in enumerate(items, 1):
        tags_str = f" [{', '.join(item.get('tags', []))}]" if item.get("tags") else ""
        source = item.get("source", "")
        source_str = f" (source: {source})" if source else ""
        lines.append(f"{i}. {item['content'][:600]}{tags_str}{source_str}")
    return "\n".join(lines)


def _format_existing_concepts(concepts: list[dict]) -> str:
    if not concepts:
        return ""
    lines = []
    for c in concepts:
        lines.append(
            f"ID: {c['id']} | Type: {c['type']} | Title: {c['title']} | "
            f"Confidence: {c['confidence']:.2f}\n  {c['content'][:400]}"
        )
    return "\n\n".join(lines)


def _parse_synthesis_response(raw: str) -> list[dict]:
    """Parse LLM output as JSON lines. Tolerant of markdown fences and extra text."""
    results = []

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)

    for line in cleaned.split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if not all(k in obj for k in ("type", "title", "content", "confidence")):
                logger.warning("skipping concept missing required fields: %s", list(obj.keys()))
                continue
            if obj["type"] not in ("mental_model", "relationship", "principle", "skill"):
                logger.warning("skipping concept with unknown type: %s", obj["type"])
                continue
            results.append(obj)
        except json.JSONDecodeError:
            logger.debug("skipping non-JSON line in synthesis output")
            continue

    return results


def run_synthesis_batch(
    cluster_items: list[dict],
    existing_concepts: list[dict],
    llm_provider: LLMProvider,
    synthesis_model: str,
    fallback_model: str | None = None,
) -> list[dict]:
    """Run synthesis on a cluster of items, returning parsed concept proposals."""
    facts_text = _format_facts(cluster_items)
    concepts_text = _format_existing_concepts(existing_concepts)

    try:
        raw_output = llm_provider.synthesize_concepts(
            facts=facts_text,
            existing_concepts=concepts_text,
            model_override=synthesis_model,
            fallback_model=fallback_model,
        )
    except LLMProviderError:
        logger.warning("synthesis LLM call failed for cluster", exc_info=True)
        return []

    proposals = _parse_synthesis_response(raw_output)

    evidence_ids = [str(item["id"]) for item in cluster_items if item.get("id")]
    for proposal in proposals:
        proposal.setdefault("scope", "global")
        proposal.setdefault("scope_ref", None)
        proposal.setdefault("matches_existing_id", None)
        proposal.setdefault("is_contradiction", False)
        proposal["evidence_item_ids"] = evidence_ids

    logger.info("synthesis batch produced %d concept proposals", len(proposals))
    return proposals
