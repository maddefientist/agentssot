import asyncio
import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import SessionLocal
from ..embeddings import EmbeddingProvider
from ..llm import LLMProvider
from ..models import Concept, KnowledgeItem, Namespace
from .clustering import cluster_items
from .reconciler import apply_feedback_signals, decay_stale_concepts, reconcile_concepts
from .synthesizer import run_synthesis_batch

logger = logging.getLogger("agentssot.synthesis.loop")


def _gather_recent_knowledge(
    session: Session,
    namespace: str,
    since: datetime,
    limit: int = 500,
) -> list[dict]:
    """Fetch knowledge items created since last synthesis run."""
    stmt = (
        select(KnowledgeItem)
        .where(KnowledgeItem.namespace == namespace)
        .where(KnowledgeItem.created_at > since)
        .where(KnowledgeItem.embedding.is_not(None))
        .order_by(KnowledgeItem.created_at.desc())
        .limit(limit)
    )
    rows = session.scalars(stmt).all()
    return [
        {
            "id": item.id,
            "content": item.content,
            "embedding": list(item.embedding) if item.embedding is not None else None,
            "tags": list(item.tags or []),
            "source": item.source,
            "created_at": item.created_at,
        }
        for item in rows
    ]


def _get_active_concepts(session: Session, namespace: str) -> list[dict]:
    """Load all non-superseded concepts for reconciliation."""
    stmt = (
        select(Concept)
        .where(Concept.namespace == namespace)
        .where(~Concept.tags.any("superseded"))
    )
    rows = session.scalars(stmt).all()
    return [
        {
            "id": str(c.id),
            "type": c.type.value,
            "scope": c.scope.value if hasattr(c.scope, "value") else str(c.scope),
            "title": c.title,
            "content": c.content,
            "confidence": c.confidence,
            "embedding": list(c.embedding) if c.embedding is not None else None,
        }
        for c in rows
    ]


def _find_related_concepts(
    cluster_items_list: list[dict],
    all_concepts: list[dict],
    threshold: float = 0.6,
    max_related: int = 5,
) -> list[dict]:
    """Find existing concepts related to a cluster by embedding similarity."""
    if not all_concepts or not cluster_items_list:
        return []

    embeddings = [it["embedding"] for it in cluster_items_list if it.get("embedding")]
    if not embeddings:
        return []
    centroid = np.mean(embeddings, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        return []

    scored = []
    for concept in all_concepts:
        if not concept.get("embedding"):
            continue
        c_emb = np.array(concept["embedding"])
        sim = float(np.dot(centroid, c_emb) / (centroid_norm * np.linalg.norm(c_emb) + 1e-9))
        if sim >= threshold:
            scored.append((sim, concept))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:max_related]]


def _build_agent_profiles(session: Session, namespace: str, llm_provider, settings) -> int:
    """Analyze recall patterns per agent to update strength topics. Returns profiles updated."""
    from ..models import AgentProfile, RecallEvent
    import json as _json

    profiles = session.scalars(
        select(AgentProfile).where(AgentProfile.namespace == namespace)
    ).all()

    updated = 0
    for profile in profiles:
        recent_queries = session.scalars(
            select(RecallEvent.query_text)
            .where(RecallEvent.agent_key == profile.agent_key)
            .where(RecallEvent.query_text.is_not(None))
            .where(RecallEvent.created_at > datetime.now(UTC) - timedelta(days=30))
            .order_by(RecallEvent.created_at.desc())
            .limit(100)
        ).all()

        if len(recent_queries) < 5:
            continue

        queries_text = "\n".join(q for q in recent_queries if q)
        prompt = (
            f"Given these search queries from agent '{profile.device_name}':\n"
            f"{queries_text}\n\n"
            "Extract the top 5 topic areas as a JSON array of strings. "
            'Example: ["docker", "python", "networking"]\n'
            "Output ONLY the JSON array, nothing else."
        )

        try:
            result = llm_provider.summarize(prompt, model=settings.synthesis_fallback_model)
            topics = _json.loads(result.strip())
            if isinstance(topics, list):
                profile.strengths = [str(t).lower() for t in topics[:10]]
                updated += 1
        except Exception:
            logger.debug("profile topic extraction failed", extra={"agent": profile.agent_key})
            continue

    if updated:
        session.flush()
    return updated


def _run_synthesis_for_namespace(
    namespace: str,
    settings,
    llm_provider: LLMProvider,
    embedding_provider: EmbeddingProvider,
    full_resynthesis: bool = False,
    skip_decay: bool = False,
) -> dict:
    """Run one synthesis cycle for a namespace. Returns stats dict."""
    stats = {"namespace": namespace, "new": 0, "updated": 0, "decayed": 0, "clusters": 0, "feedback_adjustments": 0}

    with SessionLocal() as session:
        if full_resynthesis:
            since = datetime(2020, 1, 1, tzinfo=UTC)
            logger.info("full resynthesis requested", extra={"namespace": namespace})
        else:
            latest_concept = session.scalar(
                select(func.max(Concept.updated_at)).where(Concept.namespace == namespace)
            )
            since = latest_concept or (datetime.now(UTC) - timedelta(days=7))
            min_since = datetime.now(UTC) - timedelta(hours=24)
            if since > min_since:
                since = min_since

        gather_limit = 5000 if full_resynthesis else 500
        items = _gather_recent_knowledge(session, namespace, since, limit=gather_limit)
        if not items:
            logger.info("no new knowledge to synthesize", extra={"namespace": namespace})
            return stats

        logger.info(
            "gathered items for synthesis",
            extra={"namespace": namespace, "item_count": len(items)},
        )

        clusters = cluster_items(
            items,
            similarity_threshold=settings.synthesis_similarity_threshold,
            min_cluster_size=settings.synthesis_min_cluster_size,
        )
        stats["clusters"] = len(clusters)

        if not clusters:
            logger.info("no clusters formed", extra={"namespace": namespace})
            return stats

        all_concepts = _get_active_concepts(session, namespace)

        all_touched_ids: set[UUID] = set()
        for cluster in clusters:
            related = _find_related_concepts(cluster, all_concepts)

            proposals = run_synthesis_batch(
                cluster_items=cluster,
                existing_concepts=related,
                llm_provider=llm_provider,
                synthesis_model=settings.synthesis_model,
                fallback_model=settings.synthesis_fallback_model,
            )

            # Layer 5: Extract agent attribution from cluster sources
            cluster_agent_keys = set()
            for item in cluster:
                src = item.get("source", "")
                if src and (src.startswith("device-") or src.startswith("enroll-") or src.startswith("agent-")):
                    cluster_agent_keys.add(src)

            if proposals:
                result = reconcile_concepts(
                    session=session,
                    namespace=namespace,
                    proposals=proposals,
                    embedding_provider=embedding_provider,
                    embedding_provider_kind=settings.embedding_provider,
                    embedding_dim=settings.embedding_dim,
                    agent_keys=cluster_agent_keys or None,
                )
                stats["new"] += result["new"]
                stats["updated"] += result["updated"]

                for p in proposals:
                    if p.get("matches_existing_id"):
                        try:
                            all_touched_ids.add(UUID(p["matches_existing_id"]))
                        except ValueError:
                            pass

        # --- Feedback integration (Layer 2) ---
        last_synthesis_time = datetime.now(UTC) - timedelta(days=1)
        protected_ids, feedback_adjustments = apply_feedback_signals(
            session, namespace, since=last_synthesis_time,
            feedback_protection_days=settings.synthesis_feedback_protection_days,
        )
        stats["feedback_adjustments"] = feedback_adjustments

        if not skip_decay:
            decayed = decay_stale_concepts(
                session, namespace, all_touched_ids,
                decay_rate=settings.synthesis_confidence_decay,
                min_age_days=settings.synthesis_decay_grace_days,
                decay_floor=settings.synthesis_decay_floor,
                protected_ids=protected_ids,
            )
            stats["decayed"] = decayed

        # --- Agent profile building (Layer 4) ---
        try:
            profiles_updated = _build_agent_profiles(session, namespace, llm_provider, settings)
            if profiles_updated:
                logger.info("agent profiles updated", extra={"namespace": namespace, "count": profiles_updated})
        except Exception:
            logger.exception("agent profile building failed", extra={"namespace": namespace})

        session.commit()

    return stats


async def synthesis_loop(app) -> None:
    """Daily synthesis loop. Runs at the configured hour."""
    settings = app.state.settings

    while True:
        try:
            now = datetime.now(UTC)
            target_hour = settings.synthesis_schedule_hour
            next_run = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            sleep_seconds = (next_run - now).total_seconds()

            logger.info(
                "synthesis loop sleeping until next run",
                extra={"next_run": next_run.isoformat(), "sleep_seconds": int(sleep_seconds)},
            )
            await asyncio.sleep(sleep_seconds)

            llm_provider = app.state.llm_provider
            embedding_provider = app.state.embedding_provider

            if not llm_provider.is_available:
                logger.warning("synthesis skipped; LLM provider unavailable")
                continue

            with SessionLocal() as session:
                namespaces = [ns.name for ns in session.scalars(select(Namespace)).all()]

            for ns in namespaces:
                try:
                    stats = _run_synthesis_for_namespace(
                        namespace=ns,
                        settings=settings,
                        llm_provider=llm_provider,
                        embedding_provider=embedding_provider,
                    )
                    logger.info("synthesis complete", extra=stats)
                except Exception:
                    logger.exception("synthesis failed for namespace", extra={"namespace": ns})

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("synthesis loop iteration failed")
            await asyncio.sleep(3600)
