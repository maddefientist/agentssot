import asyncio
import logging

from fastapi import HTTPException

from . import crud
from .db import SessionLocal
from .llm import LLMProviderError

logger = logging.getLogger("agentssot.background")


async def compaction_loop(app) -> None:
    settings = app.state.settings
    interval = max(int(settings.compaction_interval_seconds), 5)

    while True:
        try:
            _run_compaction_cycle(app)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("compaction cycle failed")

        await asyncio.sleep(interval)


def _run_compaction_cycle(app) -> None:
    settings = app.state.settings
    llm_provider = app.state.llm_provider
    embedding_provider = app.state.embedding_provider

    if not settings.effective_compaction_enabled:
        return

    if not llm_provider.is_available:
        logger.warning("compaction skipped; summarizer unavailable: %s", llm_provider.unavailable_reason)
        return

    with SessionLocal() as session:
        candidates = crud.find_compaction_candidates(
            session,
            event_threshold=settings.compaction_event_threshold,
            char_threshold=settings.compaction_char_threshold,
        )

        for candidate in candidates:
            try:
                result = crud.summarize_and_archive_session(
                    session=session,
                    namespace=candidate["namespace"],
                    session_id=candidate["session_id"],
                    project_id=candidate["project_id"],
                    llm_provider=llm_provider,
                    embedding_provider=embedding_provider,
                    settings=settings,
                )
                logger.info(
                    "compacted session",
                    extra={
                        "namespace": result["namespace"],
                        "session_id": result["session_id"],
                        "event_count": result["archived_events"],
                    },
                )
            except HTTPException as exc:
                session.rollback()
                logger.warning(
                    "compaction skipped for candidate: %s",
                    exc.detail,
                    extra={"namespace": candidate["namespace"], "session_id": candidate["session_id"]},
                )
            except LLMProviderError as exc:
                session.rollback()
                logger.warning(
                    "compaction failed due to LLM provider error: %s",
                    exc,
                    extra={"namespace": candidate["namespace"], "session_id": candidate["session_id"]},
                )
            except Exception:
                session.rollback()
                logger.exception(
                    "unexpected compaction failure",
                    extra={"namespace": candidate["namespace"], "session_id": candidate["session_id"]},
                )
