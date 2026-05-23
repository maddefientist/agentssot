import asyncio
import logging

from sqlalchemy import text

from ..db import SessionLocal

logger = logging.getLogger("agentssot.synapse.reaper")

_REAP_INTERVAL_SECONDS = 60
_REAP_SQL = text("DELETE FROM synapse_session WHERE last_seen < now() - interval '10 minutes'")


async def reaper_loop() -> None:
    """Delete stale synapse sessions every 60s. CASCADE kills their events."""
    while True:
        try:
            with SessionLocal() as session:
                result = session.execute(_REAP_SQL)
                session.commit()
                if result.rowcount:
                    logger.info("synapse reaper: reaped %d stale sessions", result.rowcount)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("synapse reaper cycle failed")

        await asyncio.sleep(_REAP_INTERVAL_SECONDS)
