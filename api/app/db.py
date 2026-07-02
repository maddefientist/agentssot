from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .settings import get_settings

settings = get_settings()

# Postgres server-side guard: no single query pins a connection forever
# (protects the pool from a slow recall query). We deliberately do NOT set
# idle_in_transaction_session_timeout — the synthesis batch holds one session
# open across multi-second LLM calls, and that guard would terminate it mid-run.
_CONNECT_OPTIONS = "-c statement_timeout=30000"

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_timeout=10,
    connect_args={"options": _CONNECT_OPTIONS},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
