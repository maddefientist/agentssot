from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .settings import get_settings

settings = get_settings()

# Postgres server-side guards: no query pins a connection forever, and no
# transaction can idle indefinitely (protects the pool from a stuck synthesis
# session or a slow recall query taking the whole API down).
_CONNECT_OPTIONS = "-c statement_timeout=30000 -c idle_in_transaction_session_timeout=60000"

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
