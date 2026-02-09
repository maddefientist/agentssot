import logging

from sqlalchemy import func, select, text

from .db import SessionLocal
from .models import ApiKey, ApiRole, Namespace
from .security import generate_api_key, hash_api_key

logger = logging.getLogger("agentssot.startup")


def initialize_system(settings) -> None:
    with SessionLocal() as session:
        _bootstrap_namespaces(session, settings)
        _bootstrap_admin_key_if_needed(session, settings)
        _ensure_embedding_dim(session, settings)
        _maybe_enable_hnsw_indexes(session, settings)


def _bootstrap_namespaces(session, settings) -> None:
    namespace_count = session.scalar(select(func.count()).select_from(Namespace)) or 0

    if namespace_count == 0:
        session.add(Namespace(name="default"))
        session.commit()
        logger.info("created default namespace during bootstrap")

    required = set(settings.bootstrap_namespace_list)
    required.add("default")

    existing = set(session.scalars(select(Namespace.name).where(Namespace.name.in_(required))).all())
    missing = sorted(required - existing)
    if missing:
        for name in missing:
            session.add(Namespace(name=name))
        session.commit()
        logger.info("created missing bootstrap namespaces", extra={"event_count": len(missing)})


def _bootstrap_admin_key_if_needed(session, settings) -> None:
    key_count = session.scalar(select(func.count()).select_from(ApiKey)) or 0
    if key_count > 0:
        return

    plaintext = generate_api_key()
    key = ApiKey(
        name="bootstrap-admin",
        key_hash=hash_api_key(plaintext),
        role=ApiRole.admin,
        namespaces=settings.bootstrap_namespace_list,
        is_active=True,
    )
    session.add(key)
    session.commit()

    logger.warning("BOOTSTRAP_ADMIN_API_KEY=%s", plaintext)


def _maybe_enable_hnsw_indexes(session, settings) -> None:
    if not settings.enable_hnsw_index:
        return

    try:
        session.execute(text("SELECT ssot_try_create_hnsw_indexes()"))
        session.commit()
        logger.info("attempted HNSW index creation")
    except Exception as exc:
        session.rollback()
        logger.warning("HNSW index creation skipped due to error: %s", exc)


def _ensure_embedding_dim(session, settings) -> None:
    """Keep vector column dimensions aligned with EMBEDDING_DIM.

    db/init/001_init.sql uses a fixed initial dimension for first boot. This
    startup step attempts to migrate vector columns to the configured dimension
    without requiring a manual DB re-init.
    """

    desired = int(settings.embedding_dim)
    if desired <= 0:
        return

    # Get declared type for the vector columns (ex: 'vector(1536)').
    rows = session.execute(
        text(
            """
            SELECT c.relname AS table_name,
                   a.attname AS column_name,
                   format_type(a.atttypid, a.atttypmod) AS type_repr
              FROM pg_attribute a
              JOIN pg_class c ON c.oid = a.attrelid
              JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = 'public'
               AND c.relname IN ('knowledge_items', 'requirements', 'events')
               AND a.attname = 'embedding'
               AND a.attnum > 0
               AND NOT a.attisdropped
            """
        )
    ).all()

    # Parse the current dimension (if present).
    current_dims: dict[str, int | None] = {}
    for table_name, _col, type_repr in rows:
        dim = None
        if isinstance(type_repr, str) and type_repr.startswith("vector(") and type_repr.endswith(")"):
            try:
                dim = int(type_repr[len("vector(") : -1])
            except ValueError:
                dim = None
        current_dims[str(table_name)] = dim

    if not rows:
        return

    # If all columns already match, nothing to do.
    if all((d == desired) for d in current_dims.values() if d is not None):
        return

    # HNSW indexes (if present) can block ALTER TYPE; drop them first (safe).
    try:
        session.execute(text("DROP INDEX IF EXISTS idx_knowledge_items_embedding_hnsw"))
        session.execute(text("DROP INDEX IF EXISTS idx_requirements_embedding_hnsw"))
        session.execute(text("DROP INDEX IF EXISTS idx_events_embedding_hnsw"))
        session.commit()
    except Exception:
        session.rollback()

    # Attempt migration. If embeddings already exist at a different dimension,
    # Postgres will reject the cast; we log and continue (service still runs,
    # but embeddings/backfill will fail until corrected).
    try:
        session.execute(text(f"ALTER TABLE knowledge_items ALTER COLUMN embedding TYPE VECTOR({desired})"))
        session.execute(text(f"ALTER TABLE requirements ALTER COLUMN embedding TYPE VECTOR({desired})"))
        session.execute(text(f"ALTER TABLE events ALTER COLUMN embedding TYPE VECTOR({desired})"))
        session.commit()
        logger.info("ensured embedding vector dimension", extra={"embedding_dim": desired})
    except Exception as exc:
        session.rollback()
        logger.warning(
            "failed to migrate embedding vector dimension; consider re-init or clearing existing embeddings: %s",
            exc,
            extra={"embedding_dim": desired},
        )
