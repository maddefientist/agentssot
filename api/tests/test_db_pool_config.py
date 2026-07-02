import os
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://ssot:test@localhost:5432/ssot")

from app import db


def test_engine_pool_is_bounded():
    eng = db.engine
    assert eng.pool.size() == 10
    assert eng.pool._max_overflow == 20
    assert eng.pool._timeout == 10


def test_engine_sets_statement_timeout():
    opts = db.engine.url
    assert opts is not None
    assert "statement_timeout" in db._CONNECT_OPTIONS