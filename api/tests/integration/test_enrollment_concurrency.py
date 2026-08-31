"""Real-PostgreSQL admission race test; requires an explicit isolated database."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import os
from threading import Barrier
from uuid import uuid4

from fastapi import HTTPException
import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import sessionmaker

from app import crud
from app.models import ApiKey, EnrollmentToken, Namespace


pytestmark = pytest.mark.integration


@pytest.mark.parametrize("max_uses", [1, 3])
def test_concurrent_redemption_issues_exactly_max_uses(max_uses):
    database_url = os.environ.get("AGENTSSOT_TEST_DATABASE_URL", "")
    if not database_url:
        pytest.skip("set AGENTSSOT_TEST_DATABASE_URL to an isolated PostgreSQL database")

    admin_engine = create_engine(database_url)
    schema = f"enrollment_test_{uuid4().hex}"
    with admin_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(
        database_url,
        pool_size=max_uses + 3,
        max_overflow=0,
        connect_args={"options": f"-csearch_path={schema}"},
    )
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    namespace = f"concurrency-{uuid4().hex}"
    hint = f"race-{uuid4().hex}"
    with engine.begin() as connection:
        connection.execute(text("CREATE TYPE api_role AS ENUM ('reader', 'writer', 'admin')"))
        connection.execute(text("CREATE TABLE namespaces (name TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT now())"))
        connection.execute(text("CREATE TABLE api_keys (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), name TEXT NOT NULL, key_hash TEXT NOT NULL, role api_role NOT NULL, namespaces TEXT[] NOT NULL DEFAULT ARRAY['default']::TEXT[], is_active BOOLEAN NOT NULL DEFAULT true, created_at TIMESTAMPTZ NOT NULL DEFAULT now())"))
        connection.execute(text("CREATE TABLE enrollment_tokens (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), token_hash TEXT NOT NULL, role api_role NOT NULL, namespaces TEXT[] NOT NULL DEFAULT ARRAY['default']::TEXT[], name_hint TEXT, max_uses INTEGER NOT NULL DEFAULT 1, times_used INTEGER NOT NULL DEFAULT 0, expires_at TIMESTAMPTZ, is_active BOOLEAN NOT NULL DEFAULT true, created_at TIMESTAMPTZ NOT NULL DEFAULT now())"))

    try:
        with sessions() as session:
            session.add(Namespace(name=namespace))
            session.commit()
            token_record, plaintext = crud.create_enrollment_token(
                session,
                role="writer",
                namespaces=[namespace],
                name_hint=hint,
                max_uses=max_uses,
                expires_at=datetime.now(UTC) + timedelta(minutes=5),
            )
            token_id = token_record.id

        worker_count = max_uses + 2
        barrier = Barrier(worker_count)

        def redeem(index):
            with sessions() as session:
                barrier.wait(timeout=10)
                try:
                    record, _key = crud.redeem_enrollment_token(
                        session,
                        plaintext_token=plaintext,
                        key_name=f"worker-{index}",
                    )
                    return ("ok", record.id)
                except HTTPException as exc:
                    return ("rejected", exc.status_code)

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            outcomes = list(pool.map(redeem, range(worker_count)))

        successes = [value for kind, value in outcomes if kind == "ok"]
        rejected = [value for kind, value in outcomes if kind == "rejected"]
        assert len(successes) == max_uses
        assert len(set(successes)) == max_uses
        assert rejected == [401] * (worker_count - max_uses)

        with sessions() as session:
            token = session.get(EnrollmentToken, token_id)
            assert token.times_used == max_uses
            assert token.is_active is False
            issued = session.scalar(select(func.count()).select_from(ApiKey).where(ApiKey.name.like(f"{hint}-%")))
            assert issued == max_uses
    finally:
        engine.dispose()
        with admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin_engine.dispose()
