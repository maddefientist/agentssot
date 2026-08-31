"""Regression: GET /api/v1/entities must enforce per-namespace authorization,
not just a global writer/admin role check.

Before this, list_entities checked only auth.role in (writer, admin) — a
writer key scoped to namespace A could list entities in namespace B by
supplying it as a query param. ensure_namespace_access closes that gap by
also requiring the namespace be in the caller's authorized set.
"""
import asyncio
import os

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@127.0.0.1/test")

import pytest
from fastapi import HTTPException

from app.routers import entities as entities_router
from app.security import AuthContext, ApiRole


class _FakeResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return self

    def all(self):
        return self._items

    def __iter__(self):
        return iter(self._items)


class _FakeSession:
    """Records whether a query was ever issued, to prove denial short-circuits."""

    def __init__(self, entities=None):
        self._entities = entities or []
        self.executed = 0

    def execute(self, _stmt):
        self.executed += 1
        return _FakeResult(self._entities)


def _auth(namespaces, role=ApiRole.writer.value):
    return AuthContext(key_id="k1", key_name="tester", role=role, namespaces=namespaces)


def _call(namespace, auth, session):
    return asyncio.run(
        entities_router.list_entities(namespace=namespace, limit=200, session=session, auth=auth)
    )


def test_allowed_namespace_and_role_queries_and_returns():
    session = _FakeSession(entities=[])
    auth = _auth(["default"], role=ApiRole.writer.value)
    result = _call("default", auth, session)
    assert result == []
    assert session.executed == 1


def test_admin_role_with_wildcard_namespace_allowed():
    session = _FakeSession(entities=[])
    auth = _auth(["*"], role=ApiRole.admin.value)
    result = _call("some-other-namespace", auth, session)
    assert result == []
    assert session.executed == 1


def test_denied_namespace_raises_403_before_any_query():
    """A key authorized for a different namespace must be refused, and must
    never reach the database."""
    session = _FakeSession(entities=[])
    auth = _auth(["namespace-a"], role=ApiRole.writer.value)
    with pytest.raises(HTTPException) as exc:
        _call("namespace-b", auth, session)
    assert exc.value.status_code == 403
    assert session.executed == 0


def test_reader_role_denied_even_in_authorized_namespace():
    """Role check is still enforced: reader keys cannot list entities even
    within their own authorized namespace."""
    session = _FakeSession(entities=[])
    auth = _auth(["default"], role=ApiRole.reader.value)
    with pytest.raises(HTTPException) as exc:
        _call("default", auth, session)
    assert exc.value.status_code == 403
    assert session.executed == 0
