from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import cortex
from app.models import ApiRole
from app.security import AuthContext


def _auth(namespaces=None):
    return AuthContext(
        key_id="key-id",
        key_name="cortex-client",
        role=ApiRole.writer.value,
        namespaces=namespaces or ["default"],
    )


def test_cortex_always_uses_revocable_database_api_key(monkeypatch):
    expected = _auth()
    calls = []

    def resolve(*, x_api_key, session):
        calls.append((x_api_key, session))
        return expected

    monkeypatch.setattr(cortex, "require_api_key", resolve)
    session = SimpleNamespace()

    assert cortex._require_cortex_key("formerly-special-token", session) is expected
    assert calls == [("formerly-special-token", session)]


def test_cortex_missing_key_rejects_before_database_lookup(monkeypatch):
    monkeypatch.setattr(
        cortex,
        "require_api_key",
        lambda **_kwargs: pytest.fail("database lookup should not run"),
    )
    with pytest.raises(HTTPException) as exc:
        cortex._require_cortex_key(None, SimpleNamespace())
    assert exc.value.status_code == 401


def test_cortex_namespace_is_never_bypassed():
    with pytest.raises(HTTPException) as exc:
        cortex._enforce_cortex_ns(
            _auth(["allowed"]),
            "forbidden",
            {ApiRole.writer.value, ApiRole.admin.value},
        )
    assert exc.value.status_code == 403
