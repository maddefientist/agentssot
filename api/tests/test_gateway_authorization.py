from types import SimpleNamespace
from uuid import uuid4

import pytest

from app import security
from app.models import ApiRole
from app.security import AuthContext


class _Session:
    def __init__(self, key=None, error=None):
        self.key = key
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def get(self, _model, _key_id):
        if self.error:
            raise self.error
        return self.key


def _auth(key_id):
    return AuthContext(
        key_id=str(key_id),
        key_name="gateway-admin",
        role=ApiRole.admin.value,
        namespaces=["default"],
    )


@pytest.mark.parametrize(
    ("active", "role", "namespaces", "expected"),
    [
        (True, ApiRole.admin, ["default"], True),
        (True, ApiRole.admin, ["*"], True),
        (False, ApiRole.admin, ["default"], False),
        (True, ApiRole.writer, ["default"], False),
        (True, ApiRole.admin, ["other"], False),
    ],
)
def test_long_lived_gateway_auth_reloads_current_key(
    monkeypatch, active, role, namespaces, expected
):
    key_id = uuid4()
    key = SimpleNamespace(is_active=active, role=role, namespaces=namespaces)
    monkeypatch.setattr(security, "SessionLocal", lambda: _Session(key=key))

    assert security.auth_context_is_current(
        _auth(key_id), namespace="default", minimum_role=ApiRole.admin.value
    ) is expected


def test_long_lived_gateway_auth_fails_closed_on_bad_id_missing_key_or_db_error(monkeypatch):
    assert not security.auth_context_is_current(
        _auth("not-a-uuid"), namespace="default"
    )

    monkeypatch.setattr(security, "SessionLocal", lambda: _Session(key=None))
    assert not security.auth_context_is_current(
        _auth(uuid4()), namespace="default"
    )

    monkeypatch.setattr(
        security, "SessionLocal", lambda: _Session(error=RuntimeError("database unavailable"))
    )
    assert not security.auth_context_is_current(
        _auth(uuid4()), namespace="default"
    )
