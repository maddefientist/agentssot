from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app import security
from app.models import ApiRole
from app.security import AuthContext


class _Session:
    def __init__(self, key=None, error=None):
        self.key = key
        self.error = error

    def get(self, _model, _key_id):
        if self.error:
            raise self.error
        return self.key


@pytest.fixture(autouse=True)
def _empty_auth_cache():
    security.clear_auth_cache()
    yield
    security.clear_auth_cache()


def _cached_auth(key_id):
    return AuthContext(
        key_id=str(key_id),
        key_name="before",
        role=ApiRole.admin.value,
        namespaces=["default", "removed-later"],
    )


def test_cached_bcrypt_match_rechecks_revocation_on_next_request():
    plaintext = "ssot_test_cached_key"
    security._auth_cache_set(plaintext, _cached_auth(uuid4()))

    with pytest.raises(HTTPException) as exc:
        security.require_api_key(plaintext, _Session(key=None))

    assert exc.value.status_code == 401
    assert security._auth_cache_get(plaintext) is None


def test_cached_bcrypt_match_refreshes_role_and_namespaces():
    plaintext = "ssot_test_cached_key"
    key_id = uuid4()
    security._auth_cache_set(plaintext, _cached_auth(key_id))
    current = SimpleNamespace(
        id=key_id,
        name="after",
        role=ApiRole.writer,
        namespaces=["default"],
        is_active=True,
    )

    auth = security.require_api_key(plaintext, _Session(key=current))

    assert auth.key_name == "after"
    assert auth.role == ApiRole.writer.value
    assert auth.namespaces == ["default"]


def test_cached_auth_database_failure_fails_closed():
    plaintext = "ssot_test_cached_key"
    security._auth_cache_set(plaintext, _cached_auth(uuid4()))

    with pytest.raises(RuntimeError, match="database unavailable"):
        security.require_api_key(
            plaintext,
            _Session(error=RuntimeError("database unavailable")),
        )
