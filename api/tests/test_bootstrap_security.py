from types import SimpleNamespace

import pytest

from app import startup


class _Session:
    def __init__(self, count=0):
        self.count = count
        self.added = []
        self.commits = 0

    def scalar(self, _statement):
        return self.count

    def add(self, value):
        self.added.append(value)

    def commit(self):
        self.commits += 1


def _settings(key):
    return SimpleNamespace(
        bootstrap_admin_api_key=key,
        bootstrap_namespace_list=["default"],
    )


def test_first_boot_fails_closed_without_operator_supplied_key():
    session = _Session(count=0)

    with pytest.raises(RuntimeError, match="BOOTSTRAP_ADMIN_API_KEY"):
        startup._bootstrap_admin_key_if_needed(session, _settings(""))

    assert session.added == []
    assert session.commits == 0


def test_existing_database_does_not_require_bootstrap_secret():
    session = _Session(count=1)
    startup._bootstrap_admin_key_if_needed(session, _settings(""))
    assert session.added == []


def test_bootstrap_key_is_hashed_and_never_logged(monkeypatch, caplog):
    plaintext = "ssot_" + ("a" * 40)
    session = _Session(count=0)
    monkeypatch.setattr(startup, "hash_api_key", lambda value: f"hashed:{value[-4:]}")

    startup._bootstrap_admin_key_if_needed(session, _settings(plaintext))

    assert len(session.added) == 1
    assert session.added[0].key_hash == "hashed:aaaa"
    assert session.commits == 1
    assert plaintext not in caplog.text
    assert "supplied by the operator" in caplog.text
