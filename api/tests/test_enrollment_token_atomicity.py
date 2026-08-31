from types import SimpleNamespace
from uuid import uuid4

from app import crud
from app.models import ApiRole


class _Scalars:
    def __init__(self, values):
        self._values = values

    def all(self):
        return self._values


class _Session:
    def __init__(self, candidate, locked):
        self.candidate = candidate
        self.locked = locked
        self.commits = 0
        self.refreshes = 0
        self.rollbacks = 0

    def scalars(self, _statement):
        return _Scalars([self.candidate])

    def scalar(self, statement):
        assert statement._for_update_arg is not None
        assert statement.get_execution_options().get("populate_existing") is True
        return self.locked

    def commit(self):
        self.commits += 1

    def refresh(self, _record):
        self.refreshes += 1

    def rollback(self):
        self.rollbacks += 1

    def in_transaction(self):
        return False


def test_redemption_locks_then_commits_key_and_token_once(monkeypatch):
    token_id = uuid4()
    stale = SimpleNamespace(id=token_id, token_hash="hash")
    locked = SimpleNamespace(
        id=token_id,
        is_active=True,
        expires_at=None,
        times_used=0,
        max_uses=1,
        name_hint="agent",
        role=ApiRole.writer,
        namespaces=["default"],
    )
    session = _Session(stale, locked)
    record = SimpleNamespace(id=uuid4())
    staged = []
    monkeypatch.setattr(crud, "verify_api_key", lambda _plain, _hashed: True)
    monkeypatch.setattr(
        crud,
        "_create_api_key_record_uncommitted",
        lambda **kwargs: staged.append(kwargs) or (record, "ssot_result"),
    )

    actual, plaintext = crud.redeem_enrollment_token(session, "token", "worker")

    assert actual is record
    assert plaintext == "ssot_result"
    assert staged[0]["name"] == "agent-worker"
    assert staged[0]["namespaces"] == ["default"]
    assert locked.times_used == 1
    assert locked.is_active is False
    assert session.commits == 1
    assert session.refreshes == 1
