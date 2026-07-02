import os
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://ssot:test@localhost:5432/ssot")

from app import alerting


class _Resp:
    def __init__(self, status_code): self.status_code = status_code


def test_post_alert_sends_expected_payload(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _Resp(200)

    monkeypatch.setattr(alerting.httpx, "post", fake_post)
    ok = alerting.post_alert(
        "http://sink.local/hook", "synthesis.model_missing", "error",
        "both models gone", {"missing": ["x"]}, host_label="hari",
    )
    assert ok is True
    assert captured["url"] == "http://sink.local/hook"
    body = captured["json"]
    assert body["source"] == "hive"
    assert body["host"] == "hari"
    assert body["severity"] == "error"
    assert body["event"] == "synthesis.model_missing"
    assert body["detail"] == {"missing": ["x"]}
    assert "timestamp" in body


def test_post_alert_noops_when_disabled_or_no_url(monkeypatch):
    def boom(*a, **k):  # must never be called
        raise AssertionError("httpx.post should not be called")
    monkeypatch.setattr(alerting.httpx, "post", boom)
    assert alerting.post_alert("", "e", "info", "m") is False
    assert alerting.post_alert("http://x", "e", "info", "m", enabled=False) is False


def test_post_alert_swallows_errors(monkeypatch):
    def raiser(*a, **k):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(alerting.httpx, "post", raiser)
    assert alerting.post_alert("http://x", "e", "error", "m") is False  # no raise