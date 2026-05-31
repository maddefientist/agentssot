from app.gateway.protocol import Event, InboundMessage


def test_inbound_parses_minimal():
    msg = InboundMessage.from_dict({"text": "scan fleet", "session_id": "s1"})
    assert msg.text == "scan fleet"
    assert msg.session_id == "s1"
    assert msg.intent is None


def test_inbound_carries_explicit_intent():
    msg = InboundMessage.from_dict(
        {"text": "x", "session_id": "s1", "intent": "dispatch"}
    )
    assert msg.intent == "dispatch"


def test_inbound_tolerates_missing_fields():
    msg = InboundMessage.from_dict({})
    assert msg.text == ""
    assert msg.session_id == ""
    assert msg.intent is None


def test_event_token_serializes():
    assert Event.token("hello").to_dict() == {"type": "token", "data": "hello"}


def test_event_event_serializes():
    e = Event.event({"fallover": True, "to": "deepseek-v4-pro"})
    assert e.to_dict() == {
        "type": "event",
        "data": {"fallover": True, "to": "deepseek-v4-pro"},
    }


def test_event_error_serializes():
    d = Event.error("boom", retryable=True).to_dict()
    assert d["type"] == "error"
    assert d["data"] == {"message": "boom", "retryable": True}


def test_event_error_defaults_not_retryable():
    assert Event.error("nope").to_dict()["data"]["retryable"] is False


def test_event_done_serializes_meta():
    d = Event.done({"model": "opus"}).to_dict()
    assert d == {"type": "done", "data": {"model": "opus"}}


def test_event_done_defaults_empty():
    assert Event.done().to_dict() == {"type": "done", "data": {}}
