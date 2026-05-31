import asyncio

from app.gateway.router import IntentRouter, parse_classifier_response


def run(coro):
    return asyncio.run(coro)


# --- rules table ---
def test_rule_recall_to_hive_tool():
    intent, args = run(IntentRouter().classify("recall what we decided about the gateway"))
    assert intent == "hive-tool"
    assert args == {}


def test_rule_teach_to_hive_tool():
    intent, _ = run(IntentRouter().classify("remember that the ladder starts with Opus"))
    assert intent == "hive-tool"


def test_rule_scan_fleet_to_dispatch():
    intent, _ = run(IntentRouter().classify("scan the fleet for me"))
    assert intent == "dispatch"


def test_rule_briefing():
    intent, _ = run(IntentRouter().classify("give me the morning briefing"))
    assert intent == "briefing"


# --- explicit intent short-circuits ---
def test_explicit_intent_wins():
    intent, _ = run(IntentRouter().classify("scan the fleet", explicit="orchestrate"))
    assert intent == "orchestrate"


def test_explicit_invalid_intent_ignored_falls_to_rules():
    intent, _ = run(IntentRouter().classify("scan the fleet", explicit="bogus"))
    assert intent == "dispatch"


# --- classifier fallback ---
def test_classifier_used_when_rules_miss():
    async def fake(_text):
        return '{"intent": "orchestrate", "args": {"depth": "deep"}}'

    intent, args = run(IntentRouter(classifier=fake).classify("ponder the nature of our architecture"))
    assert intent == "orchestrate"
    assert args == {"depth": "deep"}


def test_no_classifier_misses_default_to_chat_local():
    intent, _ = run(IntentRouter().classify("just chatting about nothing in particular"))
    assert intent == "chat-local"


def test_classifier_exception_defaults_chat_local():
    async def boom(_text):
        raise RuntimeError("ollama down")

    intent, _ = run(IntentRouter(classifier=boom).classify("freeform text the rules will not catch zzz"))
    assert intent == "chat-local"


# --- parse_classifier_response ---
def test_parse_bare_token():
    assert parse_classifier_response("dispatch") == ("dispatch", {})


def test_parse_json_object():
    assert parse_classifier_response('{"intent":"hive-tool","args":{"k":1}}') == ("hive-tool", {"k": 1})


def test_parse_json_wrapped_in_prose():
    raw = 'Sure! {"intent": "briefing", "args": {}} hope that helps'
    assert parse_classifier_response(raw) == ("briefing", {})


def test_parse_invalid_intent_defaults():
    assert parse_classifier_response('{"intent":"nuke_everything"}') == ("chat-local", {})


def test_parse_garbage_defaults():
    assert parse_classifier_response("???not json???") == ("chat-local", {})


def test_parse_empty_defaults():
    assert parse_classifier_response("") == ("chat-local", {})


def test_parse_non_dict_args_coerced():
    assert parse_classifier_response('{"intent":"dispatch","args":"oops"}') == ("dispatch", {})
