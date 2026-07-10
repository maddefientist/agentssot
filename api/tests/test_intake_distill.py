from __future__ import annotations

import json

import httpx
import pytest

from app.intake.distill import Lesson, parse_lessons
from app.llm import LLMProviderError
from app.llm.ollama_provider import OllamaLLMProvider
from app.llm.openai_provider import OpenAILLMProvider


# ---------------------------------------------------------------------------
# parse_lessons
# ---------------------------------------------------------------------------


def test_parse_lessons_canned_json_lines_returns_lessons():
    raw = "\n".join(
        [
            json.dumps(
                {
                    "claim": "Always pin dependency versions",
                    "citation": "00:02:14",
                    "memory_type": "skill",
                    "confidence": 0.9,
                }
            ),
            json.dumps(
                {
                    "claim": "We chose Postgres for durability",
                    "citation": "infra-decision paragraph 3",
                    "memory_type": "decision",
                    "confidence": 0.8,
                }
            ),
        ]
    )

    lessons = parse_lessons(raw)

    assert len(lessons) == 2
    assert lessons[0] == {
        "claim": "Always pin dependency versions",
        "citation": "00:02:14",
        "memory_type": "skill",
        "confidence": 0.9,
    }
    assert lessons[1]["memory_type"] == "decision"
    # TypedDict instances are plain dicts.
    assert isinstance(lessons[0], dict)


def test_parse_lessons_malformed_line_skipped_no_raise():
    raw = "\n".join(
        [
            "not valid json at all",
            json.dumps({"claim": "good", "citation": "ref", "memory_type": "fact", "confidence": 0.4}),
            "{ broken json",
            json.dumps({"claim": "second good", "citation": "ref2", "memory_type": "skill", "confidence": 0.6}),
        ]
    )

    # Must not raise; malformed lines simply dropped.
    lessons = parse_lessons(raw)

    assert len(lessons) == 2
    assert [l["claim"] for l in lessons] == ["good", "second good"]


def test_parse_lessons_missing_memory_type_defaults_to_skill():
    raw = json.dumps({"claim": "no type given", "citation": "00:01", "confidence": 0.5})
    lessons = parse_lessons(raw)
    assert len(lessons) == 1
    assert lessons[0]["memory_type"] == "skill"


def test_parse_lessons_invalid_memory_type_defaults_to_skill():
    raw = json.dumps(
        {"claim": "bad type", "citation": "00:02", "memory_type": "weird", "confidence": 0.5}
    )
    lessons = parse_lessons(raw)
    assert len(lessons) == 1
    assert lessons[0]["memory_type"] == "skill"


def test_parse_lessons_invalid_confidence_defaults_to_half():
    raw = json.dumps({"claim": "c", "citation": "r", "memory_type": "skill", "confidence": "high"})
    lessons = parse_lessons(raw)
    assert len(lessons) == 1
    assert lessons[0]["confidence"] == 0.5


def test_parse_lessons_missing_confidence_defaults_to_half():
    raw = json.dumps({"claim": "c", "citation": "r", "memory_type": "skill"})
    lessons = parse_lessons(raw)
    assert lessons[0]["confidence"] == 0.5


def test_parse_lessons_out_of_range_confidence_clamped():
    raw = json.dumps({"claim": "c", "citation": "r", "memory_type": "skill", "confidence": 5.0})
    lessons = parse_lessons(raw)
    assert lessons[0]["confidence"] == 1.0

    raw_low = json.dumps({"claim": "c", "citation": "r", "memory_type": "skill", "confidence": -3.0})
    assert parse_lessons(raw_low)[0]["confidence"] == 0.0


def test_parse_lessons_drops_entries_missing_claim_or_citation():
    raw = "\n".join(
        [
            json.dumps({"citation": "r", "memory_type": "skill", "confidence": 0.5}),  # no claim
            json.dumps({"claim": "c", "memory_type": "skill", "confidence": 0.5}),  # no citation
            json.dumps({"claim": "   ", "citation": "r", "confidence": 0.5}),  # blank claim
            json.dumps({"claim": "c", "citation": "", "confidence": 0.5}),  # blank citation
            json.dumps({"claim": "kept", "citation": "r", "memory_type": "fact", "confidence": 0.4}),
        ]
    )
    lessons = parse_lessons(raw)
    assert len(lessons) == 1
    assert lessons[0]["claim"] == "kept"


def test_parse_lessons_empty_raw_returns_empty_list():
    assert parse_lessons("") == []
    assert parse_lessons("\n\n  \n") == []


def test_parse_lessons_non_dict_json_object_skipped():
    raw = "\n".join(
        [
            json.dumps(["not", "an", "object"]),
            "42",
            "null",
            json.dumps({"claim": "kept", "citation": "r", "memory_type": "skill", "confidence": 0.5}),
        ]
    )
    lessons = parse_lessons(raw)
    assert len(lessons) == 1
    assert lessons[0]["claim"] == "kept"


# ---------------------------------------------------------------------------
# Ollama provider distill
# ---------------------------------------------------------------------------


DISTILL_PROMPT_FRAGMENT = "You are a learning-intake distiller."


def _make_ollama():
    return OllamaLLMProvider(base_url="http://ollama.local:11434", model="llama3")


def test_ollama_distill_returns_stripped_content(monkeypatch):
    captured: dict = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return httpx.Response(
            200,
            json={"message": {"content": "  {\"claim\": \"x\"}  "}},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = _make_ollama()

    result = provider.distill("transcript body")

    assert result == '{"claim": "x"}'
    assert captured["url"] == "http://ollama.local:11434/api/chat"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["model"] == "llama3"
    assert captured["payload"]["options"] == {"num_ctx": 8192}
    assert DISTILL_PROMPT_FRAGMENT in captured["payload"]["messages"][0]["content"]
    assert captured["payload"]["messages"][1]["content"] == "transcript body"


def test_ollama_distill_uses_model_override(monkeypatch):
    captured: dict = {}

    def fake_post(url, json=None, timeout=None):
        captured["payload"] = json
        return httpx.Response(200, json={"message": {"content": "out"}})

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = _make_ollama()
    provider.distill("t", model="qwen-custom")
    assert captured["payload"]["model"] == "qwen-custom"


def test_ollama_distill_raises_on_bad_status(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(500, text="boom"))
    provider = _make_ollama()
    with pytest.raises(LLMProviderError):
        provider.distill("t")


def test_ollama_distill_raises_on_transport_error(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(httpx, "post", boom)
    provider = _make_ollama()
    with pytest.raises(LLMProviderError):
        provider.distill("t")


def test_ollama_distill_raises_on_bad_format(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(200, json={"message": {"content": 7}}))
    provider = _make_ollama()
    with pytest.raises(LLMProviderError):
        provider.distill("t")


def test_ollama_distill_unavailable_raises(monkeypatch):
    provider = OllamaLLMProvider(base_url="", model="")
    with pytest.raises(LLMProviderError):
        provider.distill("t")


# ---------------------------------------------------------------------------
# OpenAI provider distill
# ---------------------------------------------------------------------------


def _make_openai():
    return OpenAILLMProvider(api_key="sk-test", model="gpt-4o-mini")


def test_openai_distill_returns_stripped_content(monkeypatch):
    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "  line\\n  "}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = _make_openai()

    result = provider.distill("transcript body")

    assert result == "line\\n"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["temperature"] == 0.2
    assert captured["payload"]["model"] == "gpt-4o-mini"
    assert DISTILL_PROMPT_FRAGMENT in captured["payload"]["messages"][0]["content"]
    assert captured["payload"]["messages"][1]["content"] == "transcript body"


def test_openai_distill_raises_on_bad_status(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(401, text="unauth"))
    provider = _make_openai()
    with pytest.raises(LLMProviderError):
        provider.distill("t")


def test_openai_distill_raises_on_transport_error(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("x")

    monkeypatch.setattr(httpx, "post", boom)
    provider = _make_openai()
    with pytest.raises(LLMProviderError):
        provider.distill("t")


def test_openai_distill_raises_on_bad_format(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(200, json={"choices": []}))
    provider = _make_openai()
    with pytest.raises(LLMProviderError):
        provider.distill("t")


def test_openai_distill_unavailable_raises():
    provider = OpenAILLMProvider(api_key="", model="gpt-4o-mini")
    with pytest.raises(LLMProviderError):
        provider.distill("t")