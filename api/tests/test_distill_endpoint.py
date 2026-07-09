"""WU3 — POST /api/v1/distill endpoint tests.

Mocks extract() and app.state.llm_provider.distill — no real network, no DB.
"""
from __future__ import annotations

import hashlib
import types
from typing import Literal

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

import app.routers.intake as intake_mod
from app.intake.extract import IntakeExtractionError
from app.llm import LLMProviderError
from app.routers.intake import router as intake_router
from app.security import AuthContext


class FakeSettings:
    def __init__(self, stt_url: str = "http://stt.local/transcribe") -> None:
        self.voice_stack_stt_url = stt_url


class FakeLLMProvider:
    def __init__(self, raw: str = "", raise_exc: Exception | None = None) -> None:
        self._raw = raw
        self._raise = raise_exc
        self.distill_calls: list[str] = []

    def distill(self, transcript: str, model: str | None = None) -> str:
        self.distill_calls.append(transcript)
        if self._raise is not None:
            raise self._raise
        return self._raw


def _fake_auth() -> AuthContext:
    return AuthContext(key_id="k1", key_name="tester", role="admin", namespaces=["*"])


@pytest.fixture(autouse=True)
def _restore_extract():
    """Restore the module-global extract after each test.

    The router calls `extract(...)` as a plain module function, NOT a FastAPI
    dependency, so app.dependency_overrides can't intercept it — patching the
    module attribute is the only thing that keeps these unit tests off the real
    network / yt-dlp. This fixture guarantees the real function is put back.
    """
    original = intake_mod.extract
    yield
    intake_mod.extract = original


def _make_app(
    *,
    extract_returns: tuple[str, dict] | Exception,
    llm: FakeLLMProvider,
    stt_url: str = "http://stt.local/transcribe",
) -> FastAPI:
    app = FastAPI()
    app.state.settings = FakeSettings(stt_url=stt_url)
    app.state.llm_provider = llm

    if isinstance(extract_returns, Exception):
        def fake_extract(source_url, text, media_type, *, stt_url):
            raise extract_returns
    else:
        transcript, provenance = extract_returns

        def fake_extract(source_url, text, media_type, *, stt_url):
            assert stt_url == "http://stt.local/transcribe"
            return transcript, provenance

    # Patch the module global the router closure resolves. dependency_overrides
    # would be a no-op here (extract is not a Depends()). The autouse fixture
    # above restores the original after the test.
    intake_mod.extract = fake_extract
    app.dependency_overrides[intake_mod.require_api_key] = _fake_auth
    app.include_router(intake_router)
    return app


# ---------------------------------------------------------------------------
# success: text request returns provenance + transcript_ref + lessons
# ---------------------------------------------------------------------------


def test_text_request_returns_provenance_transcript_ref_and_lessons():
    transcript = "Pin dependency versions in production."
    provenance = {
        "source_url": "https://example.com/post",
        "media_type": "article",
        "captured_at": "2026-07-09T12:00:00+00:00",
    }
    raw = (
        '{"claim":"Pin dependency versions","citation":"para 1","memory_type":"skill","confidence":0.9}\n'
        '{"claim":"Use Postgres for durability","citation":"para 2","memory_type":"decision","confidence":0.8}'
    )
    llm = FakeLLMProvider(raw=raw)
    app = _make_app(extract_returns=(transcript, provenance), llm=llm)

    client = TestClient(app)
    r = client.post(
        "/api/v1/distill",
        json={
            "source_url": "https://example.com/post",
            "text": transcript,
            "media_type": "article",
            "title": "My Article",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provenance"]["media_type"] == "article"
    assert body["provenance"]["source_url"] == "https://example.com/post"
    assert body["provenance"]["title"] == "My Article"
    assert body["transcript_ref"] == "sha256:" + hashlib.sha256(
        transcript.encode("utf-8")
    ).hexdigest()
    assert len(body["lessons"]) == 2
    assert body["lessons"][0]["claim"] == "Pin dependency versions"
    assert body["lessons"][0]["memory_type"] == "skill"
    assert body["lessons"][1]["memory_type"] == "decision"
    # distill was called with the extracted transcript
    assert llm.distill_calls == [transcript]


# ---------------------------------------------------------------------------
# zero lessons -> 200 empty list
# ---------------------------------------------------------------------------


def test_zero_lessons_returns_200_empty_list():
    transcript = "Nothing useful here."
    provenance = {
        "source_url": None,
        "media_type": "thread",
        "captured_at": "2026-07-09T12:00:00+00:00",
    }
    llm = FakeLLMProvider(raw="")
    app = _make_app(extract_returns=(transcript, provenance), llm=llm)

    client = TestClient(app)
    r = client.post(
        "/api/v1/distill",
        json={"text": transcript, "media_type": "thread"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lessons"] == []
    assert body["transcript_ref"].startswith("sha256:")


# ---------------------------------------------------------------------------
# extraction failure -> 502
# ---------------------------------------------------------------------------


def test_extraction_failure_returns_502():
    llm = FakeLLMProvider(raw="ignored")
    app = _make_app(
        extract_returns=IntakeExtractionError("yt-dlp failed: boom"),
        llm=llm,
    )

    client = TestClient(app)
    r = client.post(
        "/api/v1/distill",
        json={
            "source_url": "https://example.com/v.mp4",
            "media_type": "video",
        },
    )
    assert r.status_code == 502
    # Detail is redacted — internal stderr/hostnames must NOT reach the client.
    assert r.json()["detail"] == "source extraction failed"
    assert "yt-dlp" not in r.json()["detail"]
    assert "boom" not in r.json()["detail"]
    # distill never called when extraction fails
    assert llm.distill_calls == []


# ---------------------------------------------------------------------------
# distill failure -> 502
# ---------------------------------------------------------------------------


def test_distill_failure_returns_502():
    transcript = "Some transcript."
    provenance = {
        "source_url": "https://example.com/a",
        "media_type": "article",
        "captured_at": "2026-07-09T12:00:00+00:00",
    }
    llm = FakeLLMProvider(raise_exc=LLMProviderError("ollama down"))
    app = _make_app(extract_returns=(transcript, provenance), llm=llm)

    client = TestClient(app)
    r = client.post(
        "/api/v1/distill",
        json={"text": transcript, "media_type": "article"},
    )
    assert r.status_code == 502
    # Provider error detail is redacted from the client response.
    assert r.json()["detail"] == "lesson distillation failed"
    assert "ollama down" not in r.json()["detail"]


# ---------------------------------------------------------------------------
# unexpected error -> 500
# ---------------------------------------------------------------------------


def test_unexpected_error_returns_500():
    transcript = "Some transcript."
    provenance = {
        "source_url": "https://example.com/a",
        "media_type": "article",
        "captured_at": "2026-07-09T12:00:00+00:00",
    }

    class ExplodingLLM:
        def distill(self, transcript, model=None):
            raise RuntimeError("kaboom")

    app = _make_app(extract_returns=(transcript, provenance), llm=ExplodingLLM())  # type: ignore[arg-type]
    client = TestClient(app)
    r = client.post(
        "/api/v1/distill",
        json={"text": transcript, "media_type": "article"},
    )
    assert r.status_code == 500


# ---------------------------------------------------------------------------
# auth required via X-API-Key (dependency override removed)
# ---------------------------------------------------------------------------


def test_auth_required_without_override_returns_401():
    transcript = "Some transcript."
    provenance = {
        "source_url": None,
        "media_type": "thread",
        "captured_at": "2026-07-09T12:00:00+00:00",
    }
    llm = FakeLLMProvider(raw="")
    app = _make_app(extract_returns=(transcript, provenance), llm=llm)
    # Drop the auth override so the real require_api_key runs -> needs DB/X-API-Key.
    app.dependency_overrides.pop(intake_mod.require_api_key)

    client = TestClient(app)
    r = client.post(
        "/api/v1/distill",
        json={"text": transcript, "media_type": "thread"},
    )
    # No X-API-Key header -> 401 from the real dependency.
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# request validation: video/audio require source_url, article/thread require text
# ---------------------------------------------------------------------------


def test_video_requires_source_url():
    llm = FakeLLMProvider(raw="")
    app = _make_app(
        extract_returns=("x", {"source_url": "u", "media_type": "video", "captured_at": "t"}),
        llm=llm,
    )
    client = TestClient(app)
    r = client.post("/api/v1/distill", json={"media_type": "video"})
    assert r.status_code == 422


def test_article_requires_text():
    llm = FakeLLMProvider(raw="")
    app = _make_app(
        extract_returns=("x", {"source_url": "u", "media_type": "article", "captured_at": "t"}),
        llm=llm,
    )
    client = TestClient(app)
    r = client.post(
        "/api/v1/distill",
        json={"source_url": "https://example.com/a", "media_type": "article"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# main.py registers /api/v1/distill exactly once
# ---------------------------------------------------------------------------


def test_distill_route_registered_once():
    from app.main import app as main_app

    distill_routes = [
        r for r in main_app.routes if getattr(r, "path", None) == "/api/v1/distill"
    ]
    assert len(distill_routes) == 1