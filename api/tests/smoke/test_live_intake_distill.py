"""WU4 — Manual live acceptance harness for the distill endpoint.

Skipped by default. Enable by setting RUN_LIVE_INTAKE=1 plus the three env vars:
  SSOT_TEST_URL   base URL of the running API (e.g. http://localhost:8088)
  SSOT_API_KEY    X-API-Key value
  LIVE_INTAKE_URL the video URL to ingest/distill (must be operator-provided)
"""
import os

import pytest
import httpx

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_INTAKE") != "1",
    reason="set RUN_LIVE_INTAKE=1 (and SSOT_TEST_URL/SSOT_API_KEY/LIVE_INTAKE_URL) to run",
)


def test_live_intake_distill():
    base = os.getenv("SSOT_TEST_URL", "").rstrip("/")
    api_key = os.getenv("SSOT_API_KEY")
    live_url = os.getenv("LIVE_INTAKE_URL")

    assert base, "SSOT_TEST_URL env var must be set"
    assert api_key, "SSOT_API_KEY env var must be set"
    assert live_url, "LIVE_INTAKE_URL env var must be set (no hardcoded video URL)"

    payload = {
        "source_url": live_url,
        "media_type": "video",
        "title": "Live intake smoke",
    }
    headers = {"X-API-Key": api_key}

    with httpx.Client(timeout=600.0) as client:
        resp = client.post(f"{base}/api/v1/distill", json=payload, headers=headers)

    assert resp.status_code == 200, (
        f"expected 200, got {resp.status_code}; body excerpt: {resp.text[:500]}"
    )

    body = resp.json()
    transcript_ref = body.get("transcript_ref", "")
    assert isinstance(transcript_ref, str) and transcript_ref.startswith("sha256:"), (
        f"transcript_ref missing/invalid: {transcript_ref!r}; body excerpt: {str(body)[:500]}"
    )

    lessons = body.get("lessons")
    assert lessons, f"lessons missing/empty; body excerpt: {str(body)[:500]}"

    required = {"claim", "citation", "memory_type", "confidence"}
    for i, lesson in enumerate(lessons):
        missing = required - set(lesson.keys())
        assert not missing, (
            f"lesson {i} missing keys {missing}; lesson excerpt: {str(lesson)[:500]}"
        )