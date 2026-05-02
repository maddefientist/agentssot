"""Default /recall (no bucketed flag) returns the existing TieredRecallResponse shape.

Critical: existing callers must keep working through Phase 1.
"""
import os
import pytest
import httpx

BASE = os.environ.get("SSOT_TEST_URL", "http://localhost:8088")
KEY = os.environ.get("SSOT_TEST_API_KEY", "")


@pytest.mark.integration
def test_recall_default_returns_flat_results():
    if not KEY:
        pytest.skip("SSOT_TEST_API_KEY not set")
    r = httpx.post(
        f"{BASE}/api/v1/knowledge/recall",
        headers={"X-Api-Key": KEY},
        json={"query": "ssh unraid", "namespace": "claude-shared", "bucketed": False, "limit": 3},
        timeout=15,
    )
    assert r.status_code == 200
    body = r.json()
    # Flat shape: top-level "results" list
    assert "results" in body
    assert isinstance(body["results"], list)
    # Should not have bucketed shape
    assert "buckets" not in body
