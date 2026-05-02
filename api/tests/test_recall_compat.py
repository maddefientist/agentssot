"""Legacy /recall path (bucketed=False opt-out) still returns the flat
TieredRecallResponse shape for callers that haven't migrated to bucketed=True.

After T6.1 the default is bucketed=True; this test locks the explicit-opt-out
path so legacy callers (UI Search tab, scripts) keep working.
"""
import os
import pytest
import httpx

BASE = os.environ.get("SSOT_TEST_URL", "http://localhost:8088")
KEY = os.environ.get("SSOT_TEST_API_KEY", "")


@pytest.mark.integration
def test_recall_legacy_path_with_bucketed_false():
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
