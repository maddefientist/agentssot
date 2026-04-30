"""POST /loadout returns a token-budget-packed bundle for (cwd, device)."""
import os
import pytest
import httpx

BASE = os.environ.get("SSOT_TEST_URL", "http://localhost:8088")
KEY = os.environ.get("SSOT_TEST_API_KEY", "")


@pytest.mark.integration
def test_loadout_for_agentssot_cwd():
    if not KEY:
        pytest.skip("SSOT_TEST_API_KEY not set")
    r = httpx.post(
        f"{BASE}/api/v1/knowledge/loadout",
        headers={"X-Api-Key": KEY},
        json={
            "cwd": "/opt/agentssot",
            "device_id": "hari",
            "namespace": "claude-shared",
            "token_budget": 750,
        },
        timeout=15,
    )
    assert r.status_code == 200
    body = r.json()
    assert "items" in body and "tokens_used" in body
    assert body["tokens_used"] <= 750
    # Must include rules (always loaded)
    assert "rule" in body["items"]
    # cache_key is deterministic given the same inputs
    assert isinstance(body["cache_key"], str) and len(body["cache_key"]) == 64
