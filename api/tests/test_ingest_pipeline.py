"""End-to-end ingest: content → classify → layer-compute → persist."""
import os
import uuid
import pytest
import httpx

BASE = os.environ.get("SSOT_TEST_URL", "http://localhost:8088")
KEY = os.environ.get("SSOT_TEST_API_KEY", "")


@pytest.mark.integration
def test_ingest_classifies_and_populates_layers():
    if not KEY:
        pytest.skip("no test key")

    body = {
        "content": "ssh unraid",
        "namespace": "claude-shared",
        "tags": ["plan1-test", str(uuid.uuid4())],
    }
    r = httpx.post(
        f"{BASE}/api/v1/knowledge/ingest",
        headers={"X-Api-Key": KEY},
        json=body, timeout=30,
    )
    assert r.status_code == 200, r.text
    item_id = r.json()["id"]

    # Fetch via /expand to verify layers + memory_type
    e = httpx.get(
        f"{BASE}/api/v1/knowledge/items/{item_id}/expand?layer=full",
        headers={"X-Api-Key": KEY}, timeout=10,
    )
    assert e.status_code == 200
    data = e.json()
    assert data["abstract"], "abstract should be populated"
    assert data["summary"], "summary should be populated"
    # Live classifier should call this a command
    # (skipped if classifier unavailable — record gets fact/low-conf)


@pytest.mark.integration
def test_low_confidence_ingest_lands_in_review_queue():
    if not KEY:
        pytest.skip("no test key")
    # Ambiguous content classifier should be unsure about
    body = {
        "content": "things and stuff and so on and so forth",
        "namespace": "claude-shared",
        "tags": ["plan1-low-conf-test"],
    }
    r = httpx.post(
        f"{BASE}/api/v1/knowledge/ingest",
        headers={"X-Api-Key": KEY},
        json=body, timeout=30,
    )
    assert r.status_code == 200
    # Review queue check happens in T2.7 once /admin/review-queue exists


@pytest.mark.integration
def test_contradiction_creates_review_queue_entry():
    if not KEY:
        pytest.skip("no test key")
    seed = httpx.post(
        f"{BASE}/api/v1/knowledge/ingest",
        headers={"X-Api-Key": KEY},
        json={
            "content": "Never access fakeunraid — this host is OFF LIMITS",
            "namespace": "claude-shared",
            "memory_type": "rule",
            "tags": ["plan1-contradiction-test", "rule"],
        },
        timeout=30,
    )
    assert seed.status_code == 200, seed.text

    cmd = httpx.post(
        f"{BASE}/api/v1/knowledge/ingest",
        headers={"X-Api-Key": KEY},
        json={
            "content": "ssh fakeunraid",
            "namespace": "claude-shared",
            "memory_type": "command",
            "tags": ["plan1-contradiction-test", "command"],
        },
        timeout=30,
    )
    assert cmd.status_code == 200
    # Review Queue verification happens in T2.7 once /admin/review-queue exists
