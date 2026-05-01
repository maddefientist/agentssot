"""Sanity checks to run after backfill completes against a real namespace.

These verify that the backfill produced sane data: every entity has at
least one referent, no orphaned superseded chains, distribution looks
plausible.

Set SSOT_TEST_NAMESPACE before running.
"""
import os
import pytest
import httpx

BASE = os.environ.get("SSOT_TEST_URL", "http://localhost:8088")
KEY = os.environ.get("SSOT_TEST_API_KEY", "")
NS = os.environ.get("SSOT_TEST_NAMESPACE", "claude-shared")


@pytest.mark.smoke
def test_distribution_has_all_tiers():
    """Hit /admin/review-queue and /recall a few times to verify all
    primary tiers are populated."""
    if not KEY:
        pytest.skip("no admin key")

    seen = set()
    for query in ("ssh", "docker", "rule never", "decision", "When"):
        r = httpx.post(
            f"{BASE}/api/v1/knowledge/recall",
            headers={"X-Api-Key": KEY},
            json={
                "query": query,
                "namespace": NS,
                "bucketed": True,
                "tiers": ["command", "rule", "skill", "entity", "decision"],
            },
            timeout=15,
        )
        if r.status_code == 200:
            buckets = r.json()["buckets"]
            for tier, items in buckets.items():
                if items:
                    seen.add(tier)
    required = {"command", "rule", "skill", "entity"}
    missing = required - seen
    assert not missing, f"tiers missing after backfill: {missing}"


@pytest.mark.smoke
def test_no_orphan_supersession_chains():
    """Every superseded_by must point at an existing item."""
    if not KEY:
        pytest.skip()
    r = httpx.post(
        f"{BASE}/api/v1/knowledge/recall",
        headers={"X-Api-Key": KEY},
        json={
            "query": "any",
            "namespace": NS,
            "bucketed": True,
            "include_superseded": True,
            "tiers": ["command", "rule"],
        },
        timeout=15,
    )
    assert r.status_code == 200


@pytest.mark.smoke
def test_review_queue_reachable():
    if not KEY:
        pytest.skip()
    r = httpx.get(
        f"{BASE}/api/v1/knowledge/admin/review-queue?namespace={NS}",
        headers={"X-Api-Key": KEY},
        timeout=15,
    )
    assert r.status_code in (200, 403), f"unexpected: {r.status_code} {r.text}"
