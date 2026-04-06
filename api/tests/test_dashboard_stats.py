"""Tests for Milestone 12: Enhanced Dashboard Stats.

Covers:
1. Stats endpoint returns expected shape (all new fields)
2. Memory type distribution present
3. Staleness distribution buckets
4. Secret scanning status
5. Sync status field
6. Backward compat: original fields still present
"""

import os
import sys

import pytest

BASE_URL = os.environ.get("SSOT_TEST_URL", "http://localhost:8088")
API_KEY = os.environ.get("SSOT_TEST_API_KEY", "")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"

from app.settings import get_settings
get_settings.cache_clear()


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — Require running API
# ═══════════════════════════════════════════════════════════════════

requires_api = pytest.mark.skipif(
    not API_KEY,
    reason="SSOT_TEST_API_KEY not set — skipping integration tests",
)


@requires_api
class TestDashboardStatsIntegration:
    """Integration tests for /dashboard/stats endpoint."""

    @pytest.fixture
    def client(self):
        import httpx
        with httpx.Client(base_url=BASE_URL, timeout=10) as c:
            yield c

    def test_stats_endpoint_returns_200(self, client):
        """Dashboard stats endpoint is public and returns 200."""
        resp = client.get("/dashboard/stats", params={"namespace": "claude-shared"})
        assert resp.status_code == 200

    def test_stats_has_original_fields(self, client):
        """Original fields are still present for backward compat."""
        resp = client.get("/dashboard/stats", params={"namespace": "claude-shared"})
        data = resp.json()
        for field in ["concepts", "skills", "knowledge", "recalls_24h", "ingested_24h", "avg_confidence"]:
            assert field in data, f"Missing original field: {field}"

    def test_stats_has_memory_type_distribution(self, client):
        """M3 data: memory_type_distribution is a dict of type→count."""
        resp = client.get("/dashboard/stats", params={"namespace": "claude-shared"})
        data = resp.json()
        assert "memory_type_distribution" in data
        dist = data["memory_type_distribution"]
        assert isinstance(dist, dict)
        # All values should be non-negative integers
        for k, v in dist.items():
            assert isinstance(v, int) and v >= 0, f"Bad value for {k}: {v}"

    def test_stats_has_staleness_distribution(self, client):
        """M3 data: staleness_distribution has expected buckets."""
        resp = client.get("/dashboard/stats", params={"namespace": "claude-shared"})
        data = resp.json()
        assert "staleness_distribution" in data
        staleness = data["staleness_distribution"]
        expected_buckets = {"fresh", "aging", "stale", "critical", "unscored"}
        assert set(staleness.keys()) == expected_buckets

    def test_stats_has_secret_scanning(self, client):
        """M8 data: secret_scanning status is present."""
        resp = client.get("/dashboard/stats", params={"namespace": "claude-shared"})
        data = resp.json()
        assert "secret_scanning" in data
        assert "enabled" in data["secret_scanning"]
        assert isinstance(data["secret_scanning"]["enabled"], bool)

    def test_stats_has_sync_status(self, client):
        """M10 data: sync status is present."""
        resp = client.get("/dashboard/stats", params={"namespace": "claude-shared"})
        data = resp.json()
        assert "sync" in data
        assert "enabled" in data["sync"]
        assert isinstance(data["sync"]["enabled"], bool)
        assert "devices" in data["sync"]
        assert isinstance(data["sync"]["devices"], list)

    def test_memory_type_distribution_counts_correctly(self, client):
        """Verify that summing memory_type_distribution ≈ total knowledge count."""
        resp = client.get("/dashboard/stats", params={"namespace": "claude-shared"})
        data = resp.json()
        total_knowledge = data["knowledge"]
        dist_sum = sum(data["memory_type_distribution"].values())
        assert dist_sum == total_knowledge, (
            f"Distribution sum ({dist_sum}) should equal total knowledge ({total_knowledge})"
        )

    def test_staleness_distribution_sums_to_total(self, client):
        """Verify staleness buckets sum to total knowledge count."""
        resp = client.get("/dashboard/stats", params={"namespace": "claude-shared"})
        data = resp.json()
        total_knowledge = data["knowledge"]
        staleness_sum = sum(data["staleness_distribution"].values())
        assert staleness_sum == total_knowledge, (
            f"Staleness sum ({staleness_sum}) should equal total knowledge ({total_knowledge})"
        )
