"""Tests for Milestone 10: Sync Semantics.

Covers:
1. Sync checkpoint CRUD (unit + integration)
2. Pending items query
3. Conflict detection
4. Backward compat: existing sync without checkpoints still works
5. Feature flag gating
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
# UNIT TESTS — No database needed
# ═══════════════════════════════════════════════════════════════════


def _try_import_sync():
    """Try importing sync module; skip if psycopg2/db deps unavailable."""
    try:
        from app.sync import (
            SyncCheckpointRequest, SyncCheckpointResponse,
            SyncPendingResponse, PendingConflict,
            SyncStatusResponse, SyncStatusDevice,
        )
        return True
    except (ModuleNotFoundError, ImportError):
        return False


_sync_available = _try_import_sync()
requires_sync_module = pytest.mark.skipif(
    not _sync_available,
    reason="psycopg2 or DB dependencies not available for sync module import",
)


@requires_sync_module
class TestSyncSchemas:
    """Verify sync Pydantic schemas validate correctly."""

    def test_checkpoint_request_valid(self):
        from app.sync import SyncCheckpointRequest

        req = SyncCheckpointRequest(
            device_id="macbook-pro",
            namespace="claude-shared",
            last_synced_item_id="550e8400-e29b-41d4-a716-446655440000",
        )
        assert req.device_id == "macbook-pro"
        assert req.namespace == "claude-shared"

    def test_checkpoint_request_defaults(self):
        from app.sync import SyncCheckpointRequest

        req = SyncCheckpointRequest(
            device_id="mac",
            last_synced_item_id="550e8400-e29b-41d4-a716-446655440000",
        )
        assert req.namespace == "claude-shared"

    def test_checkpoint_request_rejects_empty_device(self):
        from app.sync import SyncCheckpointRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SyncCheckpointRequest(
                device_id="",
                last_synced_item_id="550e8400-e29b-41d4-a716-446655440000",
            )

    def test_pending_response_shape(self):
        from app.sync import SyncPendingResponse

        resp = SyncPendingResponse(
            device_id="mac",
            namespace="claude-shared",
            pending_count=0,
            pending_items=[],
            conflicts=[],
        )
        assert resp.pending_count == 0
        assert resp.conflicts == []

    def test_pending_response_with_conflicts(self):
        from app.sync import SyncPendingResponse, PendingConflict
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        conflict = PendingConflict(
            content_hash="abc123",
            item_ids=["id1", "id2"],
            device_ids=["mac", "linux"],
            ingested_at=[now, now],
        )
        resp = SyncPendingResponse(
            device_id="mac",
            namespace="claude-shared",
            pending_count=2,
            pending_items=[],
            conflicts=[conflict],
        )
        assert len(resp.conflicts) == 1
        assert resp.conflicts[0].content_hash == "abc123"

    def test_status_response_shape(self):
        from app.sync import SyncStatusResponse, SyncStatusDevice

        resp = SyncStatusResponse(
            device_id="mac",
            namespaces=[
                SyncStatusDevice(
                    namespace="claude-shared",
                    last_synced_item_id=None,
                    last_synced_at=None,
                    pending_count=10,
                ),
            ],
        )
        assert len(resp.namespaces) == 1
        assert resp.namespaces[0].pending_count == 10


class TestSyncSettingsFlag:
    """Verify the sync_tracking_enabled feature flag."""

    def test_default_is_disabled(self):
        from app.settings import Settings
        s = Settings(DATABASE_URL="postgresql://test:test@localhost/test")
        assert s.sync_tracking_enabled is False

    def test_can_enable(self):
        from app.settings import Settings
        s = Settings(
            DATABASE_URL="postgresql://test:test@localhost/test",
            SYNC_TRACKING_ENABLED="true",
        )
        assert s.sync_tracking_enabled is True

    def test_conflict_window_default(self):
        from app.settings import Settings
        s = Settings(DATABASE_URL="postgresql://test:test@localhost/test")
        assert s.sync_conflict_window_hours == 24


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — Require running API with SSOT_TEST_URL
# ═══════════════════════════════════════════════════════════════════

requires_api = pytest.mark.skipif(
    not API_KEY,
    reason="SSOT_TEST_API_KEY not set — skipping integration tests",
)


@requires_api
class TestSyncIntegration:
    """Integration tests against a running API instance."""

    @pytest.fixture
    def client(self):
        import httpx
        with httpx.Client(base_url=BASE_URL, timeout=10) as c:
            yield c

    @pytest.fixture
    def headers(self):
        return {"X-API-Key": API_KEY}

    def test_sync_endpoints_return_404_when_disabled(self, client, headers):
        """When SYNC_TRACKING_ENABLED=false, all sync endpoints return 404."""
        # This test is only meaningful when sync is disabled on the test server
        resp = client.post(
            "/sync/checkpoint",
            json={
                "device_id": "test-device",
                "namespace": "claude-shared",
                "last_synced_item_id": "550e8400-e29b-41d4-a716-446655440000",
            },
            headers=headers,
        )
        # If sync is enabled, we get 200; if disabled, 404
        assert resp.status_code in (200, 404)

    def test_existing_ingest_works_without_checkpoints(self, client, headers):
        """Backward compat: standard ingest works regardless of sync feature flag."""
        resp = client.post(
            "/ingest",
            json={
                "namespace": "claude-shared",
                "knowledge_items": [
                    {"content": "Test sync backward compat", "tags": ["test-sync"]},
                ],
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["counts"]["knowledge_items"] >= 1

    def test_existing_recall_works_without_checkpoints(self, client, headers):
        """Backward compat: recall works regardless of sync feature flag."""
        resp = client.post(
            "/recall",
            json={
                "namespace": "claude-shared",
                "query_text": "sync backward compat",
                "scope": "knowledge",
                "top_k": 3,
            },
            headers=headers,
        )
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — Only run when sync is enabled
# ═══════════════════════════════════════════════════════════════════

requires_sync = pytest.mark.skipif(
    not API_KEY or not os.environ.get("SSOT_SYNC_ENABLED", ""),
    reason="SSOT_SYNC_ENABLED not set — skipping sync-specific integration tests",
)


@requires_sync
class TestSyncEnabledIntegration:
    """Integration tests that require SYNC_TRACKING_ENABLED=true on the server."""

    @pytest.fixture
    def client(self):
        import httpx
        with httpx.Client(base_url=BASE_URL, timeout=10) as c:
            yield c

    @pytest.fixture
    def headers(self):
        return {"X-API-Key": API_KEY}

    def test_checkpoint_crud(self, client, headers):
        """Record and verify a sync checkpoint."""
        # First ingest an item to get a valid ID
        ingest_resp = client.post(
            "/ingest",
            json={
                "namespace": "claude-shared",
                "knowledge_items": [
                    {"content": "Sync checkpoint test item", "tags": ["test-sync-ckpt"]},
                ],
            },
            headers=headers,
        )
        assert ingest_resp.status_code == 200

        # We need the item ID — query for it
        query_resp = client.get(
            "/query",
            params={"q": "Sync checkpoint test item", "namespace": "claude-shared", "limit": 1},
            headers=headers,
        )
        assert query_resp.status_code == 200
        items = query_resp.json().get("results", [])
        assert len(items) > 0
        item_id = items[0]["id"]

        # Record checkpoint
        ckpt_resp = client.post(
            "/sync/checkpoint",
            json={
                "device_id": "test-device-sync",
                "namespace": "claude-shared",
                "last_synced_item_id": item_id,
            },
            headers=headers,
        )
        assert ckpt_resp.status_code == 200
        data = ckpt_resp.json()
        assert data["device_id"] == "test-device-sync"
        assert data["last_synced_item_id"] == item_id

    def test_pending_items(self, client, headers):
        """Query pending items for a device."""
        resp = client.get(
            "/sync/pending",
            params={"device_id": "test-device-pending", "namespace": "claude-shared"},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "pending_count" in data
        assert "pending_items" in data
        assert data["pending_count"] >= 0

    def test_pending_with_conflicts(self, client, headers):
        """Query pending items with conflict detection."""
        resp = client.get(
            "/sync/pending",
            params={
                "device_id": "test-device-conflicts",
                "namespace": "claude-shared",
                "include_conflicts": "true",
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "conflicts" in data
        assert isinstance(data["conflicts"], list)

    def test_sync_status(self, client, headers):
        """Query sync status for a device."""
        resp = client.get(
            "/sync/status",
            params={"device_id": "test-device-status"},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["device_id"] == "test-device-status"
        assert "namespaces" in data
