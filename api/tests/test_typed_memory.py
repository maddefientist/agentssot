"""Tests for Milestone 3: Typed Memory Schema Groundwork.

These tests cover:
1. Backward compatibility — items without memory_type still work
2. Recall with memory_type filter
3. Recall with staleness filter
4. Backfill heuristic logic (pure function, no DB needed)
5. Ingest with memory_type
6. Schema validation

Tests are split into:
- Unit tests (no DB): backfill heuristics, schema validation
- Integration tests (running API): ingest/recall with typed memory
"""

import os
import sys
import pytest

# ── Test configuration ──────────────────────────────────────────────

BASE_URL = os.environ.get("SSOT_TEST_URL", "http://localhost:8088")
API_KEY = os.environ.get("SSOT_TEST_API_KEY", "")

# For unit tests, ensure we can import the app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set dummy DATABASE_URL so Settings can be imported without a real DB
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"

# Clear the lru_cache on get_settings so our env var takes effect
from app.settings import get_settings
get_settings.cache_clear()


# ═══════════════════════════════════════════════════════════════════
# UNIT TESTS — No database or running API needed
# ═══════════════════════════════════════════════════════════════════


class TestMemoryTypeEnum:
    """Verify MemoryType enum has correct values and is importable."""

    def test_enum_values(self):
        from app.models import MemoryType

        expected = {"fact", "decision", "preference", "skill", "reference", "correction", "session_summary"}
        actual = {m.value for m in MemoryType}
        assert actual == expected

    def test_enum_is_string(self):
        from app.models import MemoryType

        assert isinstance(MemoryType.fact, str)
        assert MemoryType.fact == "fact"
        assert MemoryType.session_summary == "session_summary"


class TestSchemaBackwardCompat:
    """Verify schemas accept new fields as optional."""

    def test_knowledge_item_in_without_memory_type(self):
        from app.schemas import KnowledgeItemIn

        item = KnowledgeItemIn(content="test content")
        assert item.memory_type is None
        assert item.extraction_source is None
        assert item.extraction_cursor_id is None

    def test_knowledge_item_in_with_memory_type(self):
        from app.schemas import KnowledgeItemIn

        item = KnowledgeItemIn(content="test", memory_type="decision")
        assert item.memory_type == "decision"

    def test_knowledge_item_in_rejects_invalid_type(self):
        from app.schemas import KnowledgeItemIn
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            KnowledgeItemIn(content="test", memory_type="invalid_type")

    def test_recall_request_without_filters(self):
        from app.schemas import RecallRequest

        req = RecallRequest(query_text="hello")
        assert req.memory_type is None
        assert req.max_staleness is None

    def test_recall_request_with_filters(self):
        from app.schemas import RecallRequest

        req = RecallRequest(
            query_text="hello",
            memory_type="skill",
            max_staleness=0.5,
        )
        assert req.memory_type == "skill"
        assert req.max_staleness == 0.5

    def test_recall_item_with_typed_fields(self):
        from app.schemas import RecallItem
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        item = RecallItem(
            id="test-id",
            scope="knowledge",
            score=0.5,
            snippet="test",
            memory_type="fact",
            last_verified_at=now,
            staleness_score=0.3,
            extraction_source="session-abc",
        )
        assert item.memory_type == "fact"
        assert item.staleness_score == 0.3

    def test_recall_item_without_typed_fields(self):
        """Existing recall items without typed fields still valid."""
        from app.schemas import RecallItem

        item = RecallItem(
            id="test-id",
            scope="knowledge",
            score=0.5,
            snippet="test",
        )
        assert item.memory_type is None
        assert item.last_verified_at is None
        assert item.staleness_score is None
        assert item.extraction_source is None


class TestBackfillHeuristics:
    """Test the classification logic from the backfill script (pure functions)."""

    def _make_item(self, tags=None, source=None, content=None):
        """Create a mock item matching the interface expected by classify_item."""
        class MockItem:
            pass
        item = MockItem()
        item.tags = tags or []
        item.source = source
        item.content = content
        return item

    def _classify(self, item):
        # Import from the script
        script_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
        sys.path.insert(0, script_dir)
        from backfill_memory_types import classify_item
        return classify_item(item)

    def test_correction_by_tag(self):
        item = self._make_item(tags=["correction", "operator-feedback"])
        mt, rule = self._classify(item)
        assert mt == "correction"
        assert rule == "correction_by_tag"

    def test_session_summary_by_tag(self):
        item = self._make_item(tags=["summary", "compaction"])
        mt, rule = self._classify(item)
        assert mt == "session_summary"
        assert rule == "session_summary_by_tag"

    def test_session_summary_by_source(self):
        item = self._make_item(source="session_compaction")
        mt, rule = self._classify(item)
        assert mt == "session_summary"
        assert rule == "session_summary_by_tag"

    def test_skill_by_tag(self):
        item = self._make_item(tags=["skill", "operator-taught"])
        mt, rule = self._classify(item)
        assert mt == "skill"
        assert rule == "skill_by_tag"

    def test_preference_by_tag(self):
        item = self._make_item(tags=["preference"])
        mt, rule = self._classify(item)
        assert mt == "preference"
        assert rule == "preference_by_tag"

    def test_decision_by_tag(self):
        item = self._make_item(tags=["decision"])
        mt, rule = self._classify(item)
        assert mt == "decision"
        assert rule == "decision_by_tag"

    def test_reference_by_tag(self):
        item = self._make_item(tags=["reference"])
        mt, rule = self._classify(item)
        assert mt == "reference"
        assert rule == "reference_by_tag"

    def test_fact_by_extraction_tag(self):
        item = self._make_item(tags=["session-extract"])
        mt, rule = self._classify(item)
        assert mt == "fact"
        assert rule == "fact_by_extraction"

    def test_decision_by_content(self):
        item = self._make_item(content="We decided to use PostgreSQL for the database layer")
        mt, rule = self._classify(item)
        assert mt == "decision"
        assert rule == "decision_by_content"

    def test_preference_by_content(self):
        item = self._make_item(content="User prefers dark mode and vim keybindings")
        mt, rule = self._classify(item)
        assert mt == "preference"
        assert rule == "preference_by_content"

    def test_unclassified(self):
        item = self._make_item(tags=[], content="Just some random knowledge")
        mt, rule = self._classify(item)
        assert mt is None
        assert rule == "no_match"

    def test_tag_priority_correction_over_content(self):
        """Tags should take priority over content heuristics."""
        item = self._make_item(
            tags=["correction"],
            content="We decided to fix the bug",
        )
        mt, rule = self._classify(item)
        assert mt == "correction"

    def test_case_insensitive_tags(self):
        item = self._make_item(tags=["SKILL", "Operator-Taught"])
        mt, rule = self._classify(item)
        assert mt == "skill"


class TestModelColumns:
    """Verify new columns exist on the KnowledgeItem model."""

    def test_memory_type_column_exists(self):
        from app.models import KnowledgeItem
        mapper = KnowledgeItem.__table__
        col_names = {c.name for c in mapper.columns}
        assert "memory_type" in col_names

    def test_verification_columns_exist(self):
        from app.models import KnowledgeItem
        mapper = KnowledgeItem.__table__
        col_names = {c.name for c in mapper.columns}
        assert "last_verified_at" in col_names
        assert "staleness_score" in col_names
        assert "extraction_source" in col_names
        assert "extraction_cursor_id" in col_names

    def test_memory_type_is_nullable(self):
        from app.models import KnowledgeItem
        col = KnowledgeItem.__table__.columns["memory_type"]
        assert col.nullable is True

    def test_verification_columns_are_nullable(self):
        from app.models import KnowledgeItem
        for col_name in ["last_verified_at", "staleness_score", "extraction_source", "extraction_cursor_id"]:
            col = KnowledgeItem.__table__.columns[col_name]
            assert col.nullable is True, f"{col_name} should be nullable"


class TestSettingsFeatureFlag:
    """Verify the typed_memory_enabled feature flag."""

    def test_default_is_disabled(self):
        from app.settings import Settings
        # Create with minimal required fields
        s = Settings(DATABASE_URL="postgresql://test:test@localhost/test")
        assert s.typed_memory_enabled is False

    def test_can_enable(self):
        from app.settings import Settings
        s = Settings(DATABASE_URL="postgresql://test:test@localhost/test", TYPED_MEMORY_ENABLED="true")
        assert s.typed_memory_enabled is True


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — Require running API with SSOT_TEST_URL
# ═══════════════════════════════════════════════════════════════════

# Skip integration tests if no API key is configured
requires_api = pytest.mark.skipif(
    not API_KEY,
    reason="SSOT_TEST_API_KEY not set — skipping integration tests",
)


@requires_api
class TestTypedMemoryIntegration:
    """Integration tests against a running API instance."""

    @pytest.fixture
    def client(self):
        import httpx
        with httpx.Client(base_url=BASE_URL, timeout=10) as c:
            yield c

    @pytest.fixture
    def headers(self):
        return {"Authorization": f"Bearer {API_KEY}"}

    def test_ingest_without_memory_type(self, client, headers):
        """Existing ingest payloads without memory_type still work."""
        resp = client.post(
            "/ingest",
            json={
                "namespace": "claude-shared",
                "knowledge_items": [
                    {"content": "Test backward compat item", "tags": ["test-typed-memory"]},
                ],
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["counts"]["knowledge_items"] >= 1

    def test_ingest_with_memory_type(self, client, headers):
        """Ingest with memory_type field accepted."""
        resp = client.post(
            "/ingest",
            json={
                "namespace": "claude-shared",
                "knowledge_items": [
                    {
                        "content": "User prefers dark terminals",
                        "tags": ["test-typed-memory", "preference"],
                        "memory_type": "preference",
                        "extraction_source": "test-session-123",
                    },
                ],
            },
            headers=headers,
        )
        assert resp.status_code == 200

    def test_recall_without_filters(self, client, headers):
        """Recall without memory_type filter returns results as before."""
        resp = client.post(
            "/recall",
            json={
                "namespace": "claude-shared",
                "query_text": "test backward compat",
                "scope": "knowledge",
                "top_k": 5,
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data

    def test_recall_with_memory_type_filter(self, client, headers):
        """Recall with memory_type filter accepted (may return empty if flag off)."""
        resp = client.post(
            "/recall",
            json={
                "namespace": "claude-shared",
                "query_text": "dark terminals",
                "scope": "knowledge",
                "top_k": 5,
                "memory_type": "preference",
            },
            headers=headers,
        )
        assert resp.status_code == 200

    def test_recall_with_staleness_filter(self, client, headers):
        """Recall with max_staleness filter accepted."""
        resp = client.post(
            "/recall",
            json={
                "namespace": "claude-shared",
                "query_text": "test",
                "scope": "knowledge",
                "top_k": 5,
                "max_staleness": 0.8,
            },
            headers=headers,
        )
        assert resp.status_code == 200
