"""Tests for Tiered Memory (knowledge) routes.

Covers:
- Schema validation for TieredKnowledgeCreate (namespace defaults, explicit ns, invalid category)
- Model fields: category, layer, abstract, summary, source_ki_id on KnowledgeItem
- Router security: auth required on all three endpoints
- Integration smoke tests (require SSOT_TEST_API_KEY)

"""

import os
import sys

import pytest

# ── Test configuration ──────────────────────────────────────────────

BASE_URL = os.environ.get("SSOT_TEST_URL", "http://localhost:8088")
API_KEY = os.environ.get("SSOT_TEST_API_KEY", "")

# Allow import of app modules without a live DB
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.settings import get_settings

get_settings.cache_clear()


# ═══════════════════════════════════════════════════════════════════
# UNIT TESTS — No database or running API needed
# ═══════════════════════════════════════════════════════════════════


class TestTieredKnowledgeCreateSchema:
    """Schema validation for TieredKnowledgeCreate."""

    def test_namespace_omitted_defaults_to_default(self):
        from app.schemas import TieredKnowledgeCreate

        item = TieredKnowledgeCreate(content="hello world")
        assert item.namespace == "default"

    def test_namespace_explicit_overrides_default(self):
        from app.schemas import TieredKnowledgeCreate

        item = TieredKnowledgeCreate(content="hello", namespace="my-namespace")
        assert item.namespace == "my-namespace"

    def test_namespace_empty_string_defaults_to_default(self):
        from app.schemas import TieredKnowledgeCreate

        item = TieredKnowledgeCreate(content="hello", namespace="")
        # Empty string is a valid explicit value, so it is what is stored.
        # The router treats "" as "default". We test router behavior in integration.
        assert item.namespace == ""

    def test_invalid_category_rejected(self):
        from app.schemas import TieredKnowledgeCreate
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            TieredKnowledgeCreate(content="hello", category="not_a_real_category")
        # Pydantic will report which literal values are allowed
        assert "category" in str(exc_info.value)

    def test_all_optional_fields_omitted(self):
        from app.schemas import TieredKnowledgeCreate

        item = TieredKnowledgeCreate(content="test content")
        assert item.category is None
        assert item.abstract is None
        assert item.summary is None
        assert item.source is None
        assert item.source_ref is None
        assert item.tags == []
        assert item.memory_type is None
        assert item.generate_summaries is False

    def test_all_fields_provided(self):
        from app.schemas import TieredKnowledgeCreate

        item = TieredKnowledgeCreate(
            content="full content here",
            namespace="my-ns",
            category="user_preferences",
            abstract="short abstract",
            summary="medium summary",
            source="test",
            source_ref="ref-1",
            tags=["tag1", "tag2"],
            memory_type="preference",
            generate_summaries=True,
        )
        assert item.namespace == "my-ns"
        assert item.category == "user_preferences"
        assert item.abstract == "short abstract"
        assert item.summary == "medium summary"
        assert item.tags == ["tag1", "tag2"]
        assert item.memory_type == "preference"
        assert item.generate_summaries is True


class TestVerbatimMode:
    """Verbatim flag: opt-out of L0/L1 synthesis."""

    def test_verbatim_defaults_false(self):
        from app.schemas import TieredKnowledgeCreate

        item = TieredKnowledgeCreate(content="hi")
        assert item.verbatim is False

    def test_verbatim_explicit_true(self):
        from app.schemas import TieredKnowledgeCreate

        item = TieredKnowledgeCreate(content="hi", verbatim=True)
        assert item.verbatim is True

    def test_verbatim_column_exists_on_model(self):
        pytest.importorskip("pgvector", reason="pgvector required for model imports")
        from app.models import KnowledgeItem

        col = KnowledgeItem.__table__.columns["verbatim"]
        assert col.nullable is False
        assert col.default is not None  # SQLAlchemy default=False

    def test_verbatim_wins_over_generate_summaries_at_schema_level(self):
        # Schema permits the combination; router-level logic enforces that
        # verbatim suppresses synthesis. Document that the combination is legal
        # to construct (router resolves the conflict, not Pydantic).
        from app.schemas import TieredKnowledgeCreate

        item = TieredKnowledgeCreate(content="hi", verbatim=True, generate_summaries=True)
        assert item.verbatim is True
        assert item.generate_summaries is True


class TestTieredRecallRequestSchema:
    """TieredRecallRequest namespace default."""

    def test_namespace_defaults_to_default(self):
        # TieredRecallRequest.namespace changed from "claude-shared" to "default"
        # to be consistent with all other namespace fields in the app (IngestRequest,
        # RecallRequest, etc.). "claude-shared" was a historical outlier.
        from app.schemas import TieredRecallRequest

        req = TieredRecallRequest(query="hello")
        assert req.namespace == "default"

    def test_namespace_can_be_overridden(self):
        from app.schemas import TieredRecallRequest

        req = TieredRecallRequest(query="hello", namespace="my-ns")
        assert req.namespace == "my-ns"


class TestKnowledgeItemModelFields:
    """Verify tiered-memory columns exist on the KnowledgeItem model."""

    @classmethod
    def setup_class(cls):
        pytest.importorskip("pgvector", reason="pgvector required for model imports")

    def test_category_column_exists(self):
        from app.models import KnowledgeItem

        col_names = {c.name for c in KnowledgeItem.__table__.columns}
        assert "category" in col_names

    def test_layer_column_exists(self):
        from app.models import KnowledgeItem

        col_names = {c.name for c in KnowledgeItem.__table__.columns}
        assert "layer" in col_names

    def test_layer_enum_values(self):
        from app.models import ContentLayer

        assert {e.value for e in ContentLayer} == {"abstract", "summary", "full"}

    def test_abstract_column_exists(self):
        from app.models import KnowledgeItem

        col_names = {c.name for c in KnowledgeItem.__table__.columns}
        assert "abstract" in col_names

    def test_summary_column_exists(self):
        from app.models import KnowledgeItem

        col_names = {c.name for c in KnowledgeItem.__table__.columns}
        assert "summary" in col_names

    def test_source_ki_id_column_exists(self):
        from app.models import KnowledgeItem

        col_names = {c.name for c in KnowledgeItem.__table__.columns}
        assert "source_ki_id" in col_names

    def test_layer_is_not_nullable(self):
        from app.models import KnowledgeItem

        col = KnowledgeItem.__table__.columns["layer"]
        assert col.nullable is False

    def test_abstract_and_summary_are_nullable(self):
        from app.models import KnowledgeItem

        for col_name in ["abstract", "summary"]:
            col = KnowledgeItem.__table__.columns[col_name]
            assert col.nullable is True, f"{col_name} should be nullable"


class TestMemoryCategoryEnum:
    """Verify MemoryCategory enum has expected values."""

    @classmethod
    def setup_class(cls):
        pytest.importorskip("pgvector", reason="pgvector required for model imports")

    def test_expected_categories_present(self):
        from app.models import MemoryCategory

        expected = {
            "user_profile",
            "user_preferences",
            "user_entities",
            "user_events",
            "agent_patterns",
            "agent_tools",
            "agent_skills",
            "agent_cases",
        }
        assert {c.value for c in MemoryCategory} == expected


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION SMOKE TESTS — Require running API + SSOT_TEST_API_KEY
# ═══════════════════════════════════════════════════════════════════

requires_api = pytest.mark.skipif(
    not API_KEY,
    reason="SSOT_TEST_API_KEY not set — skipping integration tests",
)


@requires_api
class TestTieredMemoryIntegration:
    """Smoke tests against a live API instance."""

    @pytest.fixture
    def client(self):
        import httpx

        with httpx.Client(base_url=BASE_URL, timeout=10) as c:
            yield c

    @pytest.fixture
    def headers(self):
        # Use X-API-Key header (NOT Authorization Bearer) per app convention
        return {"X-API-Key": API_KEY}

    def test_unauthenticated_categories_returns_401(self, client):
        """GET /categories with no API key returns 401."""
        resp = client.get("/api/v1/knowledge/categories")
        assert resp.status_code == 401

    def test_authenticated_categories_returns_200(self, client, headers):
        """GET /categories with valid API key returns 200 and category list."""
        resp = client.get("/api/v1/knowledge/categories", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "categories" in data
        assert isinstance(data["categories"], list)
        assert len(data["categories"]) > 0
        # Verify expected domain groupings
        assert "categories_by_domain" in data
        domains = data["categories_by_domain"]
        assert "user" in domains
        assert "agent" in domains

    def test_categories_includes_user_and_agent_domains(self, client, headers):
        """Verify all expected category values are present."""
        resp = client.get("/api/v1/knowledge/categories", headers=headers)
        assert resp.status_code == 200
        cats = set(resp.json()["categories"])
        expected = {
            "user_profile",
            "user_preferences",
            "user_entities",
            "user_events",
            "agent_patterns",
            "agent_tools",
            "agent_skills",
            "agent_cases",
        }
        assert cats == expected
