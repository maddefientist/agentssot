"""Tests for Neural Cortex features: weighted recall, knowledge feedback, concept graph."""

import httpx
import pytest

from .conftest import BASE_URL

API_KEY = None  # Will be set by fixture


@pytest.fixture(scope="module")
def api_key():
    """Get a working API key by enrolling a test device."""
    import uuid
    name = f"cortex-test-{uuid.uuid4().hex[:8]}"
    resp = httpx.post(f"{BASE_URL}/enroll/auto", json={"name": name, "passphrase": ""})
    assert resp.status_code == 200
    return resp.json()["api_key"]


@pytest.fixture(scope="module")
def headers(api_key):
    return {"X-API-Key": api_key, "Content-Type": "application/json"}


class TestWeightedRecall:
    """Test that recall works with strength-weighted scoring."""

    def test_knowledge_recall_returns_results(self, headers):
        resp = httpx.post(
            f"{BASE_URL}/recall",
            headers=headers,
            json={
                "query_text": "docker deployment",
                "namespace": "claude-shared",
                "scope": "knowledge",
                "top_k": 3,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert data["scope"] == "knowledge"

    def test_concept_recall_returns_results(self, headers):
        resp = httpx.post(
            f"{BASE_URL}/recall",
            headers=headers,
            json={
                "query_text": "security patterns",
                "namespace": "claude-shared",
                "scope": "concepts",
                "top_k": 3,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert data["scope"] == "concepts"

    def test_all_scope_returns_mixed_results(self, headers):
        resp = httpx.post(
            f"{BASE_URL}/recall",
            headers=headers,
            json={
                "query_text": "python programming",
                "namespace": "claude-shared",
                "scope": "all",
                "top_k": 5,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["scope"] == "all"
        # Should have items from both knowledge and concepts
        assert "items" in data


class TestKnowledgeFeedback:
    """Test knowledge item feedback endpoint."""

    def _get_knowledge_item_id(self, headers):
        """Get a real knowledge item ID via recall."""
        resp = httpx.post(
            f"{BASE_URL}/recall",
            headers=headers,
            json={
                "query_text": "test",
                "namespace": "claude-shared",
                "scope": "knowledge",
                "top_k": 1,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) > 0
        return items[0]["id"]

    def test_useful_feedback_increases_strength(self, headers):
        kid = self._get_knowledge_item_id(headers)
        resp = httpx.post(
            f"{BASE_URL}/feedback",
            headers=headers,
            json={
                "signal": "useful",
                "knowledge_item_id": kid,
                "namespace": "claude-shared",
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["knowledge_item_id"] == kid
        assert data["signal"] == "useful"
        assert data["strength"] > 0

    def test_noted_feedback_works(self, headers):
        kid = self._get_knowledge_item_id(headers)
        resp = httpx.post(
            f"{BASE_URL}/feedback",
            headers=headers,
            json={
                "signal": "noted",
                "knowledge_item_id": kid,
                "namespace": "claude-shared",
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["signal"] == "noted"

    def test_concept_feedback_still_works(self, headers):
        """Ensure existing concept feedback path isn't broken."""
        # Get a concept ID
        resp = httpx.post(
            f"{BASE_URL}/recall",
            headers=headers,
            json={
                "query_text": "architecture",
                "namespace": "claude-shared",
                "scope": "concepts",
                "top_k": 1,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        items = resp.json()["items"]
        if not items:
            pytest.skip("No concepts available")

        concept_id = items[0]["id"]
        resp = httpx.post(
            f"{BASE_URL}/feedback",
            headers=headers,
            json={
                "signal": "useful",
                "concept_id": concept_id,
                "namespace": "claude-shared",
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["concept_id"] == concept_id


class TestCortexLinks:
    """Test the /cortex/links endpoint for neural network edges."""

    def test_cortex_links_returns_list(self):
        resp = httpx.get(f"{BASE_URL}/cortex/links", params={"namespace": "claude-shared"})
        assert resp.status_code == 200
        data = resp.json()
        assert "links" in data
        assert isinstance(data["links"], list)

    def test_cortex_links_structure(self):
        resp = httpx.get(f"{BASE_URL}/cortex/links", params={"namespace": "claude-shared"})
        assert resp.status_code == 200
        data = resp.json()
        if data["links"]:
            link = data["links"][0]
            assert "source" in link
            assert "target" in link
            assert "weight" in link
            assert "link_type" in link
            assert "co_occurrences" in link

    def test_cortex_links_respects_limit(self):
        resp = httpx.get(f"{BASE_URL}/cortex/links", params={"namespace": "claude-shared", "limit": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["links"]) <= 5


class TestHealthEndpoint:
    """Verify the API is healthy with all cortex features."""

    def test_health_ok(self):
        resp = httpx.get(f"{BASE_URL}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["embedding_available"] is True
        assert data["synthesis_enabled"] is True
