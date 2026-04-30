"""bucketed=true returns tier-grouped buckets + diagnostics."""
import os
import pytest
import httpx

BASE = os.environ.get("SSOT_TEST_URL", "http://localhost:8088")
KEY = os.environ.get("SSOT_TEST_API_KEY", "")


@pytest.mark.integration
def test_bucketed_recall_shape():
    if not KEY:
        pytest.skip("SSOT_TEST_API_KEY not set")
    r = httpx.post(
        f"{BASE}/api/v1/knowledge/recall",
        headers={"X-Api-Key": KEY},
        json={
            "query": "ssh unraid",
            "namespace": "claude-shared",
            "bucketed": True,
            "tiers": ["command", "rule", "skill", "entity"],
            "top_per_tier": {"command": 3, "rule": 2, "skill": 5, "entity": 3},
        },
        timeout=15,
    )
    assert r.status_code == 200
    body = r.json()
    assert "buckets" in body
    assert set(body["buckets"].keys()) >= {"command", "rule", "skill", "entity"}
    assert "diagnostics" in body
    diag = body["diagnostics"]
    assert "vec_ms" in diag and "rerank_ms" in diag
    assert diag["reranker_used"] in {"qwen3-reranker-4b", "qwen3-reranker-8b", "none"}


@pytest.mark.integration
def test_bucketed_excludes_episodic_by_default():
    if not KEY:
        pytest.skip("SSOT_TEST_API_KEY not set")
    r = httpx.post(
        f"{BASE}/api/v1/knowledge/recall",
        headers={"X-Api-Key": KEY},
        json={"query": "session log", "namespace": "claude-shared", "bucketed": True},
        timeout=15,
    )
    body = r.json()
    assert "episodic" not in body["buckets"]
