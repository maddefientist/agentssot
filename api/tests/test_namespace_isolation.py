"""Multi-tenant namespace isolation benchmark (lift #4 from ArcRift review).

Ported from ArcRift's `mcp-stress-test.ts`: write a unique secret token into
N namespaces, each reachable only by its own scoped key, then prove that no
key can read any other namespace's secret. AgentSSOT *claims* namespace RBAC
isolation but had no test asserting it — this certifies the property.

Requires a live API and an ADMIN key:
    SSOT_TEST_URL        (default http://localhost:8088)
    SSOT_TEST_ADMIN_KEY  (admin API key; test is skipped if unset)

Run:  pytest -m integration api/tests/test_namespace_isolation.py -v
"""

import os
import uuid

import httpx
import pytest

BASE = os.environ.get("SSOT_TEST_URL", "http://localhost:8088")
ADMIN_KEY = os.environ.get("SSOT_TEST_ADMIN_KEY", "")
N_TENANTS = int(os.environ.get("SSOT_ISOLATION_TENANTS", "5"))


def _admin(client: httpx.Client) -> dict:
    return {"X-Api-Key": ADMIN_KEY}


@pytest.fixture
def tenants():
    """Create N isolated namespaces, each with its own writer key and a unique secret."""
    if not ADMIN_KEY:
        pytest.skip("SSOT_TEST_ADMIN_KEY not set")

    run = uuid.uuid4().hex[:8]
    built: list[dict] = []
    with httpx.Client(base_url=BASE, timeout=20) as c:
        for i in range(N_TENANTS):
            ns = f"isolation-{run}-{i}"
            secret = f"ISOLATIONSECRET-{run}-{i}-{uuid.uuid4().hex}"

            r = c.post("/admin/namespaces", headers=_admin(c), json={"name": ns})
            assert r.status_code in (200, 201, 409), r.text

            r = c.post(
                "/admin/api-keys",
                headers=_admin(c),
                json={"name": f"key-{ns}", "role": "writer", "namespaces": [ns]},
            )
            assert r.status_code in (200, 201), r.text
            key = r.json()["api_key"]

            # Write the tenant's secret into its own namespace.
            r = c.post(
                "/ingest",
                headers={"X-Api-Key": key},
                json={
                    "namespace": ns,
                    "knowledge_items": [
                        {"content": f"The deploy token for this tenant is {secret}.",
                         "tags": ["isolation-test"]}
                    ],
                },
            )
            assert r.status_code == 200, r.text

            built.append({"ns": ns, "secret": secret, "key": key})
    return built


@pytest.mark.integration
def test_each_key_reads_only_its_own_namespace(tenants):
    """Cross-tenant access must be denied, and no foreign secret may ever surface."""
    leaks: list[str] = []
    denied = 0
    own_ok = 0

    with httpx.Client(base_url=BASE, timeout=20) as c:
        for t in tenants:
            headers = {"X-Api-Key": t["key"]}

            # 1. Own namespace: the key can read, and finds its own secret.
            r = c.get("/query", headers=headers, params={"q": "deploy token", "namespace": t["ns"]})
            assert r.status_code == 200, r.text
            body_own = r.text
            if t["secret"] in body_own:
                own_ok += 1

            # 2. Every other namespace: access must be refused (403/401), and the
            #    foreign secret must never appear in any response body.
            for other in tenants:
                if other["ns"] == t["ns"]:
                    continue
                r = c.get("/query", headers=headers,
                          params={"q": "deploy token", "namespace": other["ns"]})
                if r.status_code in (401, 403):
                    denied += 1
                elif r.status_code == 200:
                    # If RBAC ever allowed the call, it must still leak nothing.
                    if other["secret"] in r.text:
                        leaks.append(f"{t['ns']} read {other['ns']}'s secret via /query")
                else:
                    leaks.append(f"unexpected status {r.status_code} for {t['ns']}→{other['ns']}")

    assert own_ok == len(tenants), f"only {own_ok}/{len(tenants)} keys could read their own namespace"
    assert not leaks, "ISOLATION BREACH:\n" + "\n".join(leaks)
    # Expect full cross-product denial: N*(N-1)
    assert denied == len(tenants) * (len(tenants) - 1)


@pytest.mark.integration
def test_recall_does_not_cross_namespaces(tenants):
    """Semantic recall (when embeddings are available) must also respect scope.

    Skips gracefully if the embedding provider is disabled on the server.
    """
    breaches: list[str] = []
    with httpx.Client(base_url=BASE, timeout=30) as c:
        for t in tenants:
            headers = {"X-Api-Key": t["key"]}
            for other in tenants:
                if other["ns"] == t["ns"]:
                    continue
                r = c.post("/recall", headers=headers,
                           json={"namespace": other["ns"], "query_text": "deploy token", "top_k": 5})
                if r.status_code in (401, 403):
                    continue  # correct — denied
                if r.status_code == 400:
                    pytest.skip("embedding provider unavailable on server")
                if r.status_code == 200 and other["secret"] in r.text:
                    breaches.append(f"{t['ns']} recalled {other['ns']}'s secret")
    assert not breaches, "ISOLATION BREACH:\n" + "\n".join(breaches)
