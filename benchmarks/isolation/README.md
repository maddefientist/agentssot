# Namespace Isolation Benchmark

Ported from ArcRift's `mcp-stress-test.ts`. AgentSSOT enforces multi-tenant
isolation through namespace-scoped API keys (RBAC); this benchmark **proves**
the property instead of assuming it.

## What it does

1. Provisions `N` namespaces, each with its own `writer` key scoped to that
   namespace only.
2. Writes a unique secret token into each namespace.
3. Issues the full `N×(N-1)` cross-tenant access matrix: every key tries to read
   every *other* namespace.
4. Asserts: every cross-tenant call is denied (401/403) and no foreign secret
   ever appears in a response body → **100% isolation integrity**.

## Run

```bash
SSOT_TEST_URL=http://localhost:8088 \
SSOT_TEST_ADMIN_KEY=ssot_admin_xxx \
python benchmarks/isolation/runner.py --tenants 10
```

Report is written to `results/isolation_report.md`.

## CI assertion

The same property is asserted as a pytest integration test:

```bash
SSOT_TEST_URL=... SSOT_TEST_ADMIN_KEY=... \
pytest -m integration api/tests/test_namespace_isolation.py -v
```

## Target

- **Project Isolation:** 100% (absolute zero cross-namespace leakage).
