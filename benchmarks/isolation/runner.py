#!/usr/bin/env python3
"""Namespace isolation benchmark — standalone report generator.

Ported from ArcRift's `mcp-stress-test.ts` (lift #4). Writes a unique secret
into N namespaces, each behind its own scoped writer key, then issues the full
N×(N-1) cross-tenant access matrix and certifies zero leakage. Emits a markdown
report to benchmarks/isolation/results/isolation_report.md.

Usage:
    SSOT_TEST_URL=http://localhost:8088 \
    SSOT_TEST_ADMIN_KEY=ssot_admin_... \
    python benchmarks/isolation/runner.py [--tenants 10]

This is the operational sibling of api/tests/test_namespace_isolation.py:
the pytest asserts the property in CI; this runner produces the human-readable
isolation certificate.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenants", type=int, default=int(os.environ.get("SSOT_ISOLATION_TENANTS", "10")))
    ap.add_argument("--url", default=os.environ.get("SSOT_TEST_URL", "http://localhost:8088"))
    ap.add_argument("--admin-key", default=os.environ.get("SSOT_TEST_ADMIN_KEY", ""))
    args = ap.parse_args()

    if not args.admin_key:
        print("ERROR: set SSOT_TEST_ADMIN_KEY (admin API key).", file=sys.stderr)
        return 2

    run = uuid.uuid4().hex[:8]
    admin = {"X-Api-Key": args.admin_key}
    tenants: list[dict] = []

    with httpx.Client(base_url=args.url, timeout=30) as c:
        print(f"Provisioning {args.tenants} isolated tenants (run {run})...")
        for i in range(args.tenants):
            ns = f"isolation-{run}-{i}"
            secret = f"ISOLATIONSECRET-{run}-{i}-{uuid.uuid4().hex}"
            r = c.post("/admin/namespaces", headers=admin, json={"name": ns})
            if r.status_code not in (200, 201, 409):
                print(f"  namespace {ns} failed: {r.status_code} {r.text}", file=sys.stderr)
                return 1
            r = c.post("/admin/api-keys", headers=admin,
                       json={"name": f"key-{ns}", "role": "writer", "namespaces": [ns]})
            r.raise_for_status()
            key = r.json()["api_key"]
            r = c.post("/ingest", headers={"X-Api-Key": key},
                       json={"namespace": ns,
                             "knowledge_items": [{"content": f"deploy token is {secret}",
                                                  "tags": ["isolation-test"]}]})
            r.raise_for_status()
            tenants.append({"ns": ns, "secret": secret, "key": key})

        print("Running cross-tenant access matrix...")
        total_probes = 0
        denied = 0
        leaks: list[str] = []
        own_ok = 0
        for t in tenants:
            headers = {"X-Api-Key": t["key"]}
            r = c.get("/query", headers=headers, params={"q": "deploy token", "namespace": t["ns"]})
            if r.status_code == 200 and t["secret"] in r.text:
                own_ok += 1
            for other in tenants:
                if other["ns"] == t["ns"]:
                    continue
                total_probes += 1
                r = c.get("/query", headers=headers,
                          params={"q": "deploy token", "namespace": other["ns"]})
                if r.status_code in (401, 403):
                    denied += 1
                elif r.status_code == 200 and other["secret"] in r.text:
                    leaks.append(f"{t['ns']} → {other['ns']} (via /query)")
                elif r.status_code == 200:
                    pass  # allowed but no secret leaked
                else:
                    leaks.append(f"{t['ns']} → {other['ns']} unexpected status {r.status_code}")

    isolation_pct = 100.0 * (total_probes - len(leaks)) / total_probes if total_probes else 100.0
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = RESULTS_DIR / "isolation_report.md"
    lines = [
        "# Namespace Isolation Report",
        "",
        f"- **Generated:** {ts}",
        f"- **Target:** `{args.url}`",
        f"- **Tenants:** {args.tenants}",
        f"- **Cross-tenant probes:** {total_probes} (N×(N-1))",
        f"- **Denied by RBAC (401/403):** {denied}",
        f"- **Own-namespace read OK:** {own_ok}/{args.tenants}",
        f"- **Leaks detected:** {len(leaks)}",
        "",
        f"## Isolation Integrity: **{isolation_pct:.1f}%**",
        "",
        ("✅ **100% isolation — zero cross-tenant leakage.**"
         if not leaks else
         "❌ **ISOLATION BREACH** — see below."),
        "",
    ]
    if leaks:
        lines.append("### Breaches")
        lines += [f"- {l}" for l in leaks]
        lines.append("")
    report.write_text("\n".join(lines))

    print(f"\nIsolation integrity: {isolation_pct:.1f}%  ({denied} denied, {len(leaks)} leaks)")
    print(f"Report: {report}")
    return 0 if not leaks else 1


if __name__ == "__main__":
    raise SystemExit(main())
