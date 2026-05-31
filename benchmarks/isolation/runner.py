#!/usr/bin/env python3
"""Namespace isolation benchmark — standalone report generator.

Ported from ArcRift's `mcp-stress-test.ts` (lift #4). Writes a secret into N
namespaces, each behind its own scoped writer key, then issues the full N×(N-1)
cross-tenant access matrix and certifies zero leakage. Emits a markdown report
to benchmarks/isolation/results/isolation_report.md.

Two modes:

  Ephemeral (default) — unique run-tagged namespaces. For ad-hoc local runs.
    python benchmarks/isolation/runner.py --tenants 10

  CI / post-deploy (--ci) — FIXED probe namespaces (`_iso_probe_N`) with
    deterministic secrets (ingest dedup keeps exactly 1 item/namespace, so it
    never grows) and a locally-cached set of scoped keys (created once, reused
    every deploy → zero key accumulation). Designed to be a deploy gate: exits
    non-zero if any cross-tenant leak is detected.
    SSOT_TEST_ADMIN_KEY=... python benchmarks/isolation/runner.py --ci

The pytest sibling (api/tests/test_namespace_isolation.py) asserts the property;
this runner produces the human-readable isolation certificate + the deploy gate.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_KEY_CACHE = Path(__file__).resolve().parents[2] / ".isolation-keys.json"


def _provision_ephemeral(c: httpx.Client, admin: dict, n: int) -> list[dict]:
    run = uuid.uuid4().hex[:8]
    tenants: list[dict] = []
    for i in range(n):
        ns = f"isolation-{run}-{i}"
        secret = f"ISOLATIONSECRET-{run}-{i}-{uuid.uuid4().hex}"
        _ensure_namespace(c, admin, ns)
        key = _create_key(c, admin, f"key-{ns}", ns)
        _ingest_secret(c, key, ns, secret)
        tenants.append({"ns": ns, "secret": secret, "key": key})
    return tenants


def _provision_ci(c: httpx.Client, admin: dict, n: int, cache_path: Path) -> list[dict]:
    """Fixed probe namespaces + cached keys. Idempotent across deploys."""
    cached: list[dict] = []
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except Exception:
            cached = []

    tenants: list[dict] = []
    for i in range(n):
        ns = f"_iso_probe_{i}"
        secret = f"ISOLATIONPROBE-{i}"  # deterministic → ingest dedup, no growth
        hit = next((t for t in cached if t.get("ns") == ns and t.get("key")), None)
        if hit:
            key = hit["key"]
        else:
            _ensure_namespace(c, admin, ns)
            key = _create_key(c, admin, f"_iso_probe_key_{i}", ns)
        _ensure_namespace(c, admin, ns)
        _ingest_secret(c, key, ns, secret)  # deduped if unchanged
        tenants.append({"ns": ns, "secret": secret, "key": key})

    # Persist the (newly created or reused) keys, locked down.
    cache_path.write_text(json.dumps([{"ns": t["ns"], "key": t["key"]} for t in tenants]))
    try:
        cache_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600 — contains live keys
    except OSError:
        pass
    return tenants


def _ensure_namespace(c: httpx.Client, admin: dict, ns: str) -> None:
    r = c.post("/admin/namespaces", headers=admin, json={"name": ns})
    if r.status_code not in (200, 201, 409):
        raise RuntimeError(f"namespace {ns} failed: {r.status_code} {r.text}")


def _create_key(c: httpx.Client, admin: dict, name: str, ns: str) -> str:
    r = c.post("/admin/api-keys", headers=admin,
               json={"name": name, "role": "writer", "namespaces": [ns]})
    r.raise_for_status()
    return r.json()["api_key"]


def _ingest_secret(c: httpx.Client, key: str, ns: str, secret: str) -> None:
    r = c.post("/ingest", headers={"X-Api-Key": key},
               json={"namespace": ns,
                     "knowledge_items": [{"content": f"deploy token is {secret}",
                                          "tags": ["isolation-test"]}]})
    r.raise_for_status()


def _run_matrix(c: httpx.Client, tenants: list[dict]) -> dict:
    total_probes = denied = own_ok = 0
    leaks: list[str] = []
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
                leaks.append(f"{t['ns']} → {other['ns']} (read foreign secret via /query)")
            elif r.status_code != 200:
                leaks.append(f"{t['ns']} → {other['ns']} unexpected status {r.status_code}")
    return {"total": total_probes, "denied": denied, "own_ok": own_ok, "leaks": leaks}


def _write_report(args, n: int, res: dict) -> Path:
    total, leaks = res["total"], res["leaks"]
    pct = 100.0 * (total - len(leaks)) / total if total else 100.0
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = RESULTS_DIR / "isolation_report.md"
    lines = [
        "# Namespace Isolation Report", "",
        f"- **Generated:** {ts}",
        f"- **Mode:** {'CI / post-deploy (fixed probes)' if args.ci else 'ephemeral'}",
        f"- **Target:** `{args.url}`",
        f"- **Tenants:** {n}",
        f"- **Cross-tenant probes:** {total} (N×(N-1))",
        f"- **Denied by RBAC (401/403):** {res['denied']}",
        f"- **Own-namespace read OK:** {res['own_ok']}/{n}",
        f"- **Leaks detected:** {len(leaks)}", "",
        f"## Isolation Integrity: **{pct:.1f}%**", "",
        ("✅ **100% isolation — zero cross-tenant leakage.**" if not leaks
         else "❌ **ISOLATION BREACH** — see below."), "",
    ]
    if leaks:
        lines.append("### Breaches")
        lines += [f"- {l}" for l in leaks] + [""]
    report.write_text("\n".join(lines))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenants", type=int, default=int(os.environ.get("SSOT_ISOLATION_TENANTS", "10")))
    ap.add_argument("--url", default=os.environ.get("SSOT_TEST_URL", "http://localhost:8088"))
    ap.add_argument("--admin-key", default=os.environ.get("SSOT_TEST_ADMIN_KEY", ""))
    ap.add_argument("--ci", action="store_true",
                    help="deploy-gate mode: fixed probe namespaces + cached keys, no cruft")
    ap.add_argument("--key-cache", default=os.environ.get("SSOT_ISOLATION_KEY_CACHE", str(DEFAULT_KEY_CACHE)))
    args = ap.parse_args()

    if not args.admin_key:
        print("ERROR: set SSOT_TEST_ADMIN_KEY (admin API key).", file=sys.stderr)
        return 2

    n = 3 if args.ci and args.tenants > 5 else args.tenants  # CI keeps a tiny fixed pool
    admin = {"X-Api-Key": args.admin_key}

    with httpx.Client(base_url=args.url, timeout=30) as c:
        mode = "CI/post-deploy" if args.ci else "ephemeral"
        print(f"Provisioning {n} tenants ({mode})...")
        if args.ci:
            tenants = _provision_ci(c, admin, n, Path(args.key_cache))
        else:
            tenants = _provision_ephemeral(c, admin, n)

        print("Running cross-tenant access matrix...")
        res = _run_matrix(c, tenants)

    report = _write_report(args, n, res)
    pct = 100.0 * (res["total"] - len(res["leaks"])) / res["total"] if res["total"] else 100.0
    print(f"\nIsolation integrity: {pct:.1f}%  ({res['denied']} denied, {len(res['leaks'])} leaks)")
    if res["own_ok"] != n:
        print(f"WARNING: only {res['own_ok']}/{n} keys could read their own namespace", file=sys.stderr)
    print(f"Report: {report}")
    return 0 if not res["leaks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
