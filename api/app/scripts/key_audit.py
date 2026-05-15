"""
Key audit — list all API keys with usage stats and flag anomalies.

Run inside the container: python -m app.scripts.key_audit

Output: human-readable table to stdout + JSON dump to /opt/agentssot/audits/key_audit_YYYY-MM-DD.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


STALE_DAYS = 30
AUDIT_DIR = Path("/opt/agentssot/audits")

ADMIN_KEY_ENV = os.environ.get("HIVE_ADMIN_KEY", "")


def _get_admin_key() -> str:
    if ADMIN_KEY_ENV:
        return ADMIN_KEY_ENV
    for candidate in [
        os.path.expanduser("~/.claude/agentssot/local/admin.json"),
        "/root/.claude/agentssot/local/admin.json",
    ]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f).get("admin_api_key", "")
    return ""


def _get_db_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    return url.replace("postgresql+psycopg://", "postgresql://") if url else ""


def main() -> None:
    admin_key = _get_admin_key()
    if not admin_key:
        print("ERROR: No admin key. Set HIVE_ADMIN_KEY or provide admin.json.")
        sys.exit(1)

    db_url = _get_db_url()
    if not db_url:
        print("ERROR: DATABASE_URL not set.")
        sys.exit(1)

    try:
        import psycopg
        conn = psycopg.connect(db_url)
    except Exception as e:
        print(f"ERROR: DB connection failed: {e}")
        sys.exit(1)

    now = datetime.now(timezone.utc)
    stale_threshold = now - timedelta(days=STALE_DAYS)

    keys = conn.execute("""
        SELECT
            id::text,
            name,
            role,
            namespaces,
            is_active,
            created_at
        FROM api_keys
        ORDER BY created_at DESC
    """).fetchall()

    # Usage stats: count writes and reads from knowledge_items source field
    # The ingest path stores source as the agent key name
    write_counts = {row[0]: row[1] for row in conn.execute("""
        SELECT source, count(*)
        FROM knowledge_items
        WHERE source IS NOT NULL
        GROUP BY source
    """).fetchall()}

    # Recall events for read counts
    recall_counts = {row[0]: row[1] for row in conn.execute("""
        SELECT agent_key, count(*)
        FROM recall_events
        GROUP BY agent_key
    """).fetchall()}

    # Last write per key (by source name match)
    last_write = {row[0]: row[1] for row in conn.execute("""
        SELECT source, max(created_at)
        FROM knowledge_items
        WHERE source IS NOT NULL
        GROUP BY source
    """).fetchall()}

    conn.close()

    audit_rows = []
    flags_summary = []

    for row in keys:
        key_id, name, role, namespaces, is_active, created_at = row
        id_tail = key_id[-4:] if key_id else "????"
        ns_list = list(namespaces or [])
        has_wildcard = "*" in ns_list

        total_writes = write_counts.get(name, 0)
        total_reads = recall_counts.get(name, 0) + recall_counts.get(f"device-{name}-writer", 0)
        last_w = last_write.get(name)
        last_used_str = last_w.isoformat() if last_w else None

        flags = []
        if has_wildcard:
            flags.append("wildcard-scope")
        if not is_active:
            flags.append("inactive")
        if last_w is None and created_at < stale_threshold:
            flags.append("never-used")
        elif last_w and last_w < stale_threshold:
            flags.append("stale(>30d)")

        audit_rows.append({
            "id_tail": id_tail,
            "name": name,
            "role": role,
            "namespaces": ns_list,
            "is_active": is_active,
            "created_at": created_at.isoformat() if created_at else None,
            "last_write_at": last_used_str,
            "total_writes": total_writes,
            "total_reads": total_reads,
            "flags": flags,
        })
        if flags:
            flags_summary.append((name, flags))

    # Print human table
    col_widths = [6, 35, 7, 12, 10, 10, 12]
    headers = ["...ID", "Name", "Role", "Namespaces", "Writes", "Reads", "Flags"]
    sep = " | ".join("-" * w for w in col_widths)
    header = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header)
    print(sep)

    for r in audit_rows:
        ns_display = ",".join(r["namespaces"])[:12] if r["namespaces"] else ""
        flags_display = ",".join(r["flags"])[:12] if r["flags"] else ""
        print(" | ".join([
            r["id_tail"].ljust(6),
            r["name"][:35].ljust(35),
            (r["role"] or "")[:7].ljust(7),
            ns_display.ljust(12),
            str(r["total_writes"]).ljust(10),
            str(r["total_reads"]).ljust(10),
            flags_display.ljust(12),
        ]))

    print(f"\nTotal keys: {len(audit_rows)}")
    print(f"Flagged: {len(flags_summary)}")
    if flags_summary:
        print("\nFlagged keys:")
        for name, flags in flags_summary[:20]:
            print(f"  {name}: {', '.join(flags)}")

    # Write JSON dump
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = now.strftime("%Y-%m-%d")
    out_path = AUDIT_DIR / f"key_audit_{date_str}.json"
    with open(out_path, "w") as f:
        json.dump({
            "generated_at": now.isoformat(),
            "total_keys": len(audit_rows),
            "flagged_count": len(flags_summary),
            "keys": audit_rows,
        }, f, indent=2)
    print(f"\nJSON dump: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
