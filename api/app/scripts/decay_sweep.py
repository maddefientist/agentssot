"""
Decay sweep — surface stale entity knowledge items for human review.

Run inside the container: python -m app.scripts.decay_sweep

Flags items where:
  - memory_type = 'entity' (or tags contain 'entity')
  - created_at older than 60 days
  - content matches substrate-fact patterns (model versions, IPs, ports, hostnames)

Output: JSON to stdout.
Also ingests flagged items into claude-shared namespace tagged [decay-review].
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta

DECAY_THRESHOLD_DAYS = 60

SUBSTRATE_PATTERNS = [
    re.compile(r'Claude\s+(Opus|Sonnet|Haiku)\s+[\d.]+', re.IGNORECASE),
    re.compile(r'(Opus|Sonnet|Haiku)\s+[\d.]+', re.IGNORECASE),
    re.compile(r'(claude-|gpt-|gemini-)[\d\w\-\.]+', re.IGNORECASE),
    re.compile(r'Powered by\s+\S+', re.IGNORECASE),
    re.compile(r'\b192\.168\.\d+\.\d+\b'),
    re.compile(r'\b10\.\d+\.\d+\.\d+\b'),
    re.compile(r':\d{4,5}\b'),
    re.compile(r'\b(hari|webvm|dockers|blink|unraid|mypi|air|zoria)\b', re.IGNORECASE),
]

BASE_URL = os.environ.get("HIVE_BASE_URL", "http://localhost:8000")
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


def _matches_substrate(content: str) -> list[str]:
    matches = []
    for pat in SUBSTRATE_PATTERNS:
        m = pat.search(content)
        if m:
            matches.append(m.group(0))
    return matches


def _get_db_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if url:
        # Convert SQLAlchemy dialect to plain psycopg3 URL
        return url.replace("postgresql+psycopg://", "postgresql://")
    return ""


def main() -> None:
    admin_key = _get_admin_key()
    if not admin_key:
        print(json.dumps({"error": "No admin key. Set HIVE_ADMIN_KEY or provide admin.json."}))
        sys.exit(1)

    threshold = datetime.now(timezone.utc) - timedelta(days=DECAY_THRESHOLD_DAYS)
    cutoff_str = threshold.isoformat()

    flagged: list[dict] = []

    db_url = _get_db_url()
    if db_url:
        try:
            import psycopg
            conn = psycopg.connect(db_url)
            rows = conn.execute(
                """
                SELECT id::text, namespace, content, created_at::text, tags, memory_type
                FROM knowledge_items
                WHERE (memory_type = 'entity' OR 'entity' = ANY(COALESCE(tags, '{}'::text[])))
                  AND created_at < %s
                  AND status = 'active'
                ORDER BY created_at ASC
                """,
                (threshold,),
            ).fetchall()
            conn.close()

            for item_id, namespace, content, created_at, tags, memory_type in rows:
                matches = _matches_substrate(content or "")
                if matches:
                    flagged.append({
                        "id": item_id,
                        "namespace": namespace,
                        "created_at": created_at,
                        "memory_type": memory_type,
                        "substrate_matches": matches,
                        "content_preview": (content or "")[:300],
                    })
        except Exception as e:
            print(json.dumps({"error": f"DB error: {e}"}), file=sys.stderr)
            sys.exit(1)
    else:
        print(json.dumps({"error": "DATABASE_URL not set."}), file=sys.stderr)
        sys.exit(1)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decay_threshold_days": DECAY_THRESHOLD_DAYS,
        "flagged_count": len(flagged),
        "items": flagged,
    }
    print(json.dumps(report, indent=2))

    if flagged and admin_key:
        _ingest_decay_entries(flagged, admin_key)
        print(f"# Ingested {len(flagged)} decay-review entries into claude-shared", file=sys.stderr)


def _ingest_decay_entries(flagged: list[dict], admin_key: str) -> None:
    import urllib.request
    for item in flagged:
        body = json.dumps({
            "namespace": "claude-shared",
            "content": (
                f"[decay-review] {item['id']} ({item['namespace']})\n"
                f"Created: {item['created_at']}\n"
                f"Substrate matches: {', '.join(item['substrate_matches'])}\n"
                f"Preview: {item['content_preview']}"
            ),
            "memory_type": "fact",
            "tags": ["decay-review", f"origin-ns:{item['namespace']}"],
        }).encode()
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/ingest",
                data=body,
                headers={"X-API-Key": admin_key, "Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            print(f"# ingest error for {item['id']}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
