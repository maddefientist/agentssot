#!/usr/bin/env python3
"""
Build concept associative graph from evidence overlap and tag co-occurrence.
Two concepts that share evidence_ids get linked. Weight scales with overlap count.

Run once to seed, then maintained incrementally by the API on new synthesis.
"""
import os
import sys
from itertools import combinations

import psycopg2
import psycopg2.extras

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://ssot:REDACTED_DB_PASSWORD@localhost:5432/ssot",
).replace("postgresql+psycopg://", "postgresql://")

# If running inside docker network, db host is 'db'
if "db:5432" in DB_URL:
    DB_URL = DB_URL.replace("db:5432", "localhost:5432")


def build_evidence_graph(cur):
    """Link concepts that share evidence items."""
    cur.execute("""
        SELECT id, evidence_ids, tags
        FROM concepts
        WHERE namespace = 'claude-shared'
          AND NOT (tags @> ARRAY['superseded'])
          AND array_length(evidence_ids, 1) > 0
    """)
    rows = cur.fetchall()
    print(f"Found {len(rows)} concepts with evidence_ids")

    # Build evidence_id -> [concept_id] mapping
    evidence_map: dict[str, list[str]] = {}
    for concept_id, evidence_ids, tags in rows:
        for eid in (evidence_ids or []):
            evidence_map.setdefault(str(eid), []).append(str(concept_id))

    # Find concept pairs that share evidence
    pair_counts: dict[tuple[str, str], int] = {}
    for eid, concept_ids in evidence_map.items():
        if len(concept_ids) < 2:
            continue
        for a, b in combinations(sorted(set(concept_ids)), 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    print(f"Found {len(pair_counts)} concept pairs with shared evidence")

    inserted = 0
    for (a, b), count in pair_counts.items():
        weight = min(count * 0.3, 5.0)
        cur.execute("""
            INSERT INTO concept_links (concept_a, concept_b, weight, co_occurrence_count, link_type)
            VALUES (%s, %s, %s, %s, 'evidence_overlap')
            ON CONFLICT (concept_a, concept_b)
            DO UPDATE SET
                weight = GREATEST(concept_links.weight, EXCLUDED.weight),
                co_occurrence_count = concept_links.co_occurrence_count + EXCLUDED.co_occurrence_count,
                updated_at = NOW()
        """, (a, b, weight, count))
        inserted += 1

    return inserted


def build_tag_graph(cur):
    """Link concepts that share tags (excluding generic tags)."""
    SKIP_TAGS = {"superseded", "dormant", "contested", "consensus"}

    cur.execute("""
        SELECT id, tags
        FROM concepts
        WHERE namespace = 'claude-shared'
          AND NOT (tags @> ARRAY['superseded'])
          AND array_length(tags, 1) > 0
    """)
    rows = cur.fetchall()

    tag_map: dict[str, list[str]] = {}
    for concept_id, tags in rows:
        for tag in (tags or []):
            if tag not in SKIP_TAGS:
                tag_map.setdefault(tag, []).append(str(concept_id))

    pair_counts: dict[tuple[str, str], int] = {}
    for tag, concept_ids in tag_map.items():
        if len(concept_ids) < 2:
            continue
        for a, b in combinations(sorted(set(concept_ids)), 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    print(f"Found {len(pair_counts)} concept pairs with shared tags")

    inserted = 0
    for (a, b), count in pair_counts.items():
        weight = min(count * 0.2, 3.0)
        cur.execute("""
            INSERT INTO concept_links (concept_a, concept_b, weight, co_occurrence_count, link_type)
            VALUES (%s, %s, %s, %s, 'tag_overlap')
            ON CONFLICT (concept_a, concept_b)
            DO UPDATE SET
                weight = concept_links.weight + %s,
                co_occurrence_count = concept_links.co_occurrence_count + %s,
                updated_at = NOW()
        """, (a, b, weight, count, weight * 0.5, count))
        inserted += 1

    return inserted


def main():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    evidence_links = build_evidence_graph(cur)
    tag_links = build_tag_graph(cur)

    conn.commit()

    # Stats
    cur.execute("SELECT COUNT(*) FROM concept_links")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT concept_a) + COUNT(DISTINCT concept_b) FROM concept_links")
    nodes = cur.fetchone()[0]
    cur.execute("SELECT concept_a, concept_b, weight FROM concept_links ORDER BY weight DESC LIMIT 10")
    top_links = cur.fetchall()

    print(f"\nGraph built: {total} edges, ~{nodes} unique concept nodes")
    print(f"  Evidence links: {evidence_links}")
    print(f"  Tag links: {tag_links}")
    print(f"\nTop 10 strongest links:")
    for a, b, w in top_links:
        cur.execute("SELECT title FROM concepts WHERE id = %s", (str(a),))
        title_a = cur.fetchone()[0] if cur.rowcount else "?"
        cur.execute("SELECT title FROM concepts WHERE id = %s", (str(b),))
        title_b = cur.fetchone()[0] if cur.rowcount else "?"
        print(f"  {w:.1f}  {title_a[:40]} <-> {title_b[:40]}")

    conn.close()


if __name__ == "__main__":
    main()
