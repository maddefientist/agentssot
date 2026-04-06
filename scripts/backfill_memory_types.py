#!/usr/bin/env python3
"""Backfill memory_type on existing knowledge items using tag heuristics.

DRY RUN by default. Use --commit to apply changes.

Usage:
    python scripts/backfill_memory_types.py                    # dry run
    python scripts/backfill_memory_types.py --commit           # apply changes
    python scripts/backfill_memory_types.py --namespace foo    # specific namespace
    python scripts/backfill_memory_types.py --verbose          # show each classification
"""

import argparse
import os
import sys
from collections import Counter

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


# ── Heuristic classification rules ──────────────────────────────────
# Each rule is (tag_set_check, source_check, content_check, memory_type)
# First match wins. Items that match no rule stay NULL (will default to 'fact' in display).

RULES = [
    # Corrections from feedback
    {
        "name": "correction_by_tag",
        "match": lambda item: _has_any_tag(item, ["correction", "operator-feedback"]),
        "memory_type": "correction",
    },
    # Session summaries / compaction output
    {
        "name": "session_summary_by_tag",
        "match": lambda item: _has_any_tag(item, ["summary", "compaction"]) or item.source == "session_compaction",
        "memory_type": "session_summary",
    },
    # Skills taught by operator
    {
        "name": "skill_by_tag",
        "match": lambda item: _has_any_tag(item, ["skill", "operator-taught"]),
        "memory_type": "skill",
    },
    # Preferences
    {
        "name": "preference_by_tag",
        "match": lambda item: _has_any_tag(item, ["preference", "user-preference", "config"]),
        "memory_type": "preference",
    },
    # Decisions
    {
        "name": "decision_by_tag",
        "match": lambda item: _has_any_tag(item, ["decision", "architecture", "adr"]),
        "memory_type": "decision",
    },
    # References / documentation
    {
        "name": "reference_by_tag",
        "match": lambda item: _has_any_tag(item, ["reference", "documentation", "api-doc", "docs"]),
        "memory_type": "reference",
    },
    # Session-extracted facts (most common catch-all before default)
    {
        "name": "fact_by_extraction",
        "match": lambda item: _has_any_tag(item, ["session-extract", "extracted", "auto-extract"]),
        "memory_type": "fact",
    },
    # Content-based heuristics (weaker signals, checked last)
    {
        "name": "decision_by_content",
        "match": lambda item: item.content and any(
            phrase in item.content.lower()
            for phrase in ["decided to", "decision:", "we chose", "architecture decision"]
        ),
        "memory_type": "decision",
    },
    {
        "name": "preference_by_content",
        "match": lambda item: item.content and any(
            phrase in item.content.lower()
            for phrase in ["prefers ", "preference:", "user wants", "always use"]
        ),
        "memory_type": "preference",
    },
]


def _has_any_tag(item, tag_list: list[str]) -> bool:
    """Check if item has any of the specified tags (case-insensitive)."""
    item_tags = set(t.lower() for t in (item.tags or []))
    return bool(item_tags & set(t.lower() for t in tag_list))


def classify_item(item) -> tuple[str | None, str]:
    """Classify a knowledge item. Returns (memory_type, rule_name) or (None, 'no_match')."""
    for rule in RULES:
        try:
            if rule["match"](item):
                return rule["memory_type"], rule["name"]
        except Exception:
            continue
    return None, "no_match"


def main():
    parser = argparse.ArgumentParser(description="Backfill memory_type on knowledge items")
    parser.add_argument("--commit", action="store_true", help="Apply changes (default: dry run)")
    parser.add_argument("--namespace", type=str, default=None, help="Restrict to namespace")
    parser.add_argument("--verbose", action="store_true", help="Show each classification")
    parser.add_argument("--limit", type=int, default=None, help="Max items to process")
    parser.add_argument("--only-untyped", action="store_true", default=True,
                        help="Only classify items without memory_type (default)")
    args = parser.parse_args()

    # Import after path setup
    from sqlalchemy import select, and_
    from app.db import SessionLocal
    from app.models import KnowledgeItem

    mode = "COMMIT" if args.commit else "DRY RUN"
    print(f"=== Backfill memory_type [{mode}] ===\n")

    stats = Counter()
    rule_stats = Counter()

    with SessionLocal() as session:
        stmt = select(KnowledgeItem)

        if args.namespace:
            stmt = stmt.where(KnowledgeItem.namespace == args.namespace)
            print(f"Namespace filter: {args.namespace}")

        if args.only_untyped:
            stmt = stmt.where(KnowledgeItem.memory_type.is_(None))
            print("Processing: only items without memory_type")

        stmt = stmt.order_by(KnowledgeItem.created_at.asc())

        if args.limit:
            stmt = stmt.limit(args.limit)
            print(f"Limit: {args.limit}")

        print()
        items = session.scalars(stmt).all()
        print(f"Found {len(items)} items to classify\n")

        for item in items:
            memory_type, rule_name = classify_item(item)

            if memory_type is not None:
                stats[memory_type] += 1
                rule_stats[rule_name] += 1

                if args.verbose:
                    snippet = (item.content or "")[:80].replace("\n", " ")
                    tags_str = ", ".join(item.tags or [])
                    print(f"  [{memory_type:16s}] ({rule_name}) tags=[{tags_str}] \"{snippet}...\"")

                if args.commit:
                    item.memory_type = memory_type
            else:
                stats["_unclassified"] += 1
                if args.verbose:
                    snippet = (item.content or "")[:80].replace("\n", " ")
                    tags_str = ", ".join(item.tags or [])
                    print(f"  [{'unclassified':16s}] tags=[{tags_str}] \"{snippet}...\"")

        if args.commit:
            session.commit()
            print("Changes committed.\n")
        else:
            session.rollback()
            print("Dry run — no changes made.\n")

    # Summary
    print("=== Classification Summary ===")
    total = sum(stats.values())
    classified = total - stats.get("_unclassified", 0)
    print(f"Total items:   {total}")
    print(f"Classified:    {classified}")
    print(f"Unclassified:  {stats.get('_unclassified', 0)}")
    print()

    if stats:
        print("By memory_type:")
        for mt, count in sorted(stats.items()):
            if mt != "_unclassified":
                print(f"  {mt:20s}: {count}")

    if rule_stats:
        print("\nBy rule:")
        for rule, count in sorted(rule_stats.items(), key=lambda x: -x[1]):
            print(f"  {rule:30s}: {count}")

    if not args.commit:
        print(f"\nRe-run with --commit to apply these classifications.")


if __name__ == "__main__":
    main()
