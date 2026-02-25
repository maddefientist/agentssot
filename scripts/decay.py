#!/usr/bin/env python3
"""
Nightly decay: reduce strength of unused knowledge items and concept confidence.
Runs independently of the synthesis loop for reliability.

Knowledge items: strength decays if not recalled in 30+ days.
Concepts: confidence decays if not recalled/updated in grace period.

Cron: 0 3 * * * (3 AM daily)
"""
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [decay] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("cortex-decay")


def psql(sql: str) -> str:
    """Execute SQL via docker exec."""
    result = subprocess.run(
        ["docker", "exec", "agentssot-db", "psql", "-U", os.environ.get("POSTGRES_USER", "ssot"), "-d", os.environ.get("POSTGRES_DB", "ssot"), "-t", "-A", "-c", sql],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        log.error("SQL error: %s", result.stderr.strip())
        return ""
    return result.stdout.strip()


def decay_knowledge_items():
    """Decay strength of knowledge items not recalled recently."""
    # Items not recalled in 30+ days AND older than 30 days
    decayed = psql("""
        UPDATE knowledge_items
        SET strength = GREATEST(
            strength - CASE
                WHEN COALESCE(positive_feedback, 0) > 3 THEN 0.01
                ELSE 0.03
            END,
            0.1
        )
        WHERE (last_recalled_at < NOW() - INTERVAL '30 days' OR last_recalled_at IS NULL)
          AND created_at < NOW() - INTERVAL '30 days'
          AND COALESCE(strength, 1.0) > 0.1
          AND COALESCE(status, 'active') = 'active'
        RETURNING id
    """)
    count = len(decayed.splitlines()) if decayed else 0
    log.info("Knowledge items decayed: %d", count)

    # Mark dormant: not recalled in 90+ days AND low strength
    dormant = psql("""
        UPDATE knowledge_items
        SET status = 'dormant'
        WHERE (last_recalled_at < NOW() - INTERVAL '90 days' OR last_recalled_at IS NULL)
          AND created_at < NOW() - INTERVAL '90 days'
          AND COALESCE(status, 'active') = 'active'
          AND COALESCE(strength, 1.0) <= 0.3
        RETURNING id
    """)
    dormant_count = len(dormant.splitlines()) if dormant else 0
    log.info("Knowledge items marked dormant: %d", dormant_count)

    # Archive candidates: dormant for 90+ more days
    archived = psql("""
        UPDATE knowledge_items
        SET status = 'archive_candidate'
        WHERE (last_recalled_at < NOW() - INTERVAL '180 days' OR last_recalled_at IS NULL)
          AND created_at < NOW() - INTERVAL '180 days'
          AND status = 'dormant'
        RETURNING id
    """)
    archived_count = len(archived.splitlines()) if archived else 0
    log.info("Knowledge items archive candidates: %d", archived_count)


def decay_concept_links():
    """Weaken concept graph links not reinforced recently."""
    weakened = psql("""
        UPDATE concept_links
        SET weight = GREATEST(weight - 0.01, 0.1)
        WHERE updated_at < NOW() - INTERVAL '30 days' AND weight > 0.1
        RETURNING id
    """)
    count = len(weakened.splitlines()) if weakened else 0
    log.info("Concept links weakened: %d", count)

    # Prune dead links
    pruned = psql("""
        DELETE FROM concept_links
        WHERE weight <= 0.1 AND co_occurrence_count <= 1
          AND updated_at < NOW() - INTERVAL '90 days'
        RETURNING id
    """)
    pruned_count = len(pruned.splitlines()) if pruned else 0
    log.info("Concept links pruned: %d", pruned_count)


def log_health_summary():
    """Log current health stats."""
    ki_stats = psql("""
        SELECT COALESCE(status, 'active') as status, COUNT(*), ROUND(AVG(COALESCE(strength, 1.0))::numeric, 2)
        FROM knowledge_items
        WHERE namespace = 'claude-shared'
        GROUP BY COALESCE(status, 'active')
        ORDER BY 1
    """)
    if ki_stats:
        log.info("Knowledge items by status:")
        for line in ki_stats.splitlines():
            parts = line.split("|")
            if len(parts) == 3:
                log.info("  %s: %s items, avg strength %s", parts[0].strip(), parts[1].strip(), parts[2].strip())

    concept_stats = psql("""
        SELECT
            COUNT(*) FILTER (WHERE NOT tags @> ARRAY['superseded']::text[] AND NOT tags @> ARRAY['dormant']::text[]) as active,
            COUNT(*) FILTER (WHERE tags @> ARRAY['dormant']::text[]) as dormant,
            COUNT(*) FILTER (WHERE tags @> ARRAY['superseded']::text[]) as superseded,
            ROUND(AVG(confidence)::numeric, 3) as avg_conf
        FROM concepts WHERE namespace = 'claude-shared'
    """)
    if concept_stats:
        log.info("Concepts: %s", concept_stats)

    graph_stats = psql("SELECT COUNT(*) FROM concept_links")
    if graph_stats:
        log.info("Concept graph edges: %s", graph_stats)


def main():
    log.info("Starting nightly decay cycle")
    decay_knowledge_items()
    decay_concept_links()
    log_health_summary()
    log.info("Decay cycle complete")


if __name__ == "__main__":
    main()
