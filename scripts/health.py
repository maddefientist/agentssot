#!/usr/bin/env python3
"""
Weekly health report: generates a markdown report of system state.
Cron: 0 5 * * 0 (Sunday 5 AM, after consolidation)
"""
import os
import subprocess
import sys
from datetime import datetime

REPORT_DIR = os.path.expanduser("~/cortex-health")


def psql(sql: str) -> str:
    result = subprocess.run(
        ["docker", "exec", "agentssot-db", "psql", "-U", "ssot", "-d", "ssot", "-t", "-A", "-c", sql],
        capture_output=True, text=True, timeout=60,
    )
    return result.stdout.strip() if result.returncode == 0 else f"ERROR: {result.stderr.strip()}"


def generate_report() -> str:
    lines = [
        f"# Cortex Health Report — {datetime.now().strftime('%Y-%m-%d')}",
        "",
    ]

    # Knowledge items
    ki = psql("""
        SELECT COALESCE(status, 'active') as st, COUNT(*), ROUND(AVG(COALESCE(strength, 1.0))::numeric, 2)
        FROM knowledge_items WHERE namespace = 'claude-shared'
        GROUP BY COALESCE(status, 'active') ORDER BY 1
    """)
    lines.append("## Knowledge Items")
    lines.append("| Status | Count | Avg Strength |")
    lines.append("|--------|-------|-------------|")
    for row in ki.splitlines():
        parts = [p.strip() for p in row.split("|")]
        if len(parts) == 3:
            lines.append(f"| {parts[0]} | {parts[1]} | {parts[2]} |")
    lines.append("")

    # Concepts
    concepts = psql("""
        SELECT type::text, COUNT(*), ROUND(AVG(confidence)::numeric, 3)
        FROM concepts WHERE namespace = 'claude-shared'
          AND NOT (tags @> ARRAY['superseded']::text[])
        GROUP BY type ORDER BY count DESC
    """)
    lines.append("## Concepts (active)")
    lines.append("| Type | Count | Avg Confidence |")
    lines.append("|------|-------|---------------|")
    for row in concepts.splitlines():
        parts = [p.strip() for p in row.split("|")]
        if len(parts) == 3:
            lines.append(f"| {parts[0]} | {parts[1]} | {parts[2]} |")
    lines.append("")

    # Recall activity (last 7 days)
    recall = psql("""
        SELECT COUNT(*) as total_recalls,
               COUNT(DISTINCT session_id) as unique_sessions
        FROM recall_events
        WHERE created_at > NOW() - INTERVAL '7 days'
    """)
    lines.append("## Recall Activity (Last 7 Days)")
    lines.append(f"- {recall}")
    lines.append("")

    ki_recall = psql("""
        SELECT COUNT(*) FILTER (WHERE last_recalled_at > NOW() - INTERVAL '7 days') as recalled_items,
               ROUND(AVG(recall_count) FILTER (WHERE recall_count > 0)::numeric, 1) as avg_recalls
        FROM knowledge_items WHERE namespace = 'claude-shared'
    """)
    lines.append(f"- Knowledge items recalled: {ki_recall}")
    lines.append("")

    # Feedback stats
    feedback = psql("""
        SELECT signal::text, COUNT(*)
        FROM concept_feedback
        WHERE created_at > NOW() - INTERVAL '7 days'
        GROUP BY signal
    """)
    lines.append("## Feedback (Last 7 Days)")
    if feedback:
        for row in feedback.splitlines():
            parts = [p.strip() for p in row.split("|")]
            if len(parts) == 2:
                lines.append(f"- {parts[0]}: {parts[1]}")
    else:
        lines.append("- No feedback recorded")
    lines.append("")

    # Concept graph
    graph = psql("""
        SELECT COUNT(*) as edges,
               ROUND(AVG(weight)::numeric, 2) as avg_weight,
               MAX(weight) as max_weight
        FROM concept_links
    """)
    lines.append("## Concept Graph")
    lines.append(f"- {graph}")
    lines.append("")

    # Top 10 strongest knowledge items
    top_ki = psql("""
        SELECT LEFT(content, 60), strength, recall_count
        FROM knowledge_items
        WHERE namespace = 'claude-shared' AND COALESCE(status, 'active') = 'active'
        ORDER BY strength DESC LIMIT 10
    """)
    lines.append("## Top 10 Strongest Knowledge Items")
    lines.append("| Content | Strength | Recalls |")
    lines.append("|---------|----------|---------|")
    for row in top_ki.splitlines():
        parts = [p.strip() for p in row.split("|")]
        if len(parts) == 3:
            lines.append(f"| {parts[0][:55]} | {parts[1]} | {parts[2]} |")
    lines.append("")

    # Top 10 weakest active items (decay candidates)
    weak_ki = psql("""
        SELECT LEFT(content, 60), strength, recall_count, created_at::date
        FROM knowledge_items
        WHERE namespace = 'claude-shared' AND COALESCE(status, 'active') = 'active'
        ORDER BY strength ASC LIMIT 10
    """)
    lines.append("## Top 10 Weakest Knowledge Items (Decay Candidates)")
    lines.append("| Content | Strength | Recalls | Created |")
    lines.append("|---------|----------|---------|---------|")
    for row in weak_ki.splitlines():
        parts = [p.strip() for p in row.split("|")]
        if len(parts) == 4:
            lines.append(f"| {parts[0][:55]} | {parts[1]} | {parts[2]} | {parts[3]} |")
    lines.append("")

    # Ingestion rate
    ingestion = psql("""
        SELECT DATE(created_at) as dt, COUNT(*)
        FROM knowledge_items
        WHERE created_at > NOW() - INTERVAL '7 days'
        GROUP BY DATE(created_at) ORDER BY 1 DESC
    """)
    lines.append("## Ingestion Rate (Last 7 Days)")
    for row in ingestion.splitlines():
        parts = [p.strip() for p in row.split("|")]
        if len(parts) == 2:
            lines.append(f"- {parts[0]}: {parts[1]} items")
    lines.append("")

    return "\n".join(lines)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    report = generate_report()
    filename = f"report-{datetime.now().strftime('%Y-%m-%d')}.md"
    filepath = os.path.join(REPORT_DIR, filename)
    with open(filepath, "w") as f:
        f.write(report)
    print(f"Health report written to {filepath}")
    # Also print to stdout for cron log
    print(report)


if __name__ == "__main__":
    main()
