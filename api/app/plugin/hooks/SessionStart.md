---
name: SessionStart
description: Inject lightweight hive context hint on session start
enabled: true
---

# hari-hive Session Start

```bash
#!/bin/bash
PROJECT_NAME=$(basename "$(pwd)")

echo "<hive-available>"
echo "You have access to hari-hive (AgentSSOT) via MCP tools."
echo "Use hive_recall for semantic search (blends knowledge + synthesized concepts by default)."
echo "Use hive_query for text/tag search. Use hive_ingest to store knowledge."
echo "Concepts are long-term patterns extracted from your knowledge base — they surface automatically in recall."
echo "Current project: ${PROJECT_NAME}. Default namespace: claude-shared."
echo "Fetch context on demand — do NOT pre-load everything."
echo "</hive-available>"
```
