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
echo "Use hive_recall for semantic search, hive_query for text/tag search."
echo "Use hive_ingest to store knowledge. Use hive_stats for namespace info."
echo "Current project: ${PROJECT_NAME}. Default namespace: claude-shared."
echo "Fetch context on demand — do NOT pre-load everything."
echo "</hive-available>"
```
