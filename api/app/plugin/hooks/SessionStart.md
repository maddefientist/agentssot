---
name: SessionStart
description: Inject lightweight hive context hint on session start
enabled: true
---

# AgentSSOT Session Start

```bash
#!/bin/bash
PROJECT_NAME=$(basename "$(pwd)")

echo "<hive-available>"
echo "You have access to AgentSSOT via MCP tools."
echo "Use hive_recall only when prior cross-session context can materially help."
echo "Use hive_query for text/tag search. Use hive_ingest to store knowledge."
echo "If synthesis is enabled, concepts may also surface in recall."
echo "Current project: ${PROJECT_NAME}. Use the namespace configured for this agent."
echo "Skip Hive for self-contained work or when the repo/runtime/web is authoritative."
echo "Fetch context on demand — do NOT pre-load everything."
echo "</hive-available>"
```
