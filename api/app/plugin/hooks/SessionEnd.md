---
name: SessionEnd
description: Prompt fact extraction and ingest on session end
enabled: true
---

# hari-hive Session End

```bash
#!/bin/bash
PROJECT_NAME=$(basename "$(pwd)")
DEVICE_NAME=$(hostname -s 2>/dev/null || echo "unknown")

echo "<hive-session-end>"
echo "Before ending, extract 3-5 key facts from this session and ingest them."
echo "Use hive_ingest with tags: [\"session-extract\", \"device-${DEVICE_NAME}\", \"${PROJECT_NAME}\"]"
echo "Focus on: decisions made, bugs fixed, patterns learned, architecture changes."
echo "Skip: routine file reads, obvious actions, session-specific details."
echo "</hive-session-end>"
```
