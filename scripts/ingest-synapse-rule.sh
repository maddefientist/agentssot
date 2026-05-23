#!/usr/bin/env bash
# ingest-synapse-rule.sh — idempotently insert the synapse fleet-awareness rule KI.
#
# Uses source="synapse-phase4-rule-v1" as a stable dedup key.
# Checks /query for an existing item with that source text before inserting.
# Safe to run multiple times — only inserts once.
#
# Usage:
#   bash /opt/agentssot/scripts/ingest-synapse-rule.sh
set -euo pipefail

CFG="${HIVE_AGENT_JSON:-${HOME}/.claude/agentssot/local/agent.json}"
if [ ! -f "$CFG" ]; then
    echo "ERROR: agent.json not found at $CFG" >&2
    exit 1
fi

BASE_URL=$(python3 -c "import json; d=json.load(open('$CFG')); print(d.get('base_url','http://192.168.1.225:8088').rstrip('/'))")
API_KEY=$(python3 -c "import json; d=json.load(open('$CFG')); print(d.get('api_key') or d.get('admin_api_key',''))")
NS="claude-shared"
SOURCE_KEY="synapse-phase4-rule-v1"

RULE_CONTENT="Synapse layer tracks every active Claude session across the fleet. Before editing shared files call synapse_collisions(file=...) to detect concurrent edits. To see what other sessions are doing call synapse_active(). Local snapshot at ~/.claude/synapse/active_fleet.json."

# ── Dedup check: search for our stable source key ─────────────────────────────
EXISTING=$(curl -s \
    -H "X-API-Key: $API_KEY" \
    "${BASE_URL}/query?q=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$SOURCE_KEY'))")&namespace=${NS}&limit=3" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('total',0))")

if [ "${EXISTING:-0}" -gt 0 ]; then
    echo "[synapse-rule] Rule already exists (found $EXISTING item(s) matching '$SOURCE_KEY') — skipping insert."
    exit 0
fi

# ── Insert ────────────────────────────────────────────────────────────────────
PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'namespace': '${NS}',
    'content': '''${RULE_CONTENT}''',
    'abstract': '''${RULE_CONTENT}''',
    'source': '${SOURCE_KEY}',
    'tags': ['synapse', 'fleet', 'collision-detection', 'rule'],
    'memory_type': 'rule',
    'loadout_priority': 5,
    'verbatim': True,
}))
")

RESULT=$(curl -s -w "\n%{http_code}" \
    -X POST \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    "${BASE_URL}/api/v1/knowledge/ingest")

STATUS=$(echo "$RESULT" | tail -1)
BODY=$(echo "$RESULT" | head -1)

if [ "$STATUS" = "200" ]; then
    ITEM_ID=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin).get('id','?'))")
    echo "[synapse-rule] Inserted rule KI (id=$ITEM_ID, source=$SOURCE_KEY)"
else
    echo "[synapse-rule] ERROR inserting rule: HTTP $STATUS — $BODY" >&2
    exit 1
fi
