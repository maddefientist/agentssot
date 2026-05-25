#!/usr/bin/env bash
# ingest-synapse-rule-v2.sh — idempotently upsert the synapse fleet-awareness rule KI (v2).
#
# What changed from v1:
#   - Teaches agents about the disabled_local status and the enable procedure.
#   - Introduces synapse_status() verb for self-diagnosis.
#
# Strategy:
#   1. Check for existing v2 item by title (= source key). Skip insert if found.
#      Note: /query results include 'title' (derived from source) but not 'source',
#      so we match on title equality.
#   2. Insert v2 rule.
#   3. Find v1 item by title (source=synapse-phase4-rule-v1) and supersede + soft-expire it.
#
# Idempotent: running twice skips insert and re-applies supersession (no-op on repeat).
#
# Usage:
#   bash /opt/agentssot/scripts/ingest-synapse-rule-v2.sh
set -euo pipefail

CFG="${HIVE_AGENT_JSON:-${HOME}/.claude/agentssot/local/agent.json}"
if [ ! -f "$CFG" ]; then
    echo "ERROR: agent.json not found at $CFG" >&2
    exit 1
fi

BASE_URL=$(python3 -c "import json; d=json.load(open('$CFG')); print(d.get('base_url','http://192.168.1.225:8088').rstrip('/'))")
API_KEY=$(python3 -c "import json; d=json.load(open('$CFG')); print(d.get('api_key') or d.get('admin_api_key',''))")
NS="claude-shared"
SOURCE_V1="synapse-phase4-rule-v1"
SOURCE_V2="synapse-phase4-rule-v2"

RULE_CONTENT='Synapse tracks active Claude sessions fleet-wide. Call synapse_active() to see what others are doing; synapse_collisions(file=...) before editing shared files. If either returns {"status":"disabled_local"}, synapse is dormant — tell operator: set "synapse_enabled":true in ~/.claude/agentssot/local/agent.json. Call synapse_status() to diagnose.'

# ── Helper: search for items by title (source key), return first matching id or "" ──
# /query results include 'title' (set from source on ingest) but not 'source' field.
find_by_title() {
    local title="$1"
    local encoded
    encoded=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$title'))")
    python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
results = d.get('results', [])
for r in results:
    if r.get('title') == '$title':
        print(r.get('id', ''))
        break
" <<< "$(curl -s \
    -H "X-API-Key: $API_KEY" \
    "${BASE_URL}/query?q=${encoded}&namespace=${NS}&limit=5")"
}

# ── Helper: delete items by title (for cleanup of accidental duplicates) ──────
delete_extras_by_title() {
    local title="$1"
    local keep_id="$2"
    local encoded
    encoded=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$title'))")
    local ids
    ids=$(python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
results = d.get('results', [])
ids = [r.get('id','') for r in results if r.get('title') == '$title' and r.get('id') != '$keep_id']
print('\n'.join(ids))
" <<< "$(curl -s \
    -H "X-API-Key: $API_KEY" \
    "${BASE_URL}/query?q=${encoded}&namespace=${NS}&limit=20")")

    if [ -z "$ids" ]; then
        return 0
    fi

    while IFS= read -r dup_id; do
        [ -z "$dup_id" ] && continue
        ADMIN_KEY=$(python3 -c "
import json, os
admin_path = os.path.expanduser('~/.claude/agentssot/local/admin.json')
try:
    print(json.load(open(admin_path))['admin_api_key'])
except:
    print('$API_KEY')
")
        curl -s -X POST \
            -H "X-API-Key: $ADMIN_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"namespace\": \"$NS\", \"ids\": [\"$dup_id\"]}" \
            "${BASE_URL}/admin/delete-items" > /dev/null
        echo "[synapse-rule-v2] Deleted duplicate v2 item (id=$dup_id)."
    done <<< "$ids"
}

# ── Step 1: Check if v2 already exists ────────────────────────────────────────
V2_ID=$(find_by_title "$SOURCE_V2" || true)

if [ -n "$V2_ID" ]; then
    echo "[synapse-rule-v2] v2 rule already exists (id=$V2_ID) — skipping insert."
    # Clean up any accidental duplicates from previous runs
    delete_extras_by_title "$SOURCE_V2" "$V2_ID"
else
    # ── Step 2: Insert v2 rule ─────────────────────────────────────────────────
    PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'namespace': '${NS}',
    'content': '''${RULE_CONTENT}''',
    'abstract': '''${RULE_CONTENT}''',
    'source': '${SOURCE_V2}',
    'tags': ['synapse', 'fleet', 'collision-detection', 'rule', 'onboarding'],
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
        V2_ID=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin).get('id','?'))")
        echo "[synapse-rule-v2] Inserted v2 rule KI (id=$V2_ID, source=$SOURCE_V2)"
    else
        echo "[synapse-rule-v2] ERROR inserting v2 rule: HTTP $STATUS — $BODY" >&2
        exit 1
    fi
fi

# ── Step 3: Find v1 and supersede + expire it ─────────────────────────────────
V1_ID=$(find_by_title "$SOURCE_V1" || true)

if [ -z "$V1_ID" ]; then
    echo "[synapse-rule-v2] v1 rule not found (already removed or never existed) — nothing to supersede."
else
    echo "[synapse-rule-v2] Found v1 rule (id=$V1_ID) — superseding with v2 (id=$V2_ID)."

    # Mark v1 as superseded by v2
    SUP_RESULT=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "X-API-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"superseded_by\": \"$V2_ID\"}" \
        "${BASE_URL}/api/v1/knowledge/items/${V1_ID}/supersede")
    SUP_STATUS=$(echo "$SUP_RESULT" | tail -1)

    if [ "$SUP_STATUS" = "200" ]; then
        echo "[synapse-rule-v2] v1 superseded successfully."
    else
        echo "[synapse-rule-v2] WARNING: supersede returned HTTP $SUP_STATUS — continuing to expire anyway."
    fi

    # Soft-expire v1 so it drops out of loadout immediately
    EXP_RESULT=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "X-API-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"reason\": \"superseded by synapse-phase4-rule-v2 ($V2_ID)\"}" \
        "${BASE_URL}/api/v1/knowledge/items/${V1_ID}/expire")
    EXP_STATUS=$(echo "$EXP_RESULT" | tail -1)

    if [ "$EXP_STATUS" = "200" ]; then
        echo "[synapse-rule-v2] v1 soft-expired — removed from future loadouts."
    else
        echo "[synapse-rule-v2] WARNING: expire returned HTTP $EXP_STATUS — v1 may still appear in loadout until natural expiry."
    fi
fi

echo "[synapse-rule-v2] Done. v2_id=$V2_ID"
