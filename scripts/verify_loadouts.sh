#!/usr/bin/env bash
# verify_loadouts.sh — For every enrolled device, run a minimal
# hive_loadout curl from that device against the central API.
# Confirms plugin + agent.json + connectivity. Read-only — no writes.
#
# Usage:
#   ./verify_loadouts.sh
#
# This script SSHes into each enrolled device and runs curl against
# the central hive API. It does NOT modify anything on remote hosts.
set -u

HOSTS_JSON="$HOME/.claude/agentssot/hosts.json"
API_BASE="http://192.168.1.225:8088"

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

if [[ ! -f "$HOSTS_JSON" ]]; then
    echo -e "${RED}ERROR: missing $HOSTS_JSON${NC}" >&2
    exit 1
fi

# ── Resolve enrolled devices ────────────────────────────────────────
DEVICES=$(
    python3 -c "
import json, sys
hosts = json.load(open('$HOSTS_JSON'))
for d in hosts.get('devices', []):
    if d.get('ssh') is False:
        continue
    alias = d.get('ssh_alias') or d.get('host') or d.get('name')
    if alias:
        print(alias)
" 2>/dev/null
)

if [[ -z "$DEVICES" ]]; then
    echo -e "${YELLOW}WARNING: no enrolled devices found in $HOSTS_JSON${NC}" >&2
    exit 0
fi

echo "Hive API: $API_BASE"
echo "Devices:  $(echo $DEVICES | tr '\n' ' ')"
echo ""

PASS=0
FAIL=0
SKIP=0

# ── Remote test script (heredoc avoids quoting hell) ────────────────
# This runs on each remote device. It reads agent.json for the API key
# and does a single curl against the loadout endpoint. No writes.
REMOTE_SCRIPT=$(cat << 'REMOTE_EOF'
#!/bin/bash
set -u
API_BASE="__API_BASE__"
AGENT="$HOME/.claude/agentssot/local/agent.json"

if [ ! -r "$AGENT" ]; then
    echo "MISSING: no agent.json"
    exit 0
fi

KEY=$(python3 -c "import json; print(json.load(open('$AGENT'))['api_key'])" 2>/dev/null)
if [ -z "$KEY" ]; then
    echo "MISSING: no api_key in agent.json"
    exit 0
fi

DEVNAME=$(hostname -s 2>/dev/null || echo unknown)
BODY="{\"cwd\":\"/home/$USER\",\"device_id\":\"$DEVNAME\",\"namespace\":\"claude-shared\",\"token_budget\":300}"

RESP=$(curl -sS --max-time 3 \
    -H "X-Api-Key: $KEY" \
    -H "Content-Type: application/json" \
    --data "$BODY" \
    "$API_BASE/api/v1/knowledge/loadout" 2>/dev/null)

if [ -z "$RESP" ]; then
    echo "FAIL: no response from API"
    exit 0
fi

python3 -c "
import json, sys
try:
    b = json.load(sys.stdin)
    tokens = b.get('tokens_used', 0)
    overflow = b.get('overflow_count', 0)
    items_count = sum(len(v) for v in b.get('items', {}).values()) if b.get('items') else 0
    print(f'OK: tokens={tokens} overflow={overflow} items={items_count}')
except Exception as e:
    print(f'FAIL: {e}')
" <<< "$RESP" 2>/dev/null || echo "FAIL: could not parse response"
REMOTE_EOF
)

for D in $DEVICES; do
    echo -e "${BOLD}=== $D ===${NC}"

    # Check reachability first
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$D" true 2>/dev/null; then
        echo -e "  ${YELLOW}[skip] unreachable or no SSH key${NC}"
        ((SKIP++)) || true
        echo ""
        continue
    fi

    # Inject API base into the remote script template and pipe it via SSH
    SCRIPT_CONTENT="${REMOTE_SCRIPT/__API_BASE__/$API_BASE}"
    RESULT=$(echo "$SCRIPT_CONTENT" | ssh -o ConnectTimeout=5 "$D" 'bash -s' 2>/dev/null) || RESULT="FAIL: SSH exec error"

    if echo "$RESULT" | grep -q "^OK:"; then
        echo -e "  ${GREEN}${RESULT}${NC}"
        ((PASS++)) || true
    elif echo "$RESULT" | grep -q "^MISSING:"; then
        echo -e "  ${YELLOW}${RESULT}${NC}"
        ((SKIP++)) || true
    else
        echo -e "  ${RED}${RESULT}${NC}"
        ((FAIL++)) || true
    fi
    echo ""
done

# ── Summary ──────────────────────────────────────────────────────────
TOTAL=$((PASS + FAIL + SKIP))
echo "=========================================="
echo -e "  ${GREEN}Pass: ${PASS}${NC}  ${RED}Fail: ${FAIL}${NC}  ${YELLOW}Skip: ${SKIP}${NC}  Total: ${TOTAL}"
echo "=========================================="

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
