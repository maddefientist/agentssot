#!/usr/bin/env bash
# rollout_plugin.sh — Push the local hari-hive plugin to every enrolled device
# listed in ~/.claude/agentssot/hosts.json. Verify the plugin loads and
# hive_loadout works on each.
#
# Usage:
#   ./rollout_plugin.sh              # live push
#   ./rollout_plugin.sh --dry-run    # show what WOULD change (no modifications)
#
# IMPORTANT: rsync is used WITHOUT --delete — ever. The hive has strong
# feedback (cross-llm tag) that rsync --delete caused a prod wipe on
# magicobj/objserver. Stale files in the remote plugin dir are tolerable;
# data loss is not.
set -euo pipefail

SOURCE="$HOME/.claude/plugins/hari-hive"
HOSTS_JSON="$HOME/.claude/agentssot/hosts.json"
DRY_RUN=false

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Parse args ──────────────────────────────────────────────────────
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}=== DRY RUN MODE — no changes will be made ===${NC}"
    echo ""
fi

# ── Pre-flight ──────────────────────────────────────────────────────
if [[ ! -d "$SOURCE" ]]; then
    echo -e "${RED}ERROR: missing $SOURCE${NC}" >&2
    exit 1
fi
if [[ ! -f "$HOSTS_JSON" ]]; then
    echo -e "${RED}ERROR: missing $HOSTS_JSON${NC}" >&2
    exit 1
fi

# ── Resolve enrolled devices ────────────────────────────────────────
# Devices with ssh: false are excluded. Use ssh_alias if present,
# otherwise fall back to host field.
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

echo "Source:  $SOURCE"
echo "Devices: $(echo $DEVICES | tr '\n' ' ')"
echo ""

# ── Per-device push ──────────────────────────────────────────────────
for D in $DEVICES; do
    echo -e "${BOLD}=== $D ===${NC}"

    # Check reachability
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$D" true 2>/dev/null; then
        echo "  [skip] unreachable or no SSH key configured"
        echo ""
        continue
    fi

    # Resolve remote home
    REMOTE_HOME=$(ssh -o ConnectTimeout=5 "$D" 'echo "$HOME"' 2>/dev/null) || {
        echo "  [skip] could not resolve remote \$HOME"
        echo ""
        continue
    }

    REMOTE_DIR="$REMOTE_HOME/.claude/plugins/hari-hive"

    if $DRY_RUN; then
        # ── Dry-run: show what rsync WOULD change ────────────────────
        echo -e "  ${CYAN}[dry-run] rsync -az --dry-run (NO --delete)${NC}"
        echo "    from: $SOURCE/"
        echo "    to:   $D:$REMOTE_DIR/"
        echo ""
        rsync -az --dry-run \
            --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
            "$SOURCE/" "$D:$REMOTE_DIR/" 2>&1 | sed 's/^/    /' || true
        echo ""
        echo "  [dry-run] would verify plugin syntax via python3 ast.parse"
    else
        # ── Live push: rsync WITHOUT --delete ────────────────────────
        echo "  Pushing plugin..."
        # NOTE: NO --delete. Stale remote files are preferred over
        # accidental data loss. See hive feedback: rsync --delete caused
        # a prod wipe on magicobj/objserver (2026-04-17).
        rsync -az \
            --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
            "$SOURCE/" "$D:$REMOTE_DIR/"

        echo "  Verifying plugin syntax..."
        if ssh -o ConnectTimeout=5 "$D" "cd '$REMOTE_DIR' && python3 -c 'import ast; ast.parse(open(\"mcp_server.py\").read())' && echo '  plugin syntax OK'" 2>/dev/null; then
            : # verified
        else
            echo -e "  ${YELLOW}[warn] syntax check failed on $D${NC}"
        fi
    fi
    echo ""
done

# ── Summary ──────────────────────────────────────────────────────────
if $DRY_RUN; then
    echo -e "${YELLOW}=== DRY RUN COMPLETE — no changes were made ===${NC}"
else
    echo -e "${GREEN}=== ROLLOUT COMPLETE ===${NC}"
    echo "Run scripts/verify_loadouts.sh to confirm plugin loads on each device."
fi
