#!/usr/bin/env bash
# deploy.sh — Deploy the AgentSSOT API to hari.
#
# Run FROM your Mac (or any host with SSH to hari + push rights). Never run ON
# the server directly. Pushes local commits, pulls + rebuilds on hari, waits for
# health, then runs the namespace-isolation deploy gate.
#
# Usage:
#   ./scripts/deploy.sh                 # live deploy
#   ./scripts/deploy.sh --dry-run       # show what would happen
#
# Isolation gate:
#   The gate runs on hari and needs an admin API key. Provide SSOT_ADMIN_KEY in
#   your deploy environment (forwarded to hari) OR set it in hari's own env. If
#   neither is present the gate is SKIPPED with a warning — it never silently
#   passes. A detected cross-tenant leak FAILS the deploy.
#
# Rollback:
#   ssh hari 'cd /opt/agentssot && git reset --hard <prev-sha> && docker compose up -d --build api'

set -euo pipefail

DRY_RUN=false
REMOTE_HOST="${SSOT_REMOTE_HOST:-hari}"
REMOTE_DIR="${SSOT_REMOTE_DIR:-/opt/agentssot}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HEALTH_URL="${SSOT_HEALTH_URL:-http://hari:8088/health}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

[[ "${1:-}" == "--dry-run" ]] && { DRY_RUN=true; echo -e "${YELLOW}=== DRY RUN — no changes ===${NC}\n"; }

run() {
    if $DRY_RUN; then echo -e "${CYAN}[dry-run]${NC} $*"; else echo -e "${GREEN}[run]${NC} $*"; eval "$@"; fi
}

echo "========================================"
echo " Deploy AgentSSOT API → $REMOTE_HOST"
echo "========================================"
echo ""

# ── Step 0: Pre-flight ─────────────────────────────────────────────
echo -e "${YELLOW}Step 0: Pre-flight checks${NC}"
[[ -f "$LOCAL_DIR/api/app/models.py" ]] || { echo -e "${RED}ERROR: not the agentssot repo${NC}"; exit 1; }
LOCAL_COMMITS=$(cd "$LOCAL_DIR" && git log --oneline origin/main..HEAD 2>/dev/null | wc -l | tr -d ' ')
echo "  Local HEAD: $(cd "$LOCAL_DIR" && git rev-parse --short HEAD) (ahead of origin: $LOCAL_COMMITS)"
[[ "$LOCAL_COMMITS" == "0" ]] && echo -e "${YELLOW}  WARNING: nothing ahead of origin — redeploying current code${NC}"
ssh -o ConnectTimeout=5 "$REMOTE_HOST" 'echo ok' &>/dev/null || { echo -e "${RED}ERROR: cannot SSH to $REMOTE_HOST${NC}"; exit 1; }
REMOTE_STATUS=$(ssh "$REMOTE_HOST" "cd $REMOTE_DIR && git status --short" 2>&1)
if [[ -n "$REMOTE_STATUS" ]]; then
    echo -e "${RED}ERROR: remote working tree is dirty — resolve before deploying:${NC}"; echo "$REMOTE_STATUS"; exit 1
fi
echo "  SSH ok; remote tree clean"
echo ""

# ── Step 1-2: Push + pull ──────────────────────────────────────────
echo -e "${YELLOW}Step 1: Push to origin${NC}"
run "cd '$LOCAL_DIR' && git push origin main"
echo -e "${YELLOW}Step 2: Pull on $REMOTE_HOST${NC}"
run "ssh '$REMOTE_HOST' 'cd $REMOTE_DIR && git pull --ff-only'"
echo ""

# ── Step 3: Rebuild + restart ──────────────────────────────────────
echo -e "${YELLOW}Step 3: Rebuild + restart api (startup.py applies idempotent migrations)${NC}"
run "ssh '$REMOTE_HOST' 'cd $REMOTE_DIR && docker compose build api && docker compose up -d api'"
echo ""

# ── Step 4: Health ─────────────────────────────────────────────────
echo -e "${YELLOW}Step 4: Wait for API health${NC}"
if ! $DRY_RUN; then
    sleep 5
    for i in $(seq 1 6); do
        curl -sf "$HEALTH_URL" >/dev/null 2>&1 && { echo -e "  ${GREEN}healthy${NC}"; break; }
        [[ $i -eq 6 ]] && { echo -e "  ${RED}health FAILED${NC}; check: ssh $REMOTE_HOST 'cd $REMOTE_DIR && docker compose logs api --tail=50'"; exit 1; }
        echo "  attempt $i/6, retry in 5s..."; sleep 5
    done
else
    echo -e "${CYAN}[dry-run]${NC} curl -sf $HEALTH_URL"
fi
echo ""

# ── Step 5: Namespace-isolation gate ───────────────────────────────
echo -e "${YELLOW}Step 5: Namespace-isolation deploy gate${NC}"
if $DRY_RUN; then
    echo -e "${CYAN}[dry-run]${NC} ssh $REMOTE_HOST '... ./scripts/postdeploy-isolation-check.sh'"
else
    GATE_ENV=""
    [[ -n "${SSOT_ADMIN_KEY:-}" ]] && GATE_ENV="SSOT_ADMIN_KEY='${SSOT_ADMIN_KEY}' "
    set +e
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR && ${GATE_ENV}./scripts/postdeploy-isolation-check.sh"
    gate_rc=$?
    set -e
    case "$gate_rc" in
        0) echo -e "  ${GREEN}isolation gate passed${NC}" ;;
        2) echo -e "  ${YELLOW}SKIPPED: SSOT_ADMIN_KEY not set (deploy continues; gate not run)${NC}" ;;
        *) echo -e "  ${RED}ISOLATION GATE FAILED (rc=$gate_rc) — investigate before trusting this deploy${NC}"; exit "$gate_rc" ;;
    esac
fi
echo ""

echo "========================================"
$DRY_RUN && echo -e "${YELLOW}DRY RUN COMPLETE${NC}" || echo -e "${GREEN}DEPLOY COMPLETE${NC}"
echo "  Logs: ssh $REMOTE_HOST 'cd $REMOTE_DIR && docker compose logs api --tail=20'"
echo "========================================"
