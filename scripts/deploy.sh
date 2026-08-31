#!/usr/bin/env bash
# deploy.sh — Deploy the AgentSSOT API to an explicitly configured host.
#
# Run from a trusted workstation with SSH and repository push rights. Never run
# on the server directly. Pushes local commits, pulls + rebuilds remotely, waits for
# health, then runs the namespace-isolation deploy gate.
#
# Usage:
#   ./scripts/deploy.sh                 # live deploy
#   ./scripts/deploy.sh --dry-run       # show what would happen
#
# Isolation gate:
#   The gate runs remotely and needs an admin API key. Provide SSOT_ADMIN_KEY in
#   the deploy environment. A missing key or detected cross-tenant leak fails
#   the deploy; neither condition is treated as a pass.
#
# Rollback:
#   Roll back only to a revision that has already passed the P0 gateway,
#   enrollment, revocation, and namespace-isolation gates. Never roll back to a
#   known unauthenticated image.

set -euo pipefail

DRY_RUN=false
: "${SSOT_REMOTE_HOST:?Set SSOT_REMOTE_HOST to the explicit deployment host}"
: "${SSOT_HEALTH_URL:?Set SSOT_HEALTH_URL to the explicit health endpoint}"
REMOTE_HOST="$SSOT_REMOTE_HOST"
REMOTE_DIR="${SSOT_REMOTE_DIR:-/opt/agentssot}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HEALTH_URL="$SSOT_HEALTH_URL"

[[ "$REMOTE_HOST" =~ ^[A-Za-z0-9._@:-]+$ ]] || { echo "ERROR: invalid SSOT_REMOTE_HOST" >&2; exit 2; }
[[ "$REMOTE_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo "ERROR: invalid SSOT_REMOTE_DIR" >&2; exit 2; }
[[ "$HEALTH_URL" =~ ^https?:// ]] || { echo "ERROR: SSOT_HEALTH_URL must be http(s)" >&2; exit 2; }

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

[[ "${1:-}" == "--dry-run" ]] && { DRY_RUN=true; echo -e "${YELLOW}=== DRY RUN — no changes ===${NC}\n"; }

run() {
    if $DRY_RUN; then
        printf "%b" "${CYAN}[dry-run]${NC}"
        printf " %q" "$@"
        printf "\n"
    else
        echo -e "${GREEN}[run]${NC} $*"
        "$@"
    fi
}

echo "========================================"
echo " Deploy AgentSSOT API → $REMOTE_HOST"
echo "========================================"
echo ""

# ── Step 0: Pre-flight ─────────────────────────────────────────────
echo -e "${YELLOW}Step 0: Pre-flight checks${NC}"
[[ -f "$LOCAL_DIR/api/app/models.py" ]] || { echo -e "${RED}ERROR: not the agentssot repo${NC}"; exit 1; }
CURRENT_BRANCH=$(git -C "$LOCAL_DIR" branch --show-current)
[[ "$CURRENT_BRANCH" == "main" ]] || { echo -e "${RED}ERROR: deploy requires the accepted main branch (current: $CURRENT_BRANCH)${NC}"; exit 1; }
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
run git -C "$LOCAL_DIR" push origin main
echo -e "${YELLOW}Step 2: Pull on $REMOTE_HOST${NC}"
run ssh "$REMOTE_HOST" "cd '$REMOTE_DIR' && git pull --ff-only"
echo ""

# ── Step 3: Rebuild + restart ──────────────────────────────────────
echo -e "${YELLOW}Step 3: Rebuild + restart api (startup.py applies idempotent migrations)${NC}"
DEPLOY_SHA="$(cd "$LOCAL_DIR" && git rev-parse HEAD)"
run ssh "$REMOTE_HOST" "cd '$REMOTE_DIR' && GIT_SHA='$DEPLOY_SHA' docker compose build --build-arg GIT_SHA='$DEPLOY_SHA' api && GIT_SHA='$DEPLOY_SHA' docker compose up -d api"
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
    [[ -n "${SSOT_ADMIN_KEY:-}" ]] || { echo -e "  ${RED}ISOLATION GATE NOT RUN: SSOT_ADMIN_KEY is required${NC}"; exit 2; }
    set +e
    # Send the secret over SSH stdin; never interpolate it into a process
    # argument, dry-run output, or remote command line.
    printf '%s\n' "$SSOT_ADMIN_KEY" | ssh "$REMOTE_HOST" \
        "cd '$REMOTE_DIR' && IFS= read -r SSOT_ADMIN_KEY && export SSOT_ADMIN_KEY && ./scripts/postdeploy-isolation-check.sh"
    gate_rc=$?
    set -e
    case "$gate_rc" in
        0) echo -e "  ${GREEN}isolation gate passed${NC}" ;;
        2) echo -e "  ${RED}ISOLATION GATE NOT RUN: SSOT_ADMIN_KEY is required${NC}"; exit 2 ;;
        *) echo -e "  ${RED}ISOLATION GATE FAILED (rc=$gate_rc) — investigate before trusting this deploy${NC}"; exit "$gate_rc" ;;
    esac
fi
echo ""

echo "========================================"
$DRY_RUN && echo -e "${YELLOW}DRY RUN COMPLETE${NC}" || echo -e "${GREEN}DEPLOY COMPLETE${NC}"
echo "  Logs: ssh $REMOTE_HOST 'cd $REMOTE_DIR && docker compose logs api --tail=20'"
echo "========================================"
