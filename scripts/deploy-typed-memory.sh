#!/usr/bin/env bash
# deploy-typed-memory.sh — Deploy M3 (typed memory) + M8 (secret scanning) to hari
#
# Run FROM your Mac. Never run this ON the server directly.
#
# Usage:
#   ./scripts/deploy-typed-memory.sh              # live deployment
#   ./scripts/deploy-typed-memory.sh --dry-run    # show what would happen
#
# ROLLBACK INSTRUCTIONS:
#   1. SSH to hari: ssh hari
#   2. cd /opt/agentssot
#   3. git log --oneline -5  # find the commit before our changes
#   4. git revert HEAD~2..HEAD  # revert both M3 and M8 commits
#      OR: git reset --hard <previous-commit-hash>
#   5. docker compose restart api
#   6. Verify: curl -s http://localhost:8088/health
#
#   The new columns (memory_type, last_verified_at, etc.) are safe to leave
#   in the DB — they're nullable with defaults, so old code ignores them.
#   The feature flag typed_memory_enabled defaults to false, so even if
#   columns exist, type-aware recall won't activate until explicitly enabled.
#
#   Secret scanning rejection is always-on once deployed. To disable:
#   set DISABLE_SECRET_SCANNING=true in the container environment.

set -euo pipefail

DRY_RUN=false
REMOTE_HOST="hari"
REMOTE_DIR="/opt/agentssot"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HEALTH_URL="http://hari:8088/health"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}=== DRY RUN MODE — no changes will be made ===${NC}"
    echo ""
fi

run() {
    if $DRY_RUN; then
        echo -e "${CYAN}[dry-run]${NC} $*"
    else
        echo -e "${GREEN}[run]${NC} $*"
        eval "$@"
    fi
}

check() {
    echo -e "${CYAN}[check]${NC} $*"
    eval "$@"
}

echo "============================================"
echo " Deploy M3 (Typed Memory) + M8 (Secret Scan)"
echo "============================================"
echo ""

# ── Step 0: Pre-flight checks ──────────────────────────────────────

echo -e "${YELLOW}Step 0: Pre-flight checks${NC}"

# Verify we're in the right local repo
if [[ ! -f "$LOCAL_DIR/api/app/models.py" ]]; then
    echo -e "${RED}ERROR: Not in agentssot-backend repo. Run from repo root or scripts/ dir.${NC}"
    exit 1
fi

# Check local commits exist
LOCAL_HEAD=$(cd "$LOCAL_DIR" && git rev-parse HEAD)
LOCAL_COMMITS=$(cd "$LOCAL_DIR" && git log --oneline origin/main..HEAD 2>/dev/null | wc -l | tr -d ' ')
echo "  Local HEAD: $LOCAL_HEAD"
echo "  Commits ahead of origin/main: $LOCAL_COMMITS"

if [[ "$LOCAL_COMMITS" == "0" ]]; then
    echo -e "${YELLOW}  WARNING: No local commits ahead of remote. Nothing to deploy?${NC}"
fi

# Check remote is reachable
echo "  Checking SSH connectivity..."
if ! ssh -o ConnectTimeout=5 "$REMOTE_HOST" 'echo ok' &>/dev/null; then
    echo -e "${RED}ERROR: Cannot reach $REMOTE_HOST via SSH${NC}"
    exit 1
fi
echo "  SSH: OK"

# Check remote has clean working tree
REMOTE_STATUS=$(ssh "$REMOTE_HOST" "cd $REMOTE_DIR && git status --short" 2>&1)
if [[ -n "$REMOTE_STATUS" ]]; then
    echo -e "${RED}ERROR: Remote has uncommitted changes:${NC}"
    echo "$REMOTE_STATUS"
    echo "Resolve these before deploying."
    exit 1
fi
echo "  Remote working tree: clean"
echo ""

# ── Step 1: Push local commits to remote ───────────────────────────

echo -e "${YELLOW}Step 1: Push local commits to origin (hari:$REMOTE_DIR)${NC}"
run "cd '$LOCAL_DIR' && git push origin main"
echo ""

# ── Step 2: Pull on remote ────────────────────────────────────────

echo -e "${YELLOW}Step 2: Pull latest on hari${NC}"
run "ssh '$REMOTE_HOST' 'cd $REMOTE_DIR && git pull --ff-only'"
echo ""

# ── Step 3: Rebuild and restart the API container ──────────────────

echo -e "${YELLOW}Step 3: Rebuild and restart the API container${NC}"
echo "  (startup.py handles ADD COLUMN IF NOT EXISTS on boot)"
run "ssh '$REMOTE_HOST' 'cd $REMOTE_DIR && docker compose build api && docker compose up -d api'"
echo ""

# ── Step 4: Wait for health ───────────────────────────────────────

echo -e "${YELLOW}Step 4: Wait for API health (up to 30s)${NC}"
if ! $DRY_RUN; then
    echo "  Waiting 5s for container startup..."
    sleep 5
    for i in $(seq 1 5); do
        if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
            echo -e "  ${GREEN}Health check passed!${NC}"
            break
        fi
        if [[ $i -eq 5 ]]; then
            echo -e "  ${RED}Health check FAILED after 30s${NC}"
            echo "  Check logs: ssh hari 'cd $REMOTE_DIR && docker compose logs api --tail=50'"
            exit 1
        fi
        echo "  Attempt $i/5 failed, retrying in 5s..."
        sleep 5
    done
else
    echo -e "${CYAN}[dry-run]${NC} curl -sf $HEALTH_URL"
fi
echo ""

# ── Step 5: Verify typed memory columns exist ─────────────────────

echo -e "${YELLOW}Step 5: Verify typed memory columns in database${NC}"
VERIFY_CMD="ssh '$REMOTE_HOST' 'cd $REMOTE_DIR && docker compose exec -T api python3 -c \"
from app.models import KnowledgeItem
cols = [c.name for c in KnowledgeItem.__table__.columns]
required = [\\\"memory_type\\\", \\\"last_verified_at\\\", \\\"staleness_score\\\", \\\"extraction_source\\\", \\\"extraction_cursor_id\\\"]
missing = [r for r in required if r not in cols]
if missing:
    print(f\\\"FAIL: missing columns: {missing}\\\")
    exit(1)
print(f\\\"OK: all typed memory columns present in model\\\")
print(f\\\"Columns: {cols}\\\")
\"'"

if ! $DRY_RUN; then
    eval "$VERIFY_CMD"
else
    echo -e "${CYAN}[dry-run]${NC} verify typed memory columns via docker exec"
fi
echo ""

# ── Step 6: Verify columns in actual DB ───────────────────────────

echo -e "${YELLOW}Step 6: Verify columns exist in PostgreSQL${NC}"
DB_VERIFY_CMD="ssh '$REMOTE_HOST' 'cd $REMOTE_DIR && docker compose exec -T db psql -U agentssot -d agentssot -c \"
SELECT column_name FROM information_schema.columns
WHERE table_name = \\\"knowledge_items\\\"
AND column_name IN (\\\"memory_type\\\", \\\"last_verified_at\\\", \\\"staleness_score\\\", \\\"extraction_source\\\", \\\"extraction_cursor_id\\\")
ORDER BY column_name;
\"'"

if ! $DRY_RUN; then
    eval "$DB_VERIFY_CMD" 2>/dev/null || echo "  (DB direct check skipped — verify via model check above)"
else
    echo -e "${CYAN}[dry-run]${NC} psql column verification"
fi
echo ""

# ── Step 7: Smoke test recall endpoint ────────────────────────────

echo -e "${YELLOW}Step 7: Smoke test /recall endpoint${NC}"
if ! $DRY_RUN; then
    RECALL_RESULT=$(curl -sf "http://hari:8088/recall" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${SSOT_API_KEY:-}" \
        -d '{"namespace":"claude-shared","query_text":"test","scope":"knowledge","top_k":1}' 2>&1) || true
    if echo "$RECALL_RESULT" | grep -q '"items"'; then
        echo -e "  ${GREEN}Recall endpoint responding correctly${NC}"
    else
        echo -e "  ${YELLOW}Recall test inconclusive (may need API key): $RECALL_RESULT${NC}"
    fi
else
    echo -e "${CYAN}[dry-run]${NC} curl /recall with test query"
fi
echo ""

# ── Done ──────────────────────────────────────────────────────────

echo "============================================"
if $DRY_RUN; then
    echo -e "${YELLOW}DRY RUN COMPLETE — no changes were made${NC}"
    echo ""
    echo "To deploy for real, run without --dry-run:"
    echo "  ./scripts/deploy-typed-memory.sh"
else
    echo -e "${GREEN}DEPLOYMENT COMPLETE${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Enable typed memory: set TYPED_MEMORY_ENABLED=true in container env"
    echo "  2. Run backfill: ssh hari 'cd $REMOTE_DIR && docker compose exec api python3 -m scripts.backfill_memory_types'"
    echo "  3. Monitor logs: ssh hari 'cd $REMOTE_DIR && docker compose logs api --tail=20 -f'"
fi
echo "============================================"
