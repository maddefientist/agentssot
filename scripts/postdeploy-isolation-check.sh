#!/usr/bin/env bash
# postdeploy-isolation-check.sh — namespace-isolation deploy gate (runs ON hari).
#
# Runs the isolation benchmark against the freshly-deployed API and FAILS
# (non-zero exit) if any cross-tenant leak is detected. Wire this into the deploy
# flow right after `docker compose up -d` so a regression in namespace RBAC can
# never reach a healthy/"deployed" state.
#
# It lives where the backend + admin key already are (hari) — it is NOT meant to
# run inside the MBA Fastlane CI, which has no backend to talk to.
#
# Admin key resolution (first hit wins):
#   1. SSOT_ADMIN_KEY env var
#   2. SSOT_TEST_ADMIN_KEY env var
#   3. SSOT_ADMIN_KEY_FILE — a JSON file with an "admin_api_key" (or "api_key")
#      field. Defaults to ~/.claude/agentssot/local/admin.json (the hive MCP
#      plugin's admin credential). Override with the env var on other hosts.
#   If none resolve, the gate is SKIPPED (exit 2) — never a silent pass.
# Optional:
#   SSOT_URL         default http://localhost:8088
#
# Usage:
#   ./scripts/postdeploy-isolation-check.sh                 # auto-reads admin.json
#   SSOT_ADMIN_KEY=ssot_... ./scripts/postdeploy-isolation-check.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
URL="${SSOT_URL:-http://localhost:8088}"
ADMIN_KEY="${SSOT_ADMIN_KEY:-${SSOT_TEST_ADMIN_KEY:-}}"

# Fallback: read the key from a JSON credential file (hive MCP plugin convention).
ADMIN_KEY_FILE="${SSOT_ADMIN_KEY_FILE:-$HOME/.claude/agentssot/local/admin.json}"
if [[ -z "$ADMIN_KEY" && -f "$ADMIN_KEY_FILE" ]]; then
  ADMIN_KEY="$(python3 -c "import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(d.get('admin_api_key') or d.get('api_key') or '')
except Exception:
    print('')" "$ADMIN_KEY_FILE" 2>/dev/null || true)"
  [[ -n "$ADMIN_KEY" ]] && echo "  (admin key sourced from $ADMIN_KEY_FILE)"
fi

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

if [[ -z "$ADMIN_KEY" ]]; then
  echo -e "${RED}✗ SSOT_ADMIN_KEY not set — cannot run isolation gate.${NC}" >&2
  echo "  Recover it: docker logs agentssot-api 2>&1 | grep BOOTSTRAP_ADMIN_API_KEY" >&2
  exit 2
fi

# 1. Wait for the API to be healthy (max ~60s).
echo "Waiting for ${URL}/health ..."
for i in $(seq 1 20); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "${URL}/health" || true)"
  [[ "$code" == "200" ]] && { echo -e "${GREEN}✓ healthy${NC}"; break; }
  [[ "$i" == "20" ]] && { echo -e "${RED}✗ API never became healthy${NC}" >&2; exit 3; }
  sleep 3
done

# 2. Run the isolation gate (fixed probes + cached keys → no cruft).
echo "Running namespace-isolation gate..."
cd "$REPO_DIR"
set +e
SSOT_TEST_URL="$URL" SSOT_TEST_ADMIN_KEY="$ADMIN_KEY" \
  python benchmarks/isolation/runner.py --ci
rc=$?
set -e

if [[ "$rc" -eq 0 ]]; then
  echo -e "${GREEN}✓ ISOLATION GATE PASSED — 100% tenant isolation.${NC}"
else
  echo -e "${RED}✗ ISOLATION GATE FAILED — cross-tenant leak detected. Blocking deploy.${NC}" >&2
  echo -e "${YELLOW}  See benchmarks/isolation/results/isolation_report.md${NC}" >&2
fi
exit "$rc"
