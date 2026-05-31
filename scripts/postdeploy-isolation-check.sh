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
# Requires:
#   SSOT_ADMIN_KEY   admin API key (or SSOT_TEST_ADMIN_KEY). The bootstrap admin
#                    key is printed once to the API logs at first start:
#                      docker logs agentssot-api 2>&1 | grep BOOTSTRAP_ADMIN_API_KEY
#                    Store it in hari's environment / a secret manager.
# Optional:
#   SSOT_URL         default http://localhost:8088
#
# Usage:
#   SSOT_ADMIN_KEY=ssot_... ./scripts/postdeploy-isolation-check.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
URL="${SSOT_URL:-http://localhost:8088}"
ADMIN_KEY="${SSOT_ADMIN_KEY:-${SSOT_TEST_ADMIN_KEY:-}}"

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
