#!/usr/bin/env bash
# Install AgentSSOT maintenance cron jobs.
#
# Usage:
#   SSOT_API_KEY=ssot_... ./scripts/install-cron.sh
#
# Installs:
#   - Health check every 5 minutes
#   - Full maintenance (backfill + dedup scan) daily at 04:00 UTC

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SSOT_API_KEY="${SSOT_API_KEY:-}"
SSOT_URL="${SSOT_URL:-http://localhost:8088}"
SSOT_NAMESPACE="${SSOT_NAMESPACE:-default}"
LOG_FILE="${SSOT_LOG_FILE:-/var/log/agentssot-maintenance.log}"

if [ -z "$SSOT_API_KEY" ]; then
  echo "ERROR: SSOT_API_KEY is required. Export it first."
  exit 1
fi

# Ensure log file is writable
touch "$LOG_FILE" 2>/dev/null || {
  echo "WARN: Cannot write to $LOG_FILE, using /tmp/agentssot-maintenance.log"
  LOG_FILE="/tmp/agentssot-maintenance.log"
}

ENV_LINE="SSOT_URL=${SSOT_URL} SSOT_API_KEY=${SSOT_API_KEY} SSOT_NAMESPACE=${SSOT_NAMESPACE} SSOT_LOG_FILE=${LOG_FILE}"

HEALTH_CRON="*/5 * * * * ${ENV_LINE} ${SCRIPT_DIR}/maintenance.sh --health-only >/dev/null 2>&1"
FULL_CRON="0 4 * * * ${ENV_LINE} ${SCRIPT_DIR}/maintenance.sh >/dev/null 2>&1"

# Remove any existing agentssot cron entries, then add new ones
(crontab -l 2>/dev/null | grep -v 'agentssot' | grep -v 'maintenance.sh' || true; echo "$HEALTH_CRON"; echo "$FULL_CRON") | crontab -

echo "Cron jobs installed:"
echo "  - Health check: every 5 minutes"
echo "  - Full maintenance: daily at 04:00 UTC"
echo "  - Log file: ${LOG_FILE}"
echo ""
echo "Verify with: crontab -l"
