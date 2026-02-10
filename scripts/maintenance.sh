#!/usr/bin/env bash
# AgentSSOT maintenance script
# Performs: health check, embedding backfill, dedup scan
#
# Usage:
#   ./scripts/maintenance.sh                    # Run all checks
#   ./scripts/maintenance.sh --health-only      # Health check only
#   ./scripts/maintenance.sh --backfill-only    # Backfill only
#   ./scripts/maintenance.sh --dedup-only       # Dedup dry-run only
#
# Environment variables:
#   SSOT_URL        Base URL (default: http://localhost:8088)
#   SSOT_API_KEY    Admin API key (required)
#   SSOT_NAMESPACE  Target namespace (default: default)
#   SSOT_LOG_FILE   Log file path (default: /var/log/agentssot-maintenance.log)

set -euo pipefail

SSOT_URL="${SSOT_URL:-http://localhost:8088}"
SSOT_API_KEY="${SSOT_API_KEY:-}"
SSOT_NAMESPACE="${SSOT_NAMESPACE:-default}"
LOG_FILE="${SSOT_LOG_FILE:-/var/log/agentssot-maintenance.log}"

log() {
  local level="$1"; shift
  local msg
  msg="$(date -u '+%Y-%m-%dT%H:%M:%SZ') [$level] $*"
  echo "$msg"
  echo "$msg" >> "$LOG_FILE" 2>/dev/null || true
}

check_key() {
  if [ -z "$SSOT_API_KEY" ]; then
    log ERROR "SSOT_API_KEY is not set. Export it or pass via environment."
    exit 1
  fi
}

health_check() {
  log INFO "health check: ${SSOT_URL}/health"
  local resp
  resp=$(curl -sf --max-time 10 "${SSOT_URL}/health" 2>&1) || {
    log ERROR "health check FAILED — API unreachable"
    return 1
  }

  local status_val
  status_val=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")

  if [ "$status_val" = "ok" ]; then
    log INFO "health check OK"
    return 0
  else
    log WARN "health check returned status=$status_val"
    return 1
  fi
}

run_backfill() {
  check_key
  log INFO "backfill embeddings: namespace=${SSOT_NAMESPACE}"

  for scope in knowledge requirements events; do
    local resp
    resp=$(curl -sf --max-time 300 \
      -X POST "${SSOT_URL}/admin/backfill-embeddings" \
      -H "X-API-Key: ${SSOT_API_KEY}" \
      -H "Content-Type: application/json" \
      -d "{\"namespace\":\"${SSOT_NAMESPACE}\",\"scope\":\"${scope}\",\"limit\":500}" 2>&1) || {
      log WARN "backfill ${scope} request failed"
      continue
    }
    local updated
    updated=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('updated',0))" 2>/dev/null || echo "?")
    log INFO "backfill ${scope}: updated=${updated}"
  done
}

run_dedup() {
  check_key
  log INFO "dedup scan: namespace=${SSOT_NAMESPACE} (dry_run=true)"

  local resp
  resp=$(curl -sf --max-time 60 \
    -X POST "${SSOT_URL}/admin/dedup" \
    -H "X-API-Key: ${SSOT_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"namespace\":\"${SSOT_NAMESPACE}\",\"dry_run\":true}" 2>&1) || {
    log WARN "dedup scan request failed"
    return 1
  }

  local groups deleted
  groups=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('duplicate_groups',0))" 2>/dev/null || echo "?")
  deleted=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('deleted',0))" 2>/dev/null || echo "?")
  log INFO "dedup scan: ${groups} duplicate groups, ${deleted} would be deleted"

  if [ "$groups" != "0" ] && [ "$groups" != "?" ]; then
    log WARN "duplicates found! Run with dry_run=false to clean: curl -X POST ${SSOT_URL}/admin/dedup -H 'X-API-Key: ...' -d '{\"namespace\":\"${SSOT_NAMESPACE}\",\"dry_run\":false}'"
  fi
}

# ── Main ──────────────────────────────────────────────────────────

MODE="${1:-all}"

case "$MODE" in
  --health-only)
    health_check
    ;;
  --backfill-only)
    run_backfill
    ;;
  --dedup-only)
    run_dedup
    ;;
  all|"")
    health_check || true
    run_backfill
    run_dedup
    ;;
  *)
    echo "Usage: $0 [--health-only|--backfill-only|--dedup-only]"
    exit 1
    ;;
esac

log INFO "maintenance complete"
