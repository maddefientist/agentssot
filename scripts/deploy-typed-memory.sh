#!/usr/bin/env bash
# deploy-typed-memory.sh — DEPRECATED shim → scripts/deploy.sh
#
# The original M3 (typed memory) + M8 (secret scanning) deploy has long shipped.
# This name is kept only so existing references / muscle memory keep working.
# The real, generalized deploy now lives in scripts/deploy.sh (correct DB user,
# namespace-isolation gate, no one-off column checks).
#
# Usage is unchanged:
#   ./scripts/deploy-typed-memory.sh [--dry-run]

exec "$(cd "$(dirname "$0")" && pwd)/deploy.sh" "$@"
