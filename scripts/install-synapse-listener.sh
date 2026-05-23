#!/usr/bin/env bash
# install-synapse-listener.sh — idempotent installer for the synapse-listener systemd user unit.
#
# Usage:
#   bash /opt/agentssot/scripts/install-synapse-listener.sh
#
# What it does:
#   1. Checks that systemd --user is available (skips silently on macOS / containers without systemd).
#   2. Copies synapse-listener.service to ~/.config/systemd/user/.
#   3. Runs daemon-reload + enable --now.
#
# Safe to run multiple times.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_SRC="$SCRIPT_DIR/synapse-listener.service"
SERVICE_NAME="synapse-listener.service"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

# ── Guard: systemd --user available? ──────────────────────────────────────────
if ! command -v systemctl &>/dev/null; then
    echo "[synapse-listener] systemctl not found (likely macOS or minimal container) — skipping install." >&2
    exit 0
fi

if ! systemctl --user --no-pager status &>/dev/null; then
    # On some headless systems 'systemctl --user' may fail if no user session bus
    echo "[synapse-listener] systemd user bus unavailable — skipping install." >&2
    exit 0
fi

# ── Verify source files exist ──────────────────────────────────────────────────
if [ ! -f "$SERVICE_SRC" ]; then
    echo "[synapse-listener] ERROR: Service file not found at $SERVICE_SRC" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/synapse_listener.py" ]; then
    echo "[synapse-listener] ERROR: synapse_listener.py not found at $SCRIPT_DIR/synapse_listener.py" >&2
    exit 1
fi

# ── Install ────────────────────────────────────────────────────────────────────
mkdir -p "$SYSTEMD_USER_DIR"
cp "$SERVICE_SRC" "$SYSTEMD_USER_DIR/$SERVICE_NAME"
echo "[synapse-listener] Installed $SERVICE_NAME to $SYSTEMD_USER_DIR/"

systemctl --user daemon-reload
echo "[synapse-listener] daemon-reload done"

systemctl --user enable --now "$SERVICE_NAME"
echo "[synapse-listener] enabled and started"

# ── Status ─────────────────────────────────────────────────────────────────────
echo ""
systemctl --user --no-pager status "$SERVICE_NAME" || true
