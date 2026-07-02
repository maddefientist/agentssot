"""Best-effort outbound alerting — channel-agnostic webhook POST.

Alerting must NEVER raise into or block hive. All failures are logged and
swallowed. Point ALERT_WEBHOOK_URL at ntfy / Discord / Slack / Madi / the
:9877 fleet server — swapping channels never touches this code.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger("agentssot.alerting")

ALERT_TIMEOUT_SECONDS = 5.0


def post_alert(
    webhook_url: str,
    event: str,
    severity: str,
    message: str,
    detail: dict[str, Any] | None = None,
    *,
    host_label: str = "hive",
    enabled: bool = True,
) -> bool:
    """POST a structured alert. Returns True if sent, False if skipped/failed.
    Never raises."""
    if not enabled or not webhook_url:
        logger.debug("alert suppressed (disabled or no url): %s", event)
        return False
    payload = {
        "source": "hive",
        "host": host_label,
        "severity": severity,
        "event": event,
        "message": message,
        "detail": detail or {},
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        resp = httpx.post(webhook_url, json=payload, timeout=ALERT_TIMEOUT_SECONDS)
        if resp.status_code >= 400:
            logger.warning("alert webhook returned %s for %s", resp.status_code, event)
            return False
        return True
    except Exception as exc:  # noqa: BLE001 — alerting must never propagate
        logger.warning("alert webhook failed for %s: %s", event, exc)
        return False


def send_alert(event: str, severity: str, message: str, detail: dict[str, Any] | None = None) -> bool:
    """Settings-aware wrapper used by app code. Imports settings lazily so this
    module stays import-clean for unit tests."""
    from .settings import get_settings

    s = get_settings()
    return post_alert(
        getattr(s, "alert_webhook_url", ""),
        event,
        severity,
        message,
        detail,
        host_label=getattr(s, "alert_host_label", "hive"),
        enabled=getattr(s, "alert_enabled", True),
    )