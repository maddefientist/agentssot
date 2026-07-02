# Preflight Model Validation + Webhook Alerting for Hive

**Date:** 2026-07-02
**Component:** agentssot-api (hive)
**Status:** Draft — awaiting operator review

## Problem

On 2026-07-02 hive became API-unreachable. Root cause: the nightly synthesis
batch (03:00 UTC) used `synthesis_fallback_model=qwen3.5:27b`, a model that had
been retired (migrated to `qwen3.6:27b`). Every fallback call returned Ollama
404, and the synthesis loop ground through cluster after cluster issuing
synchronous `httpx.post(..., timeout=120)` calls that starved the ASGI event
loop — so `/health`, `hive_recall`, everything hung. Docker reported the
container "healthy" because its healthcheck has `Retries: 10`, masking ~5 min of
failing probes. Nothing alerted the operator; the failure was discovered only by
noticing recall was dead.

Two gaps this exposes:

1. **No preflight validation** that a configured model actually exists before it
   is used. Models get migrated/retired out from under long-lived containers.
2. **No alerting.** When hive gets stuck it fails silently.

This spec covers those two gaps. It does **not** cover an interactive
model-picker (a separate, later feature) — but its webhook is the seam that
feature will plug into.

## Scope

**In scope**
- Validate synthesis models (primary + fallback) against Ollama's live model
  list before a synthesis run, and at startup.
- On missing model: skip the run (never loop), degrade gracefully when only one
  of primary/fallback is missing, and emit one alert.
- A config-driven, best-effort webhook alert path usable for this and future
  alerts.
- Alert on synthesis batch failure (not just preflight).

**Out of scope (noted, not built here)**
- Interactive "pick a replacement model" interface.
- Auto-selecting a replacement model.
- Converting the blocking synthesis calls to non-blocking (see Related Work).
- Docker healthcheck/autoheal tuning (see Related Work).

## Design

### 1. Model validation helper — `app/llm/model_validation.py`

Reuses the live-ping pattern already used by `/doctor` (which pings Ollama
rather than trusting config presence).

```
def list_available_models(base_url: str, timeout: float = 5.0) -> set[str]:
    """GET {base_url}/api/tags → set of model names. Raises on unreachable."""

def validate_models(base_url, required: list[str]) -> tuple[set[str], set[str]]:
    """Return (present, missing) for the required model names.
    /api/tags lists BOTH local and cloud models (verified), so one call
    covers qwen3.6:27b and qwen3.5:397b-cloud alike."""
```

- Single source of truth for "does this model exist," callable from startup,
  the synthesis preflight, and `/doctor`.
- Network/timeout errors surface as an exception the caller treats as
  "cannot validate" → skip + alert (fail safe, never loop).

### 2. Synthesis preflight — in `synthesis/loop.py`, before the cluster loop

Resolve effective `synthesis_model` and `synthesis_fallback_model` (DB overrides
win over `.env`, as today), then:

| Primary | Fallback | Action |
|---------|----------|--------|
| present | present  | proceed normally |
| present | missing  | proceed; WARN + alert (degraded: no fallback) |
| missing | present  | proceed using fallback as primary; WARN + alert |
| missing | missing  | **skip run**; ERROR + alert; return stats early |
| Ollama unreachable | — | skip run; ERROR + alert |

Checked **once per run**, not per cluster — cheap and prevents the per-cluster
404 storm. The per-cluster savepoint/try-except stays as the last-resort guard.

### 3. Alert path — `app/alerting.py`

```
def send_alert(event: str, severity: str, message: str, detail: dict | None = None) -> None:
    """Best-effort POST to settings.alert_webhook_url. Never raises, never blocks
    meaningfully (timeout 5s). No-op when alerting disabled or URL unset."""
```

Payload:
```json
{
  "source": "hive",
  "host": "hari",
  "severity": "error|warning|info",
  "event": "synthesis.model_missing",
  "message": "human-readable summary",
  "detail": { "missing": ["qwen3.5:27b"], "namespace": "claude-shared" },
  "timestamp": "2026-07-02T03:00:00Z"
}
```

- Transport: generic webhook (`ALERT_WEBHOOK_URL`). Channel-agnostic — point it
  at ntfy/Discord/Slack/Madi/the :9877 fleet server. Swapping channels never
  touches hive code.
- Failure of the webhook itself is logged and swallowed — alerting must never
  take hive down.
- De-dup: synthesis runs once/day, so per-run alerting is naturally
  rate-limited. Within a run, at most one preflight alert + one failure-summary
  alert. No persistent alert-state store needed for v1.

### 4. Alert triggers (v1)
- `synthesis.model_missing` — preflight found a required model absent (severity
  per table above).
- `synthesis.unreachable` — Ollama `/api/tags` unreachable at preflight.
- `synthesis.run_failed` — the run raised, or ended with a high cluster-failure
  ratio (e.g. >50% of clusters failed). Summary alert, once per run.

### 5. Config — `settings.py` + `.env`
```
ALERT_ENABLED=true
ALERT_WEBHOOK_URL=            # empty = alerting no-ops
ALERT_HOST_LABEL=hari         # identifies the source in payloads
```
No DB runtime-override needed for v1 (not hot-tuned like model choices).

### 6. Startup preflight — `main.py` lifespan
When the synthesis loop starts, run one validation of the synthesis models and
log the result (info if OK, warning + alert if not). Catches config drift at
boot instead of waiting until 03:00.

## Data Flow
```
03:00 loop wake / startup
   └─ preflight: list_available_models(ollama) → validate synthesis models
        ├─ ok            → run synthesis batch (existing path)
        ├─ degraded/miss → send_alert(...) ; skip or run-with-warning per table
        └─ unreachable   → send_alert(...) ; skip
   run batch → on raise / high failure ratio → send_alert("synthesis.run_failed")
send_alert → POST ALERT_WEBHOOK_URL (best-effort, 5s, swallow errors)
```

## Testing
- `validate_models`: unit test with a fake `/api/tags` payload (present, partial,
  empty, unreachable).
- Preflight decision table: parametrized test over the 5 rows → asserts
  proceed/skip + which alerts fire (mock `send_alert`).
- `send_alert`: posts expected JSON to a mock URL; no-ops when URL empty; does
  not raise when the endpoint errors/times out.
- Regression: reproduce the incident — set fallback to a bogus model, assert the
  run skips and fires `synthesis.model_missing` instead of looping.

## Error Handling
- Every failure mode fails **safe**: skip the run rather than loop; swallow
  webhook errors rather than crash synthesis.
- All decisions logged with structured `extra={...}` matching existing logging.

## Related Work (not in this spec)
- **Event-loop blocking:** synthesis uses synchronous `httpx.post` inside an
  async task, so a slow/large batch freezes `/health` and all requests. Offload
  to a thread (`asyncio.to_thread`) or switch to async httpx so the API stays
  responsive during synthesis. Recommended as an immediate follow-up — it is the
  reason the incident caused *total* unreachability rather than just failed
  synthesis.
- **Healthcheck honesty:** `Retries: 10` hid the failure for ~5 min; consider
  lowering it and/or adding autoheal so `unhealthy` triggers a restart.
- **Interactive model-picker:** the ambitious "ask the operator to choose a live
  model" interface. This webhook is its outbound seam.

## Open Question (operator)
- **Alert sink:** defaulted to generic `ALERT_WEBHOOK_URL` (empty until you set
  it). Confirm the channel — ntfy, Discord/Slack, Madi, or the :9877 fleet
  server — and I'll document the exact URL/format in the plan.
