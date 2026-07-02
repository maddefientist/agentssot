"""Pure preflight decision for a synthesis run: validate the configured models
against live Ollama and decide proceed / degrade / skip. No DB, no side effects
(alerting + skipping happen in the caller based on the returned result)."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.model_validation import ModelListUnavailable, validate_models


@dataclass
class PreflightResult:
    proceed: bool
    primary: str | None
    fallback: str | None
    severity: str | None            # None = clean; else "warning" | "error"
    event: str | None               # alert event name, or None
    message: str | None
    detail: dict = field(default_factory=dict)


def evaluate(base_url: str, primary: str, fallback: str) -> PreflightResult:
    try:
        _present, missing = validate_models(base_url, [primary, fallback])
    except ModelListUnavailable as exc:
        return PreflightResult(
            False, None, None, "error", "synthesis.unreachable",
            f"Synthesis skipped: Ollama model list unreachable ({exc})",
            {"base_url": base_url},
        )
    primary_ok = bool(primary) and primary not in missing
    fallback_ok = bool(fallback) and fallback not in missing
    if not primary_ok and not fallback_ok:
        return PreflightResult(
            False, None, None, "error", "synthesis.model_missing",
            f"Synthesis skipped: both models missing from Ollama: {sorted(missing)}",
            {"missing": sorted(missing), "primary": primary, "fallback": fallback},
        )
    if not primary_ok and fallback_ok:
        return PreflightResult(
            True, fallback, None, "warning", "synthesis.model_missing",
            f"Synthesis primary {primary} missing; running on fallback {fallback}",
            {"missing": [primary]},
        )
    if primary_ok and not fallback_ok:
        return PreflightResult(
            True, primary, None, "warning", "synthesis.model_missing",
            f"Synthesis fallback {fallback} missing; running with no fallback",
            {"missing": [fallback]},
        )
    return PreflightResult(True, primary, fallback, None, None, None, {})