"""DB-backed runtime overrides for hot control-plane settings.

Absence of a row means "use the Settings/.env default". Values are stored as
text and coerced against the live Settings field type at read/write time.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session


HOT_KEYS = frozenset({
    "synthesis_model",
    "synthesis_fallback_model",
    "synthesis_window_days",
    "synthesis_similarity_threshold",
    "synthesis_min_cluster_size",
    "ollama_reranker_model",
    "ollama_reranker_base_url",
    "ollama_reranker_fast_model",
    "ollama_reranker_fast_base_url",
    "reranker_candidate_multiplier",
    "ollama_embed_model",
    "ollama_base_url",
    "ollama_chat_model",
    "classifier_model",
    "classifier_base_url",
    "semantic_dedup_threshold",
    "supersession_similarity_threshold",
})

URL_KEYS = frozenset({
    "ollama_base_url",
    "ollama_reranker_base_url",
    "ollama_reranker_fast_base_url",
    "classifier_base_url",
})

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "synthesis_similarity_threshold": (0.0, 1.0),
    "semantic_dedup_threshold": (0.0, 1.0),
    "supersession_similarity_threshold": (0.0, 1.0),
    "synthesis_min_cluster_size": (1, 1000),
    "synthesis_window_days": (1, 3650),
    "reranker_candidate_multiplier": (1, 10),
}


def ensure_runtime_config_table(session: Session) -> None:
    session.execute(text(
        """
        CREATE TABLE IF NOT EXISTS runtime_config (
            key         TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by  TEXT
        )
        """
    ))
    session.commit()


def load_overrides(session: Session) -> dict[str, dict[str, Any]]:
    ensure_runtime_config_table(session)
    rows = session.execute(
        text("SELECT key, value, updated_at, updated_by FROM runtime_config")
    ).mappings().all()
    return {
        str(row["key"]): {
            "value": str(row["value"]),
            "updated_at": row["updated_at"],
            "updated_by": row["updated_by"],
        }
        for row in rows
    }


def _field_type(settings: Any, key: str) -> type:
    if not hasattr(settings, key):
        raise ValueError(f"Unknown settings field: {key}")
    current = getattr(settings, key)
    if isinstance(current, bool):
        return bool
    if isinstance(current, int):
        return int
    if isinstance(current, float):
        return float
    return str


def _validate_url(key: str, value: str) -> None:
    if value == "":
        return
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{key}: expected http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{key}: credentials in provider URLs are not allowed")


def coerce_value(settings: Any, key: str, raw: Any) -> Any:
    if key not in HOT_KEYS:
        raise ValueError(f"Unknown runtime override key: {key}")
    expected = _field_type(settings, key)

    if expected is bool:
        if isinstance(raw, bool):
            value = raw
        elif isinstance(raw, str) and raw.lower() in {"true", "1", "yes"}:
            value = True
        elif isinstance(raw, str) and raw.lower() in {"false", "0", "no"}:
            value = False
        else:
            raise ValueError(f"{key}: expected bool")
    elif expected is int:
        if isinstance(raw, bool):
            raise ValueError(f"{key}: expected int, not bool")
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}: expected int") from exc
    elif expected is float:
        if isinstance(raw, bool):
            raise ValueError(f"{key}: expected float, not bool")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}: expected float") from exc
    else:
        value = str(raw)

    if key in NUMERIC_RANGES:
        lo, hi = NUMERIC_RANGES[key]
        if not (lo <= value <= hi):
            raise ValueError(f"{key}: value {value} out of range [{lo}, {hi}]")
    if key in URL_KEYS:
        _validate_url(key, str(value))
    return value


def stringify_value(settings: Any, key: str, raw: Any) -> str:
    value = coerce_value(settings, key, raw)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def effective(settings: Any, overrides: Mapping[str, Any] | None, key: str) -> Any:
    if overrides and key in overrides:
        entry = overrides[key]
        raw = entry.get("value") if isinstance(entry, Mapping) else entry
        return coerce_value(settings, key, raw)
    return getattr(settings, key)


def apply_overrides(settings: Any, overrides: Mapping[str, Any]) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    for key in HOT_KEYS:
        if key not in overrides:
            continue
        value = effective(settings, overrides, key)
        object.__setattr__(settings, key, value)
        applied[key] = value
    return applied


def set_override(session: Session, settings: Any, key: str, value: Any, by: str | None) -> Any:
    try:
        stored = stringify_value(settings, key, value)
        effective_value = coerce_value(settings, key, stored)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    session.execute(
        text(
            """
            INSERT INTO runtime_config (key, value, updated_at, updated_by)
            VALUES (:key, :value, now(), :updated_by)
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = now(),
                updated_by = EXCLUDED.updated_by
            """
        ),
        {"key": key, "value": stored, "updated_by": by},
    )
    session.commit()
    return effective_value


def delete_override(session: Session, key: str) -> None:
    if key not in HOT_KEYS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown runtime override key")
    session.execute(text("DELETE FROM runtime_config WHERE key = :key"), {"key": key})
    session.commit()
