from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

MemoryType = Literal["skill", "decision", "fact"]

_VALID_MEMORY_TYPES: frozenset[str] = frozenset({"skill", "decision", "fact"})


class Lesson(TypedDict):
    claim: str
    citation: str
    memory_type: MemoryType
    confidence: float


def _coerce_memory_type(value: Any) -> MemoryType:
    if isinstance(value, str) and value in _VALID_MEMORY_TYPES:
        return value  # type: ignore[return-value]
    return "skill"


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, bool):
        # bool is an int subclass; treat as invalid for confidence.
        return 0.5
    if isinstance(value, (int, float)):
        try:
            clamped = float(value)
        except (TypeError, ValueError):
            return 0.5
        if clamped < 0.0 or clamped > 1.0:
            # Clamp out-of-range values into the valid band.
            return max(0.0, min(1.0, clamped))
        return clamped
    return 0.5


def parse_lessons(raw: str) -> list[Lesson]:
    """Parse tolerant JSON-lines of lessons.

    Never raises for a single bad line or malformed object — silently skips.
    Missing/invalid memory_type defaults to "skill"; confidence clamped to
    0.0..1.0 (default 0.5); lessons missing non-empty claim or citation are dropped.
    """
    lessons: list[Lesson] = []
    if not raw:
        return lessons

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj: Any = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue

        claim = obj.get("claim")
        citation = obj.get("citation")
        if not isinstance(claim, str) or not claim.strip():
            continue
        if not isinstance(citation, str) or not citation.strip():
            continue

        memory_type = _coerce_memory_type(obj.get("memory_type"))
        confidence = _coerce_confidence(obj.get("confidence"))

        lessons.append(
            Lesson(
                claim=claim.strip(),
                citation=citation.strip(),
                memory_type=memory_type,
                confidence=confidence,
            )
        )

    return lessons