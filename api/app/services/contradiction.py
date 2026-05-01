"""Contradiction detector: scan rules for negation patterns targeting an
entity that the new command/skill references.

Closes the OFF-LIMITS-unraid scenario: an old rule "Never access unraid"
must surface for review when a new command "ssh unraid" is ingested.
"""
from __future__ import annotations

import re
from typing import Any, Iterable


_NEGATION_PATTERNS = [
    r"\bnever\b",
    r"\bdo not\b",
    r"\bdon'?t\b",
    r"\boff[- ]limits\b",
    r"\bforbidden\b",
    r"\bmust not\b",
    r"\bshould not\b",
    r"\bshouldn'?t\b",
    r"\bavoid(?:ed)?\b",
]
_NEG_RE = re.compile("|".join(_NEGATION_PATTERNS), re.IGNORECASE)


def detect_contradictions(
    new_type: str,
    new_entity_refs: list[str],
    existing_rules: Iterable[Any],
) -> list[Any]:
    """Return rule items that contradict the new command/skill.

    new_type: type of the new item being ingested (only 'command' and
        'skill' trigger contradiction checks; everything else returns []).
    new_entity_refs: entity ids the new item links to.
    existing_rules: candidate rules in the same namespace.
    """
    if new_type not in ("command", "skill"):
        return []
    if not new_entity_refs:
        return []
    target = set(new_entity_refs)
    out: list[Any] = []
    for rule in existing_rules:
        if str(getattr(rule, "memory_type", "")) != "rule":
            continue
        rule_entities = set(getattr(rule, "entity_refs", []) or [])
        if not (rule_entities & target):
            continue
        content = getattr(rule, "content", "") or ""
        if _NEG_RE.search(content):
            out.append(rule)
    return out
