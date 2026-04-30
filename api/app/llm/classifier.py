"""gemma4:31b auto-classifier for ingested items.

Returns memory_type, confidence, abstract, summary, cwd_hints,
device_hints, entity_mentions, supersedes_likely. Strict JSON
output schema; out-of-band failures return a stub with confidence 0.0
so the caller can route to Review Queue.
"""
from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.settings import get_settings


SYSTEM_PROMPT = """You are a memory-typing classifier for a developer's
knowledge base. Output JSON only — no prose.

CONTEXT:
- The knowledge base stores typed items: command, rule, skill, entity,
  decision, episodic, fact.
- Multiple devices in a fleet share this store. Common entities include
  hosts (hari, unraid, dockers, blink, webvm, agent), services (jellyfin,
  qbittorrent, gluetun), and projects (agentssot, hive).

TIER DEFINITIONS:
- command: an exact invocation — something you can paste into a shell or
  API call. May be chained with &&, ;, or pipe, but it's still an exact
  invocation. Examples:
  "ssh unraid", "docker restart GluetunVPN", "curl :8088/health",
  "git rebase -i HEAD~3 && git push --force-with-lease origin main",
  "pyenv install 3.12.3 && pyenv global 3.12.3".
  NOT a command: "Use parameterized SQL queries" (that's a rule).
- rule: a universal constraint, guardrail, or hard requirement. Uses words
  like must, never, always, forbidden, require. It applies globally, not
  only in a specific situation. Examples:
  "Never rm -rf with wildcards", "Always specify namespace",
  "String concatenation into SQL is forbidden",
  "Explicit confirmation is required before destructive operations".
  NOT a rule: a diagnostic recipe for a specific error (that's a skill),
  or a description of how a system works (that's a fact).
- skill: a WHEN-X-DO-Y diagnostic or procedural recipe. It is triggered by
  a specific situation and describes steps to resolve or handle it.
  Examples:
  "When Gluetun port forwarding fails, restart Gluetun then qbit",
  "When FastAPI returns 422, check Pydantic alias mismatches",
  "If pytest fails with ImportError, check PYTHONPATH".
  NOT a skill: a generic command sequence without a trigger (that's a command),
  or a universal constraint (that's a rule).
- entity: a noun describing a host, service, person, or project with
  identifying details. Examples:
  "unraid (192.168.1.116) — storage hub",
  "jellyfin (192.168.1.116:8096) — media server on unraid".
- decision: a formal architectural or tooling choice recorded with rationale.
  Often includes "switched to...", "chose...", "decided..." along with a reason.
  Examples:
  "Embeddings: switched to nomic-embed-text on 2026-02-25 because...",
  "Chose SQLAlchemy 2.x over 1.4 for the new models. Reason:...".
- episodic: a session log, reflection, or narrative about something that happened
  on a specific date. Usually mentions a date and describes events or observations.
  Examples:
  "Session: hari on harihome 2026-02-10. Files touched: .ssh/config...",
  "2026-04-20 troubleshooting session: ingest pipeline started failing...",
  "Morning standup (async) 2026-04-22: deferred HNSW index...".
  NOT episodic: a formal decision without a date or narrative frame (that's a decision).
- fact: a neutral statement about how something works. No imperative, no
  constraint, no trigger condition. Use sparingly. Examples:
  "Postgres pgvector max_supported dimensions are 16000 for halfvec",
  "Ollama models are stored under ~/.ollama/models".

OUTPUT SCHEMA (strict JSON, all fields required):
{
  "memory_type": "<one of the seven values>",
  "confidence": 0.0..1.0,
  "abstract": "<≤50 token, single-sentence summary>",
  "summary": "<≤500 token paragraph>",
  "cwd_hints": ["<path or path-prefix mentioned>", ...],
  "device_hints": ["<host/device name mentioned>", ...],
  "entity_mentions": ["<entity slug mentioned>", ...],
  "supersedes_likely": true|false
}
"""

USER_TEMPLATE = """INPUT:
content: {content}
tags: {tags}
hint: {hint}

Respond with JSON only.
"""

_OUTPUT_KEYS = {
    "memory_type", "confidence", "abstract", "summary",
    "cwd_hints", "device_hints", "entity_mentions", "supersedes_likely",
}


def _ollama_url() -> str:
    s = get_settings()
    return (s.classifier_base_url or s.ollama_base_url).rstrip("/")


def _stub_low_conf(reason: str, content: str) -> dict[str, Any]:
    return {
        "memory_type": "fact",
        "confidence": 0.0,
        "abstract": (content[:120] or "").strip(),
        "summary": (content[:480] or "").strip(),
        "cwd_hints": [],
        "device_hints": [],
        "entity_mentions": [],
        "supersedes_likely": False,
        "_reason": reason,
    }


def classify(content: str, tags: list[str] | None = None, hint: str | None = None) -> dict[str, Any]:
    """Classify a single item. Always returns a dict with the 8 schema keys.

    On failure (Ollama down, malformed JSON, etc.) returns a low-confidence
    stub with reason — caller enqueues to Review Queue.
    """
    s = get_settings()
    if s.classifier_provider != "ollama":
        return _stub_low_conf("classifier_disabled", content)

    payload = {
        "model": s.classifier_model,
        "prompt": USER_TEMPLATE.format(
            content=content[:4000],
            tags=json.dumps(tags or []),
            hint=hint or "null",
        ),
        "system": SYSTEM_PROMPT,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1},
    }
    try:
        r = httpx.post(
            f"{_ollama_url()}/api/generate",
            json=payload,
            timeout=s.classifier_timeout_seconds,
        )
        r.raise_for_status()
        body = r.json()
        raw = body.get("response", "").strip()
        # Strip code fences if model wraps output
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return _stub_low_conf("no_json_in_response", content)
        parsed = json.loads(m.group(0))
        # Schema check
        if not _OUTPUT_KEYS.issubset(parsed.keys()):
            return _stub_low_conf("missing_keys", content)
        parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
        return parsed
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as e:
        return _stub_low_conf(f"classifier_error:{type(e).__name__}", content)
