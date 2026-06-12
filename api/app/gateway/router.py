"""Hybrid intent router — the gateway's front door.

Two stages, fast-path first:

1. **Deterministic rules.** A short regex table maps obvious phrasings to an
   intent with zero model cost. An explicit intent on the inbound message
   short-circuits everything.
2. **Cheap local classifier.** Freeform text the rules miss is handed to a
   small local Ollama model that returns *only* a classification (intent +
   optional args) — never generated prose. If it is unavailable or returns
   garbage, we fall back to ``DEFAULT_INTENT`` (chat-local) so the front door
   never blocks and never crashes.

Classification is deliberately decoupled from generation: picking the executor
is cheap and local; the expensive model (if any) runs *inside* the chosen
executor.
"""
from __future__ import annotations

import json
import re
from typing import Awaitable, Callable, Optional

from .config import DEFAULT_INTENT, LOCAL_MODEL, OLLAMA_URL, VALID_INTENTS

# (compiled pattern, intent). First match wins, so order = priority.
_RULE_SOURCE: list[tuple[str, str]] = [
    (r"\b(teach|remember that|note that|store this|make a note)\b", "hive-tool"),
    (r"\b(recall|remember when|what do you know about|search (your )?memory)\b", "hive-tool"),
    (r"\b(memory stats|hive stats|are you (up|online|alive)|your status|system status)\b", "hive-tool"),
    (r"\b(scan|fleet|deploy|build|rebuild|run (a )?chain|dispatch|kick off|chain\.sh|spin up)\b", "dispatch"),
    (r"\b(brief|briefing|catch me up|what'?s new|morning report)\b", "briefing"),
]

RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), intent) for p, intent in _RULE_SOURCE
]

# An async callable: raw user text -> raw classifier response string.
Classifier = Callable[[str], Awaitable[str]]


def parse_classifier_response(raw: str) -> tuple[str, dict]:
    """Turn a classifier's raw output into ``(intent, args)``.

    Accepts either a bare intent token (``"dispatch"``) or a JSON object
    (``{"intent": "dispatch", "args": {...}}``). Anything unparseable, or an
    intent outside ``VALID_INTENTS``, degrades to ``(DEFAULT_INTENT, {})`` —
    the router must never raise on a bad model response.
    """
    if not raw:
        return DEFAULT_INTENT, {}
    text = raw.strip()

    # Try JSON first (it may be wrapped in prose/code fences — grab the object).
    candidate = text
    brace = text.find("{")
    if brace != -1:
        candidate = text[brace : text.rfind("}") + 1]
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            intent = str(obj.get("intent", "")).strip()
            args = obj.get("args", {})
            if not isinstance(args, dict):
                args = {}
            if intent in VALID_INTENTS:
                return intent, args
            return DEFAULT_INTENT, {}
    except (ValueError, TypeError):
        pass

    # Fall back to treating the whole (short) response as a bare intent token.
    token = text.split()[0].strip().strip('"').lower() if text.split() else ""
    if token in VALID_INTENTS:
        return token, {}
    return DEFAULT_INTENT, {}


class IntentRouter:
    """Routes inbound text to an intent class.

    ``classifier`` is optional; with none, only the rules table is consulted and
    misses fall straight to ``DEFAULT_INTENT``.
    """

    def __init__(self, classifier: Optional[Classifier] = None) -> None:
        self._classifier = classifier

    def match_rule(self, text: str) -> Optional[str]:
        for pattern, intent in RULES:
            if pattern.search(text):
                return intent
        return None

    async def classify(
        self, text: str, explicit: Optional[str] = None
    ) -> tuple[str, dict]:
        """Return ``(intent, args)`` for ``text``.

        Precedence: explicit intent → rules → classifier → default.
        """
        if explicit and explicit in VALID_INTENTS:
            return explicit, {}

        ruled = self.match_rule(text)
        if ruled is not None:
            return ruled, {}

        if self._classifier is not None:
            try:
                raw = await self._classifier(text)
            except Exception:
                return DEFAULT_INTENT, {}
            return parse_classifier_response(raw)

        return DEFAULT_INTENT, {}


def make_ollama_classifier(
    base_url: str = OLLAMA_URL, model: str = LOCAL_MODEL
) -> Classifier:
    """Build an async classifier backed by a local Ollama model.

    The model is instructed to emit a single JSON object and nothing else. Any
    transport error propagates to ``IntentRouter.classify``, which swallows it
    into the default intent.
    """
    import httpx

    intents = ", ".join(sorted(VALID_INTENTS))
    system = (
        "You are an intent classifier for an AI agent gateway. "
        f"Classify the user message into exactly one intent from: {intents}. "
        "Respond with ONLY a JSON object: "
        '{"intent": "<intent>", "args": {}}. No prose, no code fences.'
    )

    async def _classify(text: str) -> str:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "think": False,  # reasoning models: classify fast, no CoT
                    "format": "json",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": text},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")

    return _classify
