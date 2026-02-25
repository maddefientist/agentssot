#!/usr/bin/env python3
"""
Monthly abstraction: extract cross-cutting principles from synthesized concepts.
Uses qwen3.5:cloud (Ollama Cloud, frontier reasoning) -- 1 call/month.
Fallback: kimi-k2.5:cloud.

Cron: 0 5 1 * * (1st of month at 5 AM)
"""
import json
import logging
import subprocess
import sys
from datetime import datetime
from uuid import uuid4

import requests

OLLAMA_URL = "http://localhost:11434"
CHAT_URL = f"{OLLAMA_URL}/api/chat"
EMBED_URL = f"{OLLAMA_URL}/api/embed"
EMBED_MODEL = "qwen3-embedding:latest"
CLOUD_MODELS = ["qwen3.5:cloud", "kimi-k2.5:cloud"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [abstract] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("cortex-abstract")

PROMPT = """You are a knowledge abstraction engine analyzing an AI developer's knowledge base. Below are synthesized concepts spanning multiple topics and projects from the last 3 months.

Extract PRINCIPLES — recurring patterns, preferences, and approaches that appear across 3+ different concept topics. These are general truths about how this developer works, not project-specific facts.

Rules:
- Only extract principles with evidence from 3+ different concept titles/topics
- Each principle must be actionable (a future AI session could apply it)
- Maximum 10 principles
- Each principle: 1-2 sentences
- Do NOT include project names or specific technologies — abstract to the pattern level

Respond with ONLY a JSON array, no markdown wrapping, no explanation:
[
  {
    "principle": "Clear statement of the cross-cutting principle",
    "evidence": ["concept_title_1", "concept_title_2", "concept_title_3"],
    "applies_when": "When to surface this in future sessions"
  }
]

Synthesized concepts:
"""


def psql(sql: str) -> str:
    result = subprocess.run(
        ["docker", "exec", "agentssot-db", "psql", "-U", os.environ.get("POSTGRES_USER", "ssot"), "-d", os.environ.get("POSTGRES_DB", "ssot"), "-t", "-A", "-c", sql],
        capture_output=True, text=True, timeout=60,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def get_embedding(text: str) -> list[float] | None:
    try:
        resp = requests.post(
            EMBED_URL,
            json={"model": EMBED_MODEL, "input": text},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        return embeddings[0] if embeddings else None
    except Exception as e:
        log.warning("Embedding failed: %s", e)
        return None


def call_llm(prompt: str) -> str | None:
    for model in CLOUD_MODELS:
        log.info("Trying model: %s", model)
        try:
            resp = requests.post(
                CHAT_URL,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=600,
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "").strip()
            if content:
                log.info("Got response from %s (%d chars)", model, len(content))
                return content
        except Exception as e:
            log.warning("%s failed: %s", model, e)
            continue
    return None


def main():
    # Get synthesized concepts from last 3 months
    raw = psql("""
        SELECT title, content, type::text
        FROM concepts
        WHERE namespace = 'claude-shared'
          AND NOT (tags @> ARRAY['superseded']::text[])
          AND created_at > NOW() - INTERVAL '3 months'
        ORDER BY confidence DESC, updated_at DESC
        LIMIT 200
    """)

    if not raw:
        log.info("No concepts found for abstraction")
        return

    concepts = []
    for line in raw.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            concepts.append({"title": parts[0].strip(), "content": parts[1].strip(), "type": parts[2].strip()})

    if len(concepts) < 10:
        log.info("Only %d concepts — need 10+ for meaningful abstraction", len(concepts))
        return

    log.info("Abstracting from %d concepts", len(concepts))

    # Format concepts for the prompt
    concept_text = "\n".join(
        f"[{c['type']}] {c['title']}: {c['content'][:300]}"
        for c in concepts
    )

    full_prompt = PROMPT + concept_text
    result = call_llm(full_prompt)

    if not result:
        log.error("All LLM calls failed")
        return

    # Parse JSON (handle markdown wrapping)
    cleaned = result.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        principles = json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.error("Failed to parse LLM output as JSON: %s", e)
        log.error("Raw output: %s", cleaned[:500])
        return

    if not isinstance(principles, list):
        log.error("Expected list, got %s", type(principles))
        return

    stored = 0
    for p in principles:
        text = p.get("principle", "")
        applies = p.get("applies_when", "general")
        evidence = p.get("evidence", [])
        if not text or len(text) < 20:
            continue

        full_text = f"{text} [Applies: {applies}] [Evidence: {', '.join(evidence[:5])}]"

        embedding = get_embedding(full_text)
        if not embedding:
            log.warning("Failed to embed principle, skipping: %s", text[:60])
            continue

        # Insert as a high-confidence principle concept
        concept_id = str(uuid4())
        embed_str = "[" + ",".join(str(x) for x in embedding) + "]"
        evidence_titles = ", ".join(f"'{e[:60]}'" for e in evidence[:5])

        safe_title = text[:200].replace("'", "''").replace("\\", "\\\\")
        safe_content = full_text.replace("'", "''").replace("\\", "\\\\")
        psql(f"""
            INSERT INTO concepts (id, namespace, type, scope, title, content, confidence, version, tags, embedding)
            VALUES (
                '{concept_id}',
                'claude-shared',
                'principle',
                'global',
                E'{safe_title}',
                E'{safe_content}',
                0.8,
                1,
                ARRAY['abstraction-engine', 'cross-cutting']::text[],
                '{embed_str}'
            )
        """)
        stored += 1
        log.info("Principle: %s", text[:80])

    log.info("Abstraction complete: %d principles stored from %d concepts", stored, len(concepts))


if __name__ == "__main__":
    main()
