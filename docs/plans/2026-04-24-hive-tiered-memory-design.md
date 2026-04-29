# AgentSSOT / Hive — Tiered Memory Overhaul Design

**Date:** 2026-04-24
**Author:** Madi (in collaboration with MadDefientist)
**Status:** Approved design, pending implementation plan
**Repo:** `/opt/agentssot` @ commit `7ae39bf`

---

## Problem Statement

The hive has 4541 items, 959 concepts, 6 enrolled devices, and visible signs of intelligence regression:

1. Agents skip `hive_recall` and go down rabbit holes finding facts the hive already holds (e.g. "ssh unraid" → 192.168.1.116 is well-indexed but agents explore for it instead).
2. Episodic content (session logs, reflections) drowns procedural knowledge (commands, rules, skills) in a single ranked list — recall returns the right answer next to noise.
3. Stale items contradict newer truth without expiring (e.g. an old `openclaw/identity/system.md` declared `unraid` "OFF LIMITS" while ~30 newer skills depend on accessing it).
4. No "real memory" mechanism — knowledge sits behind a pull-based API that agents must remember to call.
5. Each device has its own namespace but no unified, tier-aware view across them.

The trust the operator built in this knowledge base is cracking. Errors that exist under their nose (silent supersession, unindexed contradictions, stale guardrails) are eroding confidence.

## Design Goals

1. **Typed knowledge with tier-aware retrieval.** Commands, rules, skills, entities, decisions, and episodic content occupy distinct tiers with distinct retrieval semantics.
2. **Push-based context.** Agents receive a cwd-aware "loadout" of relevant abstracts at SessionStart automatically, without needing to remember to call `hive_recall`.
3. **Lifecycle that catches contradictions.** Auto-supersession, soft expiration, low-confidence flagging, and operator review queue.
4. **Ollama-powered heavy lifting.** Classification, layer pre-compute, reranking, summarization run locally on Hari's Ollama. Anthropic tokens reserved for Claude reasoning.
5. **Migratable end-to-end.** Every change additive. Every phase reversible. The system stays green at every checkpoint.
6. **Trust restored through observability.** Review Queue, daily lifecycle digest, recall regression alarms, loadout snapshot diffing.

## Non-Goals

- Replacing the underlying Postgres + pgvector store.
- Changing the embedding model (stays on `nomic-embed-text`, 768d).
- Adding a frontier model (Anthropic) into the write path. Classification is local Ollama.
- Building a public-facing API. The hive remains LAN-only.

---

## 1. Architecture

Three layers communicating through clean interfaces.

```
┌──────────────────────────────────────────────────────────────────┐
│  CLAUDE-FACING SURFACE (thin, cheap, abstract-first)             │
│  ┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐  │
│  │ SessionStart │   │  hive_recall    │   │  hive_expand     │  │
│  │ Loadout Hook │   │  (tier-bucket)  │   │  (L2 on demand)  │  │
│  └──────┬───────┘   └────────┬────────┘   └──────┬───────────┘  │
└─────────┼────────────────────┼───────────────────┼──────────────┘
          │                    │                   │
┌─────────▼────────────────────▼───────────────────▼──────────────┐
│  RETRIEVAL & ASSEMBLY (Ollama-powered, no Claude tokens)         │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────────────────┐  │
│  │ Cwd→Entity   │  │ Tier-aware  │  │ Two-tier Reranker      │  │
│  │ Resolver     │  │ Recall      │  │ (4B fast / 8B deep)    │  │
│  └──────────────┘  └─────────────┘  └────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Loadout Builder: shared core + device overlay + budget cap │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Auto-classify + Layer pre-compute + Contradiction detector │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────┬────────────────────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────────────────────┐
│  TYPED MEMORY STORE (Postgres + pgvector)                        │
│  KnowledgeItem  ─►  type ∈ {command,rule,skill,entity,decision,episodic}
│                 ─►  layer columns: abstract, summary, full_content │
│                 ─►  expires_at, superseded_by, cwd_hints[],        │
│                     entity_refs[], rule_refs[], confidence,        │
│                     loadout_priority                                │
│  Entity         ─►  canonical fleet hosts/services/people          │
│  Concept        ─►  synthesized cross-item knowledge (existing)    │
│  deletion_log   ─►  audit row on every hard delete                 │
└──────────────────────────────────────────────────────────────────┘
```

### What changes vs preserved

| Component | Status |
|---|---|
| Postgres + pgvector + namespaces + auth | Preserved as-is |
| `MemoryType`, `MemoryCategory`, `ContentLayer`, `Entity` enums/tables | Activated (already in schema, mostly unwired) |
| Daily synthesis → `Concept` table | Preserved; synthesis model bumped to `qwen3.6:27b` |
| `Entity` table | Promoted to first-class hub of the typed graph |
| Cwd→entity resolver | New — small Postgres lookup with `cwd_hints` GIN index |
| Loadout hook | New — replaces SessionStart hint with full structured bundle |
| Tier-bucketed `/recall` response | New — opt-in via `bucketed=true`, default flips after caller migration |
| `hive_expand` MCP tool | New — fetches L1/L2 by id |
| Auto-typing on ingest | New — `gemma4:31b` classifies; sub-0.6 confidence flagged for review |
| Contradiction detector | New — at ingest, checks for negation rules contradicting incoming command/skill |
| Layer pre-compute (L0/L1) | New — computed at ingest, not query time |
| `expires_at` / `superseded_by` / `confidence` lifecycle | New schema columns + indexes |
| Two-tier reranker | New — 4B for procedural, 8B for nuanced (existing 8B preserved) |
| MCP admin operations (`delete`, `dedup`) | Fixed — currently silently 403 due to writer-only key resolution |
| `hive_loadout()` / `hive_supersede` / `hive_expire` / `hive_promote` / `hive_review_queue` / `hive_guide` | New MCP tools |
| `/agent-guide` REST endpoint | New — runtime LAN documentation, per-key tailored |
| Cortex `/review`, `/loadout`, `/entities` routes | New top-level pages |
| Cortex 3D map node coloring by tier | New visual cue |
| Daily `lifecycle_sweep` background job | New — extends existing `api/app/background.py` |

### Behavioral change the operator and agents will feel

1. Every new session, the agent receives ~600–800 tokens of cwd-relevant `command` + `rule` + `entity` abstracts pre-loaded. No more "let me find the ssh command" exploration loops.
2. `hive_recall` returns `{commands, rules, skills, entities, decisions}` buckets. Episodic excluded by default, opt-in via `tiers=[episodic]`.
3. Stale items are flagged or expired automatically; contradictions surface in the Review Queue rather than silently misleading sessions.

---

## 2. Tier Taxonomy & Data Model

### The 6 tiers

| Tier | `MemoryType` value | Purpose | Typical L0 abstract | Loadout default |
|---|---|---|---|---|
| `command` | new enum value | Exact invocations & references — `ssh unraid`, ports, URLs, API references | `ssh unraid → 192.168.1.116 root` | yes, cwd-filtered |
| `rule` | new enum value (also keeps existing `preference` for backcompat) | Always/never directives, guardrails | `Never use rm -rf with wildcards on system dirs` | yes, all loaded |
| `skill` | existing `skill` | When-X-do-Y-verify-Z recipes | `When Gluetun port forwarding fails: restart Gluetun then qbit` | top-3 cwd-relevant |
| `entity` | new enum value (rows also written to `Entity` table) | Canonical hosts/services/people/projects | `unraid (192.168.1.116, root) — Storage hub, NFS to hari` | yes, cwd-filtered |
| `decision` | existing `decision` | Architectural choices + rationale | `Embeddings: nomic-embed-text (768d), 2026-02-25, replaced qwen3-embedding (4096d)` | no (recall on demand) |
| `episodic` | new enum value (also accepts `session_summary`, `fact` for backcompat) | Session logs, reflections, run-insights | not pre-computed | no (opt-in only) |

Existing values (`fact`, `preference`, `reference`, `correction`, `session_summary`) remain valid and resolvable. Backfill (Phase 3) re-classifies many of them to the new tiers; items where classifier confidence < 0.6 keep their existing types and enter the Review Queue.

### Schema additions (Alembic migration in Phase 0)

All additive, all defaulted. Existing rows remain valid without modification.

```sql
ALTER TABLE knowledge_items
  ADD COLUMN expires_at         TIMESTAMPTZ NULL,
  ADD COLUMN superseded_by      UUID NULL REFERENCES knowledge_items(id),
  ADD COLUMN confidence         FLOAT NOT NULL DEFAULT 1.0,
  ADD COLUMN entity_refs        JSONB NOT NULL DEFAULT '[]',
  ADD COLUMN rule_refs          JSONB NOT NULL DEFAULT '[]',
  ADD COLUMN cwd_hints          JSONB NOT NULL DEFAULT '[]',
  ADD COLUMN device_hints       JSONB NOT NULL DEFAULT '[]',
  ADD COLUMN loadout_priority   INT NOT NULL DEFAULT 0,
  ADD COLUMN abstract           TEXT NULL,
  ADD COLUMN summary            TEXT NULL,
  ADD COLUMN last_classified_at TIMESTAMPTZ NULL;

CREATE INDEX ix_ki_expires_at         ON knowledge_items(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX ix_ki_superseded_by      ON knowledge_items(superseded_by) WHERE superseded_by IS NOT NULL;
CREATE INDEX ix_ki_loadout_priority   ON knowledge_items(loadout_priority DESC) WHERE confidence >= 0.5;
CREATE INDEX ix_ki_cwd_hints_gin      ON knowledge_items USING GIN (cwd_hints);
CREATE INDEX ix_ki_entity_refs_gin    ON knowledge_items USING GIN (entity_refs);
CREATE INDEX ix_ki_last_classified_at ON knowledge_items(last_classified_at);

CREATE TABLE deletion_log (
  id          UUID PRIMARY KEY,
  item_id     UUID NOT NULL,
  namespace   VARCHAR NOT NULL,
  reason      TEXT,
  deleted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  deleted_by  VARCHAR,
  payload     JSONB
);

ALTER TYPE memory_type ADD VALUE IF NOT EXISTS 'command';
ALTER TYPE memory_type ADD VALUE IF NOT EXISTS 'rule';
ALTER TYPE memory_type ADD VALUE IF NOT EXISTS 'entity';
ALTER TYPE memory_type ADD VALUE IF NOT EXISTS 'episodic';
```

Rollback: drop new columns, indexes, and `deletion_log` table. Enum values cannot be dropped without table rewrite but are harmless if unused.

### Confidence semantics — clarified to prevent drift

There are two confidence systems in the model. They are intentionally distinct and never directly affect each other:

- **`KnowledgeItem.confidence` — lifecycle confidence.** "Was this the right answer?" Driven by classifier on ingest, decayed on supersession, decayed by aging without recall. Determines loadout eligibility (≥ 0.5) and exclusion thresholds.
- **`Concept.confidence` — synthesis confidence.** "Does this pattern hold across many items?" Driven by `RecallEvent` and `ConceptFeedback`. Determines how strongly a synthesized cross-item concept is reinforced.

A KnowledgeItem may be high-confidence (correct) but referenced by a low-confidence Concept (pattern weak). Or vice versa. The two systems coexist and both surface in `/review` for distinct reasons.

This distinction is documented in `agent_guide.md.tmpl`, the `models.py` docstrings, and the `/agent-guide` endpoint.

### Entity table activation

`Entity` table already exists in `models.py:135`. We start populating it canonically:

- Each fleet host, service, project, person gets exactly one row.
- `entity.id` is referenced by `KnowledgeItem.entity_refs[]`.
- `entity.cwd_hints` mirrors the knowledge-item field for fast loadout filtering.
- Updating an entity (e.g. unraid IP changes) updates everything that links to it implicitly.

---

## 3. Retrieval & Loadout (read path)

Two distinct flows: the **loadout** (push, runs once at SessionStart and on demand mid-session) and **recall** (pull, runs whenever the agent asks).

### A. Loadout — cwd-aware push context

```
SessionStart hook fires (or hive_loadout() called explicitly)
        │
        ▼
┌────────────────────────────┐
│ resolve_loadout_context()  │  Python in MCP server / FastAPI
│  inputs: cwd, device_id    │
└─────────┬──────────────────┘
          ▼
┌────────────────────────────────────────────────────┐
│ 1. Cwd→Entity resolver (Postgres lookup)           │
│    SELECT id, slug FROM entities                    │
│      WHERE :cwd LIKE ANY(cwd_hints) OR slug = :hint │
│    → entity_ids = {hari, agentssot, hive}           │
└─────────┬──────────────────────────────────────────┘
          ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. Pull tier slots in priority order, capped per slot          │
│    rules:    all active rules in claude-shared + device         │
│    commands: top N where entity_refs ∩ entity_ids               │
│    entities: rows for entity_ids + linked entities (1 hop)      │
│    skills:   top M where entity_refs ∩ entity_ids               │
│    decisions: 0 by default (recall on demand)                   │
│    episodic:  0 by default                                      │
└─────────┬──────────────────────────────────────────────────────┘
          ▼
┌────────────────────────────────────────────────────────────────┐
│ 3. Apply device overlay rules (asymmetric privacy)              │
│    - device commands targeting same entity REPLACE shared       │
│    - device rules ADD to shared (never replace; merge by union) │
│    - device entities ANNOTATE shared (cannot redefine)          │
└─────────┬──────────────────────────────────────────────────────┘
          ▼
┌────────────────────────────────────────────────────────────────┐
│ 4. Token budget pack (greedy by loadout_priority desc)          │
│    target ≤ 750 tokens of L0 abstracts                          │
│    item shape: "[type] [title] — [abstract] (id=...)"           │
│    overflow → noted as "+N more, call hive_expand"              │
└─────────┬──────────────────────────────────────────────────────┘
          ▼
   Hook output: SessionStart context block (text/plain)
```

### Loadout output (what Claude sees in SessionStart hook)

```
=== Hive Loadout (cwd=/opt/agentssot, device=hari) ===

[rules] (3)
- Never `rm -rf` with wildcards on system dirs (id=r1)
- Always specify namespace on /recall (id=r2)
- No hardcoded secrets in code or output (id=r3)

[entities] (3)
- hari (192.168.1.225, hari@) — AI workhorse, Ollama host (id=e1)
- agentssot — /opt/agentssot, FastAPI memory service @ :8088 (id=e2)
- hive — Postgres+pgvector store, 4541 items, 959 concepts (id=e3)

[commands] (5)
- ssh hari (id=c1)
- docker compose up -d --build api (id=c2)
- curl -H "X-Api-Key: $KEY" :8088/api/v1/knowledge/recall (id=c3)
- ollama list (id=c4)
- pytest api/tests -k recall (id=c5)

[skills] (3)
- When recall returns stale data → check superseded_by chain (id=s1)
- When admin op 403s → use admin.json key, not agent.json (id=s2)
- When Ollama OOM → check OLLAMA_EMBED_CPU_ONLY=true (id=s3)

+8 more items available — hive_expand or hive_recall to fetch.
```

Target ~600–800 tokens, deterministic ordering for prompt-cache hit rate.

### Mid-session loadout reproducibility

The loadout is also delivered via the `hive_loadout()` MCP tool, accepting optional cwd and device overrides (defaults inferred from environment). After Claude compacts mid-session, the post-compaction context loses the SessionStart loadout. The recovery is one MCP call: `hive_loadout()` returns the same bundle. This is referenced in the post-compaction protocol in `~/.claude/CLAUDE.md` (updated in Phase 4).

### Loadout fallback on failure

The SessionStart hook has a 2-second budget. If exceeded (Postgres slow, Ollama down, etc.), the hook falls back to a static "call hive_recall on your task keywords" hint. The session never blocks on loadout assembly.

### B. Recall — tier-bucketed response

```http
POST /api/v1/knowledge/recall
{
  "query": "ssh unraid",
  "namespace": "claude-shared",
  "bucketed": true,
  "tiers": ["command", "rule", "skill", "entity", "decision"],
  "top_per_tier": {"command": 3, "rule": 2, "skill": 5, "entity": 3, "decision": 2},
  "expand_layer": "abstract"
}

Response:
{
  "buckets": {
    "command":  [{"id": ..., "abstract": "ssh unraid → 192.168.1.116", "score": 0.78, ...}],
    "rule":     [],
    "skill":    [{"id": ..., "abstract": "...", ...}],
    "entity":   [{"id": ..., "abstract": "unraid (192.168.1.116) — ...", ...}],
    "decision": []
  },
  "diagnostics": {
    "candidates_per_tier": {"command": 17, "skill": 42, ...},
    "rerank_ms": 84,
    "vec_ms": 42,
    "reranker_used": "qwen3-reranker-4b"
  }
}
```

Default tier set when `tiers` unspecified: `[command, rule, skill, entity, decision]` — episodic excluded.

`bucketed=true` is opt-in until Phase 6 default flip. Old callers continue receiving the existing flat list.

### C. Two-tier reranker

```
Vector search → candidates → tier-routing decision
                                      │
                  ┌───────────────────┴──────────────────┐
                  ▼                                      ▼
        tiers ⊆ {command, rule, entity}         tiers includes {skill, decision, episodic}
                  │                                      │
                  ▼                                      ▼
        Qwen3-Reranker-4B                       Qwen3-Reranker-8B
        ~80–150ms, exact-ish                    ~300–500ms, nuanced
```

The 4B reranker fires for procedural-only queries (commands, rules, entities). The 8B fires when nuanced ranking matters (skills, decisions, episodic). The two can run on separate Ollama hosts via `OLLAMA_RERANKER_BASE_URL` (split-host capability landed in commit `1d49033`).

If 4B is unavailable, system falls back to 8B for all queries (slower but correct). If 8B is unavailable, system falls back to vector-only (no rerank, with a warning in `diagnostics`).

### D. `hive_expand` MCP tool

```
hive_expand(item_id, layer="full")
  → returns L1 summary or L2 full content for the item
  → idempotent, cheap, no side effects
  → use when an abstract from loadout/recall isn't enough
```

Same auth as recall (reader+). The loadout ships abstracts only; expand is the one-call escape hatch when an agent needs concrete steps.

---

## 4. Ingestion & Lifecycle (write path)

### A. Ingest pipeline

```
POST /api/v1/knowledge/ingest    (or hive_teach via MCP)
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│ 1. Validate + dedup-check                                  │
│    - schema validation                                      │
│    - vector similarity vs last 30d (≥ 0.95 → flag dup)     │
│    - if dup: caller chooses skip / merge / supersede        │
└────────┬───────────────────────────────────────────────────┘
         ▼
┌────────────────────────────────────────────────────────────┐
│ 2. Auto-classify (gemma4:31b, ~200–400ms local)            │
│    inputs: content, tags, caller-supplied type hint         │
│    outputs:                                                 │
│      - memory_type ∈ {command|rule|skill|entity|decision|   │
│                       episodic|fact}                        │
│      - confidence (0..1)                                    │
│      - cwd_hints[]                                          │
│      - device_hints[]                                       │
│      - entity_mentions[]                                    │
│      - rule_refs_likely[]                                   │
│      - supersedes_likely (bool)                             │
│    if confidence < 0.6: keep as 'fact', flag review queue   │
└────────┬───────────────────────────────────────────────────┘
         ▼
┌────────────────────────────────────────────────────────────┐
│ 3. Layer pre-compute (gemma4:31b, ~300–600ms)              │
│    - L0 abstract: 1 sentence, ≤ 50 tokens                   │
│    - L1 summary:  paragraph, ≤ 500 tokens                   │
│    - L2 full:     original content                          │
│    Stored in KnowledgeItem.{abstract, summary, full_content}│
└────────┬───────────────────────────────────────────────────┘
         ▼
┌────────────────────────────────────────────────────────────┐
│ 4. Embed (existing nomic-embed-text path, no change)       │
│    - embed L1 summary (best signal-to-noise)               │
│    - HNSW index update                                     │
└────────┬───────────────────────────────────────────────────┘
         ▼
┌────────────────────────────────────────────────────────────┐
│ 5. Supersession check                                      │
│    if memory_type ∈ {command, rule, entity, decision}:     │
│      find existing where (entity_refs ∩ new) AND           │
│        (memory_type == new) AND                            │
│        (subject overlap judged by classifier)              │
│      if found: set old.superseded_by = new.id              │
│        old.confidence *= 0.3, set expires_at = now() + 30d │
└────────┬───────────────────────────────────────────────────┘
         ▼
┌────────────────────────────────────────────────────────────┐
│ 6. Contradiction detector (NEW — closes the OFF-LIMITS gap)│
│    if memory_type ∈ {command, skill}:                      │
│      for each linked entity, query existing rule items      │
│        where content matches negation patterns              │
│        ("never", "off limits", "do not", "forbidden")       │
│        targeting that entity                                │
│      if matches: enqueue ContradictionReview row at HIGH    │
│        priority — operator confirms whether the rule is      │
│        stale or the new item should be rejected             │
│    Same gemma4:31b call as classify, two extra prompt lines │
└────────┬───────────────────────────────────────────────────┘
         ▼
┌────────────────────────────────────────────────────────────┐
│ 7. Persist + return id                                     │
└────────────────────────────────────────────────────────────┘
```

### Why the contradiction detector is critical

The OFF-LIMITS-unraid scenario was: a stale rule contradicting newer commands and skills, never detected because rules and commands are different `memory_type`s and the supersession check (step 5) only compares within the same type.

Step 6 closes the gap. When a new `command` ("ssh unraid") or `skill` ("when working on unraid mount...") is ingested for entity `unraid`, the system checks for `rule` items mentioning `unraid` with negation language. If found, a `ContradictionReview` row is created at HIGH priority. The operator confirms in the Review page whether the rule is stale (delete or supersede it) or whether the new item should be rejected.

This is structural prevention of the exact failure that started this overhaul.

### Failure modes — explicit

- **gemma4:31b unreachable at ingest:** classifier returns confidence 0.0; item persists as `fact`, enters Review Queue with reason `classifier_unavailable`.
- **Postgres unreachable:** standard service-down; ingest endpoint returns 503; `hive_status` reports degraded.
- **Loadout hook timeout (>2s):** falls back to static hint, never blocks session start.

### B. Auto-typing classifier — concrete

One prompt, strict JSON schema, gemma4:31b.

```
SYSTEM: You are a memory-typing classifier. Output JSON only.

INPUT:
  content: <the new item>
  tags: <caller-provided tags, may be empty>
  hint: <caller-provided type hint, may be null>

OUTPUT (strict JSON):
{
  "memory_type": "command|rule|skill|entity|decision|episodic|fact",
  "confidence": 0.0..1.0,
  "abstract": "≤50 tokens, one sentence",
  "summary": "≤500 tokens, paragraph",
  "cwd_hints": ["..."],
  "device_hints": ["..."],
  "entity_mentions": ["unraid", "hari", ...],
  "supersedes_likely": true|false
}

DECISION RULES:
- imperative single line ("ssh unraid", "docker restart X")     → command
- "always/never X" or "must/must not X"                          → rule
- "when X, do Y" or "when X happens, ..."                        → skill
- noun describing host/service/person/project                    → entity
- "we chose X because Y" or "decided on X"                       → decision
- session log, reflection, run-insight                           → episodic
- otherwise                                                      → fact
```

`hive_teach` continues to accept `success_hint` separately; it is preserved on the `KnowledgeItem` row alongside the classifier output.

`hive_session_end` extractions also flow through this classifier so session-end facts get typed consistently rather than dumped raw.

### C. Lifecycle states

Every item has a state derived from columns; no separate status enum.

| State | Condition | Behavior |
|---|---|---|
| active | `superseded_by IS NULL AND (expires_at IS NULL OR expires_at > now()) AND confidence >= 0.5` | Eligible for loadout + recall |
| superseded | `superseded_by IS NOT NULL` | Excluded from loadout/recall by default; visible with `include_superseded=true` |
| expired | `expires_at < now()` | Same as superseded |
| low-confidence | `confidence < 0.5` | Excluded from loadout; included in recall with score penalty; flagged in UI |
| deleted | row removed | Hard delete, audit row in `deletion_log` |

### D. Lifecycle-driving processes

1. **On ingest (synchronous):** dedup, classify, layer compute, supersession, contradiction detection. Most contradictions caught here.
2. **Nightly `lifecycle_sweep` (extends `api/app/background.py`):**
    - Decay confidence on items unreached for 90+ days (`-0.05/week`).
    - Auto-expire `episodic` items > 180 days.
    - Run dedup pass (≥ 0.92 cosine on summary embedding within tier+entity).
    - Generate daily digest report → `lifecycle_reports` table → surfaced on `/review` page.
3. **Manual via UI/MCP:**
    - `hive_supersede(old_id, new_id)`
    - `hive_expire(id, reason)` (soft; sets `expires_at = now()`)
    - `hive_promote(id, priority)`
    - All available as MCP admin tools and Cortex UI buttons.

---

## 5. UI & MCP Surface

### A. MCP tool surface

Existing tools preserved. New/changed:

| Tool | Status | Auth | Purpose |
|---|---|---|---|
| `hive_recall` | changed — adds `bucketed`, `tiers`, `expand_layer` | writer+ | Tier-bucketed retrieval |
| `hive_query` | preserved | writer+ | Full-text search (unchanged) |
| `hive_teach` | changed — auto-classifies, preserves `success_hint` | writer+ | Skill ingestion |
| `hive_ingest` | changed — auto-classifies | writer+ | Generic ingest |
| `hive_status` | preserved | reader+ | Health |
| `hive_stats` | preserved | reader+ | Counts |
| `hive_summarize` | preserved | writer+ | Compaction |
| `hive_session_end` | changed — extracted facts route through classifier | writer+ | Session compaction hook |
| `hive_feedback` | preserved | writer+ | Recall relevance training |
| `hive_profile` | preserved | writer+ | Per-agent profile |
| `hive_create_namespace` / `hive_create_key` / `hive_list_keys` | preserved | admin | Namespace mgmt |
| `hive_delete_items` | fixed — admin auth path | admin | Hard delete |
| `hive_dedup` | fixed — admin auth path | admin | Dedup pass |
| `hive_expand` | new | reader+ | Fetch L1/L2 by id |
| `hive_supersede` | new | writer+ | Mark old superseded by new |
| `hive_expire` | new | writer+ | Soft-expire item |
| `hive_promote` | new | writer+ | Bump `loadout_priority` |
| `hive_loadout` | new | reader+ | Compute loadout for (cwd, device); used post-compaction |
| `hive_review_queue` | new | writer+ | List pending review items |
| `hive_guide` | new | reader+ | Fetch agent guide markdown |

### MCP plugin auth resolution (the silent-403 fix)

```python
# ~/.claude/plugins/hari-hive/mcp_server.py
def _api_key_for(role: str) -> str:
    admin_path = Path("~/.claude/agentssot/local/admin.json").expanduser()
    agent_path = Path("~/.claude/agentssot/local/agent.json").expanduser()
    if role == "admin":
        if admin_path.exists():
            return json.loads(admin_path.read_text())["admin_api_key"]
        raise PermissionError(
            "admin operation requires admin.json on this device — contact operator"
        )
    return json.loads(agent_path.read_text())["api_key"]
```

Each MCP tool tagged with required role; the wrapper picks the right key. Devices without `admin.json` (e.g. agent, agenthive) keep working for non-admin tools and return a clear actionable error for admin ones.

### B. Cortex UI changes

Existing 3D Three.js neural map and slide-out drawer preserved. Added:

**Top-level routes**
- `/review` — full Review page (digest + interactive queue)
- `/loadout` — loadout preview tool (operator debugging)
- `/entities` — entity table editor

**Tier-aware browse** in the existing drawer's Browse tab: tier filter chips, entity filter, status filter (active/superseded/expired), confidence slider.

**3D map node coloring by tier:** commands cyan, rules red, skills purple, entities gold, decisions green, episodic dim grey. Visual scan reveals tier balance at a glance.

### `/review` page — daily-digest destination

```
┌─ Hive Review (last 24h: 12 new, 8 resolved, 4 dismissed) ──────┐
│  Filters: [All] [Low-conf] [Duplicates] [Supersession]          │
│           [Contradiction] [Expired]                             │
│  Sort:    [Newest] [Confidence] [Impact]                        │
│                                                                  │
│  Today's digest (from nightly lifecycle_sweep at 03:00 UTC):    │
│   • 4 supersession candidates auto-flagged                       │
│   • 6 dedup candidates (≥ 0.92 sim within tier)                 │
│   • 2 episodic items expired (180d)                              │
│   • 3 items decayed below conf=0.5                               │
│   • 1 contradiction (HIGH priority)                              │
│                                                                  │
│  [Active items — full-width cards with Accept/Edit/Dismiss]     │
│                                                                  │
│  Resolved this week: 23  |  Dismissed: 7  |  Trust score: 94%   │
└──────────────────────────────────────────────────────────────────┘
```

**Trust score** is a rolling 7-day heuristic: `1 − (contradictions_caught / items_used)` from `recall_events`. Drop below 90% surfaces a banner.

The Review page is the trust-rebuilding mechanism: every contradiction, low-confidence type, or duplicate the system spotted shows up here for operator confirmation. Trust returns through observability.

### `/loadout` page — operator debugging

```
[Cwd: /opt/agentssot ▼]  [Device: hari ▼]  [Compute]

Token budget: 612 / 750 used   (15 items shipped, 8 overflow)
Cache key: sha256(...)         (89% hit rate last 7d)

Preview:
=== Hive Loadout (cwd=/opt/agentssot, device=hari) ===
[full loadout text, copy-to-clipboard]
```

Reproduces the exact loadout a session would receive. Critical for diagnosing "why didn't the agent know X?" moments.

### `/entities` page — canonical entity editor

Table view of every Entity row. Columns: slug, type, IPs, cwd_hints, device_hints, linked items count. One-click edit opens a form; saving triggers re-embed of dependent items.

### C. New REST endpoints

```
POST  /api/v1/knowledge/loadout                 body: {cwd, device_id, namespace}
GET   /api/v1/knowledge/items/{id}/expand?layer=full
POST  /api/v1/knowledge/items/{id}/supersede    body: {superseded_by}
POST  /api/v1/knowledge/items/{id}/expire       body: {reason}
POST  /api/v1/knowledge/items/{id}/promote      body: {priority}
GET   /admin/review-queue?namespace=...&kind=low_conf|dup|supersede|contradiction
GET   /admin/lifecycle-report                   (last 7 daily reports)
GET   /agent-guide                              (text/plain markdown, per-key)
```

All authenticated by existing `X-Api-Key` middleware. New endpoints honor existing namespace permissions.

### D. `/agent-guide` — runtime documentation surface

A LAN endpoint any agent on the network can hit, returning text/plain markdown that lands cleanly in any LLM context. Auth required so namespace-aware tailoring is possible.

```
GET http://192.168.1.225:8088/agent-guide
Headers: X-Api-Key: <key>

→ text/plain markdown (~2–3 KB), generated from
  api/app/agent_guide.md.tmpl (Jinja2), per-key tailored:
```

The guide includes:

- **You are connected as:** key role, accessible namespaces, device id, service version
- **Tier model:** when to use each of the 6 tiers, with examples
- **How to recall / how to write:** cheat sheet with concrete tool calls
- **Troubleshooting table:** symptom → likely cause → fix, including 401/403 handling, FTS-vs-recall guidance, connection-refused recovery
- **Connectivity:** service host (hari, 192.168.1.225:8088), LAN-only by design, SSH alias hint, health endpoint
- **Where to look for human help:** operator name, source-of-truth path, web UI URL

The guide is also exposed as MCP tool `hive_guide()` so agents inside an MCP session can fetch it without curl. It is the **anti-rabbit-hole device** — the next time an agent doesn't know how to reach a host or which tool to call, the guide provides exact steps.

Cached 60s. Versioned alongside the API; bumps on every tier-model change so agents pulling it always see current truth.

### E. CLI helper

`api/scripts/hive` becomes a small CLI mirroring the MCP tools:

```bash
hive recall "ssh unraid" --tiers command,rule
hive loadout /opt/agentssot --device hari
hive review                       # interactive review queue walker
hive supersede <old_id> <new_id>
hive expire <id> --reason "moved to nomic-embed-text"
hive entities list
hive entities edit unraid         # opens $EDITOR on the canonical record
hive guide                        # prints /agent-guide content
```

Same auth resolution as the MCP plugin (admin.json if present, fall back to agent.json).

---

## 6. Migration & Testing

### A. Rollout phases

```
Phase 0  Schema additions only (no behavior change)
Phase 1  Read path: tier-bucketed recall (opt-in flag)
Phase 2  Write path: auto-classify + contradiction detection on ingest
Phase 3  Backfill: classify the 4541 existing items
Phase 4  Loadout hook + agent guide endpoint + CLAUDE.md update
Phase 5  Lifecycle sweep + Review/Loadout/Entities pages
Phase 6  Default flips (bucketed=true default, episodic excluded)
Phase 7  Cross-device rollout (plugin admin auth, per-device CLAUDE.md)
```

Each phase is a separate PR. Production stays green throughout. Each phase rolls back independently.

### B. Phase details

**Phase 0 — Schema (1 Alembic migration)**
- Additive columns, indexes, deletion_log table, enum extensions (see §2).
- Rollback: drop new columns/indexes/table.

**Phase 1 — Read path (no data change)**
- `bucketed=true` flag on `/recall` (default false).
- `hive_expand`, `hive_loadout` MCP tools.
- Two-tier reranker (4B procedural / 8B nuanced) with fallback chain.
- `diagnostics` block in response.
- Rollback: revert MCP plugin + API. Existing callers unaffected.

**Phase 2 — Write path (no data change yet)**
- Auto-classify new ingests (`gemma4:31b`).
- L0/L1 layer pre-compute.
- Supersession check on writes.
- Contradiction detector on commands/skills.
- Existing items have NULL `last_classified_at`; classifier only runs on new writes.
- Rollback: revert API. Existing items unaffected.

**Phase 3 — Backfill (the 4541-item event)**

`scripts/backfill_classify.py` — single script, idempotent, resumable, rate-limited.

```
python -m scripts.backfill_classify --batch 200 --rps 5 --namespace claude-shared

1. Pull rows where last_classified_at IS NULL OR < schema_version
2. For each batch:
   - call gemma4:31b classifier (5 rps, 4 parallel workers)
   - persist memory_type, abstract, summary, cwd_hints, device_hints, confidence
   - if confidence ≥ 0.6 AND existing memory_type ∈ {NULL, fact}: update memory_type
   - if confidence < 0.6: keep existing memory_type, enqueue Review Queue
   - regenerate embedding from L1 summary (HNSW reindex on changed rows only)
   - mark last_classified_at = now()
3. After all rows classified:
   - resolve entity_mentions → entity_refs (insert into Entity table if missing)
   - run supersession sweep within (entity, type) groups
   - run contradiction sweep (rules-vs-commands/skills per entity)
4. Run dedup sweep (≥ 0.92 cosine on summary embed within tier+entity)
```

Capacity: 4541 items × 200ms classify = ~15 min serial; ~4 min wall time at 5 rps × 4 workers; ~3 min embed regen; ~10 min total.

**Pre-backfill snapshot:** `pg_dump` taken automatically by the script, retained 30 days.

**Verification gate:** script outputs a distribution report. Operator reviews before Phase 4. If distribution looks wrong (e.g. >80% episodic, very few commands), tune classifier prompt and re-run.

**Phase 4 — Loadout hook + agent guide + CLAUDE.md**
- Replace SessionStart hint with full loadout in `~/.claude/plugins/hari-hive/hooks/`.
- Ship `/agent-guide` endpoint and Jinja2 template.
- Update `~/.claude/CLAUDE.md` first-turn protocol: loadout already loaded → step 3 (`hive_recall on keywords`) becomes optional supplement rather than mandatory. Add post-compaction recovery line: "if context was compacted, call `hive_loadout()` to restore push context."
- Rollback: revert hook to old hint. Endpoint stays (harmless).

**Phase 5 — Lifecycle sweep + Cortex pages**
- Nightly `lifecycle_sweep` job in `api/app/background.py`.
- Cortex `/review`, `/loadout`, `/entities` routes.
- Tier color-coding on existing 3D map.
- Rollback: disable cron, remove routes. Data unchanged.

**Phase 6 — Default flips**
- `bucketed=true` becomes default on `/recall`.
- Episodic excluded by default.
- Pre-flip: grep codebase for `/recall` callers; ≤5 expected (api/app/ui, MCP plugin, scripts). Update each.
- Rollback: flip defaults back.

**Phase 7 — Cross-device rollout**
- Push updated MCP plugin to all enrolled devices: hari, dockers, webvm, blink, air, agent, agenthive (zoria deferred — DNS unreachable).
- Reuse `~/.claude/agentssot/scripts/push_keys_ssh.sh` pattern for plugin sync.
- Update CLAUDE.md per device.
- Test loadout per device: `hive loadout <cwd> --device <id>`.
- Devices without admin.json keep working for non-admin tools; admin-tool calls return clear error.
- Rollback: SSH-revert plugin per device, independent.

### C. Testing strategy

```
api/tests/
  unit/
    test_classifier.py        gemma4:31b classifier on golden set, fail < 90%
    test_layer_compute.py     abstract ≤ 50 tok, summary ≤ 500 tok
    test_supersession.py      same-type collision detection
    test_contradiction.py     rule-vs-command negation detection
    test_loadout_budget.py    token budget honored, priority sort correct
    test_cwd_resolver.py      cwd patterns map to expected entities
  integration/
    test_recall_bucketed.py   bucketed=true returns expected shape
    test_recall_compat.py     bucketed=false returns flat (existing callers)
    test_ingest_pipeline.py   full ingest → classify → supersede → embed
    test_admin_auth.py        admin.json fallback, writer 403s with clear msg
    test_loadout_endpoint.py  POST /loadout for known cwd returns expected items
    test_agent_guide.py       /agent-guide returns per-key tailored markdown
  golden/
    classifier_corpus.jsonl   50 hand-labelled items (all 6 tiers + edges)
    loadout_snapshots.jsonl   known (cwd, device) → expected loadout
    recall_snapshots.jsonl    known queries → expected top-3 per bucket
  smoke/
    test_post_backfill.py     post-backfill: distribution sanity, no orphan
                              superseded chains, all entities have ≥1 referent
```

**Golden classifier set** is hand-curated by the operator (50 items spanning all 6 tiers and edge cases). Classifier prompt iterates against this set until ≥ 90% accuracy, then becomes a regression gate. Any prompt or model change must hold ≥ 90%.

Per-phase smoke scripts in `scripts/smoke_phaseN.sh`.

### D. Trust verification (the cracking-trust problem)

Three tripwires that catch silent corruption going forward:

1. **Daily lifecycle digest** posted to `/review` (and stored in `lifecycle_reports`). Decayed-without-supersession items, unflagged dups, orphaned entity refs all surface here.
2. **Recall regression alarm:** `recall_events` already exists. Alarm fires if `feedback.signal=wrong` exceeds 5% of recalls in 24h. Surfaces in Cortex banner.
3. **Loadout snapshot diffing:** weekly snapshot of every device's loadout. Diffs >30% item churn get logged. Catches stale rules silently displacing useful commands.

### E. Estimated effort

| Phase | Work | Owner |
|---|---|---|
| 0 — Schema | 1 day | coder via chain.sh |
| 1 — Read path | 3 days | coder + operator (classifier prompt) |
| 2 — Write path | 3 days | coder |
| 3 — Backfill | 1–2 days | mostly Ollama, distribution review by operator |
| 4 — Loadout + guide | 2 days | coder + designer |
| 5 — Lifecycle + UI | 4 days | designer-heavy |
| 6 — Default flips | 1 day | coder |
| 7 — Cross-device rollout | 1 day per device, parallelizable | scripted |

Total: ~3–4 calendar weeks at sustainable pace. Critical path is Phases 1–3.

---

## Appendix A — Existing schema features being activated

These already exist in `api/app/models.py` and are largely unused by the retrieval/MCP surface. The overhaul activates them:

- `MemoryType` enum (line 53): retrieval never filters by it currently.
- `MemoryCategory` enum (line 68): never written.
- `ContentLayer` enum (line 83): every item is `full`; abstract/summary unused.
- `Entity` table (line 135): exists but barely populated.
- `Concept` synthesis pipeline: working; preserved with model bump.
- `RecallEvent` and `ConceptFeedback`: working; informs `Concept.confidence`.

The overhaul is largely about wiring these into ingestion, retrieval, and UI — not building from scratch.

## Appendix B — Cleanup landmines noted during diagnosis

- Stale `openclaw/identity/system.md` declaring `unraid` "OFF LIMITS" — deleted 2026-04-24 (`d0bccdb9-9989-4ac6-96cc-83aa096a63a2`).
- ~15 `openclaw/skills/*` items referencing `~/.openclaw/skills/...` paths that don't exist on this machine — left in place, will be flagged by Phase 5 dedup/expiration sweep.
- MCP plugin auth bug: `hive_delete_items` and `hive_dedup` silently 403 because plugin reads writer key only — fixed in Phase 7 (or earlier as part of MCP plugin update).

## Appendix C — Model assignments

| Function | Model | Notes |
|---|---|---|
| Embedding | `nomic-embed-text` (768d) | unchanged; CPU-only per recent commit `733b13f` |
| Reranker (procedural) | `dengcao/Qwen3-Reranker-4B:Q4_K_M` | new fast path |
| Reranker (nuanced) | `dengcao/Qwen3-Reranker-8B:Q8_0` | preserved |
| Classifier (ingest auto-typing) | `gemma4:31b` | local |
| Layer pre-compute (abstract/summary) | `gemma4:31b` | local |
| Synthesis (daily) | `qwen3.6:27b` (or `kimi-k2.6:cloud` if longer context wanted) | bump from `qwen3:latest` |
| LLM (retrieval / MCP responses) | `qwen3:latest` | unchanged |

All Ollama-resident or Ollama-cloud. No Anthropic tokens in the write/classify/rerank/synthesis paths.
