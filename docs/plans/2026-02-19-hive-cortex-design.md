# Hive Cortex: Conceptual Memory Layer

**Date:** 2026-02-19
**Status:** Approved
**Author:** Hari + Madi

## Problem

Hive stores flat knowledge items and session events. Compaction summarizes individual sessions, but nothing synthesizes patterns *across* sessions over time. There's no mechanism for building conceptual understanding - mental models, learned principles, or relationship maps - that grows with experience.

## Solution

A new **Concept** data type and **daily synthesis loop** that uses a local LLM (Qwen3 30B-A3B MoE) to review accumulated knowledge, extract patterns, and maintain evolving conceptual memory. Concepts integrate seamlessly into existing recall via vector search.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage | Dedicated `Concept` table | Clean versioning, evidence linking, confidence scores - can't do this well with tags on KnowledgeItems |
| Recall | Blended into existing `/recall` | Zero changes for callers, concepts surface naturally when relevant |
| LLM | Qwen3 30B-A3B (MoE, local) | Strong reasoning at ~6GB active VRAM. Runs alongside embedding model |
| Schedule | Daily (3 AM default) | Good balance of freshness vs GPU cost |
| Evolution | Versioned with parent chain | Full audit trail, only latest version in recall index |

## Data Model

### Concept Table

```sql
CREATE TABLE concepts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace VARCHAR NOT NULL REFERENCES namespaces(name),
    type VARCHAR NOT NULL,          -- mental_model | relationship | principle
    scope VARCHAR NOT NULL DEFAULT 'global',  -- global | project | device
    scope_ref VARCHAR,              -- project slug or device name when scoped
    title VARCHAR NOT NULL,
    content TEXT NOT NULL,
    evidence_ids UUID[] DEFAULT '{}',  -- refs to source KnowledgeItems/Events
    confidence FLOAT DEFAULT 0.5,   -- 0.0-1.0, grows with corroboration
    version INT DEFAULT 1,
    parent_id UUID REFERENCES concepts(id),  -- previous version
    tags VARCHAR[] DEFAULT '{}',
    embedding vector(4096),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_concepts_namespace ON concepts(namespace);
CREATE INDEX idx_concepts_type ON concepts(type);
CREATE INDEX idx_concepts_scope ON concepts(scope, scope_ref);
CREATE INDEX idx_concepts_embedding ON concepts USING hnsw (embedding vector_cosine_ops);
```

### Concept Types

| Type | Description | Example |
|------|-------------|---------|
| `mental_model` | Learned heuristics about preferences, workflows, patterns | "Hari prefers FastAPI for internal APIs, Next.js for user-facing" |
| `relationship` | Entity-to-entity connections that grow over time | "project agentssot depends on Ollama for embeddings, reranking, and synthesis" |
| `principle` | Distilled wisdom that evolves with experience | "Docker healthchecks on hari need relaxed intervals (30s+) due to resource contention" |

### Scope

| Scope | When | Example |
|-------|------|---------|
| `global` | Applies everywhere | "Always use Cloudflare tunnels for external exposure" |
| `project` | Specific to a project (scope_ref = project slug) | "agentssot UI changes require container rebuild" |
| `device` | Specific to a host (scope_ref = device name) | "hari GPU can run 8B models concurrently but needs model swaps for 30B+" |

## Synthesis Loop

### Overview

Runs as a background task alongside existing compaction loop. Configurable schedule (default: daily 3 AM).

### Pipeline

```
1. GATHER
   Query knowledge items created since last synthesis run.
   Filter: namespace, created_at > last_run_timestamp
   Exclude items already referenced in existing concept evidence_ids.
   Also load all active (non-superseded) concepts for reconciliation.

2. CLUSTER
   Group new items by semantic similarity using embeddings.
   For each cluster, also find existing concepts within cosine distance threshold.
   Goal: batches of related facts that might form or reinforce a concept.

3. SYNTHESIZE
   Feed each cluster to Qwen3 30B-A3B with structured prompt.

   Input: batch of knowledge item contents + any related existing concepts
   Output: JSON array of concept proposals:
     { type, scope, scope_ref, title, content, confidence,
       matches_existing_concept_id, evidence_item_ids }

4. RECONCILE
   For each proposed concept:
   - No match to existing → INSERT new concept (version=1, confidence from LLM)
   - Reinforces existing → UPDATE: bump confidence, append evidence_ids,
     increment version, update content if refined
   - Contradicts existing → Create new version (parent_id = old),
     reduce old concept's confidence, tag old as 'superseded'

5. EMBED & STORE
   Generate embeddings for new/updated concept content.
   Upsert to DB. Mark superseded concepts to exclude from recall.
```

### Synthesis Prompt Strategy

The LLM receives structured input:

```
You are a knowledge synthesis engine. Review these facts and identify:
1. Mental models - recurring patterns, preferences, heuristics
2. Relationships - connections between entities (projects, hosts, tools, people)
3. Principles - actionable wisdom derived from experience

For each concept, provide:
- type: mental_model | relationship | principle
- scope: global | project | device
- scope_ref: (if scoped, the project/device name)
- title: concise label (under 80 chars)
- content: full description (2-5 sentences)
- confidence: 0.0-1.0 based on strength of evidence
- evidence_summary: why you believe this

If any facts reinforce or contradict existing concepts (provided below),
note the concept ID and whether it's reinforcement or contradiction.

=== NEW FACTS ===
{batch of knowledge items}

=== EXISTING CONCEPTS ===
{related concepts from vector search}
```

### Configuration

```env
# Synthesis settings
SYNTHESIS_ENABLED=true
SYNTHESIS_SCHEDULE=0 3 * * *          # cron expression, default 3 AM daily
SYNTHESIS_MODEL=qwen3:30b-a3b         # Ollama model for synthesis
SYNTHESIS_BATCH_SIZE=20               # items per LLM call
SYNTHESIS_SIMILARITY_THRESHOLD=0.75   # cosine distance for clustering
SYNTHESIS_MIN_CLUSTER_SIZE=3          # minimum items to form a concept
SYNTHESIS_CONFIDENCE_DECAY=0.05       # per-run decay if concept not reinforced
```

## Recall Integration

### Changes to /recall

Minimal. The recall pipeline already:
1. Embeds query
2. Searches by cosine distance across tables
3. Reranks with cross-encoder

Add `concepts` table to the vector search union. Concept results include:
- `kind: "concept"` (alongside existing `knowledge_item`, `event`, `requirement`)
- `concept_type: "mental_model" | "relationship" | "principle"`
- `confidence: float`
- `version: int`

Filter: exclude concepts tagged `superseded`.

### No boost needed

Concepts will naturally rank well because:
- Their content is dense and semantically rich (synthesized, not raw)
- The reranker will score them highly when they're relevant to the query
- If they're not relevant, they won't surface - which is correct

## Version History

```
Concept v1 (created 2026-02-20)
  "Docker on hari needs extended healthcheck intervals"
  confidence: 0.6, evidence: [item_a, item_b]

  └─ Concept v2 (updated 2026-02-27, parent_id = v1)
     "Docker on hari needs 30s+ healthcheck intervals due to GPU memory pressure"
     confidence: 0.8, evidence: [item_a, item_b, item_c, item_d]

     └─ Concept v3 (updated 2026-03-10, parent_id = v2)
        "Docker on hari needs 30s+ healthcheck intervals; services sharing GPU
         should stagger startup to avoid OOM"
        confidence: 0.9, evidence: [item_a, item_b, item_c, item_d, item_e, item_f]
```

Only v3 has an active embedding and appears in recall. v1 and v2 are tagged `superseded` and retained for audit.

## Confidence Mechanics

- **Initial:** Set by synthesis LLM based on evidence strength (typically 0.4-0.7)
- **Reinforcement:** +0.1 per run that corroborates (capped at 1.0)
- **Contradiction:** New version created, old confidence reduced by 0.2
- **Decay:** -0.05 per synthesis run with no new evidence (floor at 0.1)
- **Pruning:** Concepts below 0.1 confidence for 30+ days auto-archived

## API Endpoints

### New endpoints

```
GET  /concepts                  — List concepts (filter by namespace, type, scope)
GET  /concepts/{id}             — Get concept with version history
GET  /concepts/{id}/evidence    — Get source items that formed this concept
POST /concepts/synthesize       — Trigger manual synthesis run (admin only)
```

### Modified endpoints

```
POST /recall  — Now also searches concepts table (no API change needed)
GET  /stats   — Include concept counts by type, scope, avg confidence
```

## Files to Create/Modify

### New files
- `api/app/models.py` — Add Concept model
- `api/app/synthesis/` — New package
  - `__init__.py`
  - `clustering.py` — Semantic clustering logic
  - `synthesizer.py` — LLM prompt construction and parsing
  - `reconciler.py` — Match proposals to existing concepts, handle versioning
  - `loop.py` — Background loop (schedule, gather, orchestrate)

### Modified files
- `api/app/crud.py` — Add concept CRUD, extend recall to include concepts
- `api/app/schemas.py` — Add Concept request/response schemas
- `api/app/main.py` — Add concept endpoints, register synthesis loop
- `api/app/settings.py` — Add synthesis config vars
- `api/app/background.py` — Register synthesis loop alongside compaction
- `docker-compose.yml` — Add synthesis env vars
- `.env` — Add synthesis config defaults

## Non-Goals (for v1)

- Graph visualization UI (future: add to web dashboard)
- Cross-namespace concept sharing (each namespace synthesizes independently)
- Real-time synthesis (daily batch is sufficient)
- Concept deletion via API (only via confidence decay + auto-archive)
- MCP tool exposure (future: `hive_concepts` tool for direct querying)
