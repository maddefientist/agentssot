# Layer 2: Feedback Loop — Design Document

## Mission

Build a persistent intelligence layer beneath any AI model that makes every interaction smarter than the last. The feedback loop teaches the cortex to distinguish signal from noise — concepts that prove useful in real work get reinforced, concepts flagged as incorrect get reviewed, and dormant concepts sleep indefinitely until needed again. All learning runs on local/cloud Ollama models to avoid burning Claude tokens.

## Goals

1. **Learn from outcomes** — concepts that help solve problems get stronger
2. **Three-signal feedback** — useful, noted, wrong (not binary thumbs up/down)
3. **Zero Claude token overhead for learning** — synthesis, extraction, and feedback processing all run on Ollama
4. **Fuzzy correction** — operator can flag wrong concepts by description, not just ID
5. **Long memory** — concepts go dormant, never die. Resurrection on re-engagement.
6. **Model agnostic** — feedback works across any enrolled agent/model

## Architecture

### Data Model

**`recall_events`** — implicit signal (auto-logged on every recall)

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| concept_id | UUID FK → concepts | Which concept was surfaced |
| namespace | TEXT | |
| session_id | TEXT | Ties events to a session |
| agent_key | TEXT | Which agent/model made the recall |
| query_text | TEXT | What was asked |
| score | FLOAT | Vector/rerank score at recall time |
| session_completed | BOOL default false | Set true at session end |
| created_at | TIMESTAMPTZ | |

**`concept_feedback`** — explicit signal (operator-driven)

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| concept_id | UUID FK → concepts | Resolved concept (by ID or fuzzy match) |
| namespace | TEXT | |
| signal | ENUM(useful, noted, wrong) | Three-way signal |
| agent_key | TEXT | Which agent recorded this |
| session_id | TEXT nullable | |
| note | TEXT nullable | Correction context or reason |
| created_at | TIMESTAMPTZ | |

### Signal Model

| Signal | Meaning | Confidence effect | Behavior |
|--------|---------|-------------------|----------|
| **useful** | "This helped solve my task" | +0.15 (cap +0.30/cycle) | Strong reinforcement |
| **noted** | "Good reminder, not for now" | +0.05 (cap +0.10/cycle) | Keeps concept alive |
| **wrong** | "This is outdated/incorrect" | No penalty | Tags `contested`, synthesis LLM reviews |
| **implicit recall** | Concept surfaced + session completed | +0.02 (cap +0.10/cycle) | Weak background signal |

**Key principle:** User feedback never directly penalizes confidence. Only the synthesis LLM can supersede a concept after reviewing evidence including "wrong" flags.

### Decay Model (revised)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Grace period | 90 days | Projects go dormant for months |
| Decay rate | -0.02/cycle | Slow erosion, not aggressive pruning |
| Decay floor | 0.15 confidence | Concepts go dormant, never die |
| Feedback protection | 6 months | Any positive signal in window skips decay |
| Resurrection | Jump to 0.5 | Dormant concept + positive feedback = full revival |

### MCP Tool: `hive_feedback`

```
hive_feedback(
    signal: "useful" | "noted" | "wrong",
    concept_id: str = None,      # direct reference by ID
    query: str = None,            # fuzzy match via semantic search
    note: str = None              # correction context or reason
)
# Must provide concept_id OR query (not both required, but at least one)
```

**Fuzzy match flow (query mode):**
1. Embed the query text
2. Search concepts by vector similarity (top 3)
3. If top match score > threshold, attach feedback to it
4. If "wrong" signal, also ingest the `note` as a new knowledge item tagged `["correction", "operator-feedback"]`
5. Return matched concept title + updated confidence

~20 Claude tokens per call.

### SessionEnd Migration to Ollama

**Current:** SessionEnd hook prompts Claude to extract facts → burns Claude tokens every session.

**New:** `POST /session-complete` API endpoint

**Input:**
```json
{
  "session_id": "...",
  "conversation_summary": "...",
  "recalled_concept_ids": ["uuid1", "uuid2"]
}
```

**Server-side (zero Claude tokens):**
1. Run fact extraction via Ollama qwen3:latest (local, fast, <2s)
2. Ingest extracted facts with tags `["session-extract", "device-{name}"]`
3. Mark matching recall_events as `session_completed = true`
4. Return extraction count

**MCP hook:** Single tool call `hive_session_end(summary)` → ~30 tokens.

### Ollama Model Routing

| Task | Model | Why |
|------|-------|-----|
| Fact extraction (SessionEnd) | qwen3:latest (local 8B) | Fast, structured work, no network latency |
| Synthesis (concept generation) | kimi-k2.5:cloud / qwen3.5:cloud | Needs reasoning depth |
| Synthesis fallback | qwen3:latest (local) | When cloud times out |
| Contested concept review | qwen3.5:cloud | Needs judgment for supersede decisions |

### Claude Token Budget Per Session

```
SessionStart hint:          ~50 tokens (existing, unchanged)
hive_feedback calls:        ~20 tokens each (only when explicitly used)
SessionEnd tool call:       ~30 tokens (one call, Ollama does extraction)
All learning/synthesis:       0 Claude tokens
                            ─────────
Overhead per session:       ~100 tokens baseline
```

### Reconciler Integration

At synthesis time, before decay:

1. Query `concept_feedback` since last synthesis, group by concept_id
2. Query `recall_events` where `session_completed = true`, group by concept_id
3. For each concept:
   - Apply explicit feedback confidence adjustments (per-session caps)
   - Apply implicit recall reinforcement (per-session caps)
   - If any "wrong" flags: tag concept `contested`, include flags + notes in synthesis prompt
   - If any positive feedback in last 6 months: skip decay
4. Run decay on remaining unprotected concepts older than 90 days
5. Synthesis LLM reviews `contested` concepts with correction notes, may supersede

### Cortex Visualization Data (future)

The new tables provide all signals needed for live visualization:
- `recall_events` = synapse firing (pulse from agent to concept)
- `concept_feedback` useful = green flash on node
- `concept_feedback` noted = amber flash on node
- `concept_feedback` wrong = red flag on node
- Confidence changes = node size/brightness shifts
- Resurrection = dormant node relighting

## Implementation Scope

1. Alembic migration: `recall_events` + `concept_feedback` tables + enum
2. SQLAlchemy models for both tables
3. CRUD functions: log_recall_event, create_feedback, get_feedback_summary
4. `POST /session-complete` endpoint (Ollama-powered extraction)
5. `hive_feedback` MCP tool (with fuzzy match)
6. `hive_session_end` MCP tool (replaces current SessionEnd hook)
7. Reconciler updates: feedback-aware confidence, decay protection, contested review
8. Update SessionEnd hook to use new MCP tool
9. Update decay parameters (90 day grace, -0.02 rate, 0.15 floor)
