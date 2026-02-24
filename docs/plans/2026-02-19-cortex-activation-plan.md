# Cortex Activation — Making Concepts Actually Useful

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the loop so synthesized concepts surface in every session automatically, creating a real short-term ← → long-term memory cycle.

**Architecture:** Add blended "all" scope to recall, update MCP plugin defaults, fix decay mechanics, lower clustering thresholds, and surface concepts in hive_stats.

**Tech Stack:** Python (FastAPI, SQLAlchemy), MCP plugin (mcp_server.py), bash hooks

---

## Problem Summary

Concepts are synthesized but never read. The MCP plugin defaults `scope="knowledge"`, no "all" scope exists, and the SessionStart hook doesn't mention concepts. This means zero sessions benefit from synthesis — the GPU cycles are wasted.

---

### Task 1: Add "all" scope to recall API

Blended recall searches knowledge + concepts in one call, merges results by vector score, and lets the reranker sort them together. This is the critical unlock.

**Files:**
- Modify: `api/app/schemas.py` (add "all" to Literal types)
- Modify: `api/app/crud.py` (add "all" branch in recall())

**Step 1: Update schemas**

In `api/app/schemas.py`, update the `Literal` type in 3 places:

```python
# RecallRequest.scope (line ~79)
scope: Literal["knowledge", "requirements", "events", "concepts", "all"] = "knowledge"

# RecallItem.scope (line ~89)
scope: Literal["knowledge", "requirements", "events", "concepts"]
# ^^^ leave this unchanged — individual items still have their specific scope

# RecallResponse.scope (line ~101)
scope: Literal["knowledge", "requirements", "events", "concepts", "all"]
```

**Step 2: Add "all" branch in crud.py recall()**

Insert this block BEFORE the final `raise HTTPException` at line ~504 of `api/app/crud.py`:

```python
if payload.scope == "all":
    # --- knowledge items ---
    ki_score = KnowledgeItem.embedding.cosine_distance(query_embedding).label("score")
    ki_stmt = (
        select(KnowledgeItem, ki_score)
        .where(KnowledgeItem.namespace == payload.namespace)
        .where(KnowledgeItem.embedding.is_not(None))
        .order_by(ki_score)
        .limit(candidate_k)
    )
    if project_id:
        ki_stmt = ki_stmt.where(KnowledgeItem.project_id == project_id)
    if entity_id:
        ki_stmt = ki_stmt.where(KnowledgeItem.entity_id == entity_id)

    ki_rows = session.execute(ki_stmt).all()
    items = [
        {
            "id": str(item.id),
            "scope": "knowledge",
            "score": float(score_value),
            "snippet": _clip(item.content, settings.max_snippet_chars),
            "tags": list(item.tags or []),
            "created_at": item.created_at,
        }
        for item, score_value in ki_rows
    ]

    # --- concepts ---
    c_score = Concept.embedding.cosine_distance(query_embedding).label("score")
    c_stmt = (
        select(Concept, c_score)
        .where(Concept.namespace == payload.namespace)
        .where(Concept.embedding.is_not(None))
        .where(~Concept.tags.any("superseded"))
        .order_by(c_score)
        .limit(candidate_k)
    )

    c_rows = session.execute(c_stmt).all()
    items.extend([
        {
            "id": str(item.id),
            "scope": "concepts",
            "score": float(score_value),
            "snippet": _clip(f"[{item.type.value}] {item.title}: {item.content}", settings.max_snippet_chars),
            "tags": list(item.tags or []),
            "created_at": item.created_at,
            "concept_type": item.type.value,
            "confidence": item.confidence,
        }
        for item, score_value in c_rows
    ])

    # merge by vector score (lower = closer for cosine_distance)
    items.sort(key=lambda x: x["score"])
    items = items[:candidate_k]

    return _apply_reranker(payload.query_text, items, top_k, reranker_provider)
```

**Step 3: Rebuild and test**

```bash
docker compose up -d --build api
# Test blended recall:
curl -s -X POST http://localhost:8088/recall \
  -H "X-API-Key: $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"query_text":"how does the agent architecture work","namespace":"claude-shared","top_k":5,"scope":"all"}'
```

Expected: mix of knowledge items and concepts in results, sorted by relevance.

**Step 4: Commit**

```bash
git add api/app/schemas.py api/app/crud.py
git commit -m "feat: add blended 'all' scope to recall — knowledge + concepts in one query"
```

---

### Task 2: Update MCP plugin — default to "all", surface concepts

The plugin is the interface every session uses. It must default to blended recall and display concept metadata.

**Files:**
- Modify: `~/.claude/plugins/hari-hive/mcp_server.py`

**Step 1: Change hive_recall default scope**

```python
# Line ~85: change default from "knowledge" to "all"
async def hive_recall(
    query: str,
    namespace: str = "",
    scope: str = "all",           # was "knowledge"
    top_k: int = 5,
) -> str:
    """Semantic (vector) recall from hari-hive memory. Returns the most relevant items by embedding similarity.

    Args:
        query: Natural-language search query.
        namespace: Namespace to search (default: claude-shared).
        scope: Scope filter -- knowledge, requirements, events, concepts, or all (default: all, blends knowledge + concepts).
        top_k: Max results to return.
    """
```

**Step 2: Update _fmt_recall_item to show concept metadata**

Replace the `_fmt_recall_item` function:

```python
def _fmt_recall_item(item: dict, idx: int) -> str:
    score = item.get("score", 0)
    rs = item.get("reranker_score")
    tags = ", ".join(item.get("tags", []))
    score_str = f"vec={score:.3f}"
    if rs is not None:
        score_str += f" rerank={rs:.3f}"
    snippet = item.get("snippet", "")
    created = item.get("created_at", "")
    item_scope = item.get("scope", "knowledge")

    header = f"[{idx}] ({score_str}) [{tags}] {created}"
    if item_scope == "concepts":
        ctype = item.get("concept_type", "?")
        conf = item.get("confidence", 0)
        header = f"[{idx}] CONCEPT ({ctype}, conf={conf:.2f}) ({score_str}) [{tags}]"
    return f"{header}\n{snippet}"
```

**Step 3: Update hive_stats to include concepts**

In the `hive_stats` function, change the iteration from:
```python
for section in ("knowledge_items", "requirements", "events"):
```
to:
```python
for section in ("knowledge_items", "requirements", "events", "concepts"):
```

**Step 4: Test**

```bash
# Restart Claude Code to pick up plugin changes, then:
# Use hive_recall with default scope — should return blended results
# Use hive_stats — should show concept counts
```

**Step 5: Commit (plugin is outside the main repo, just note the change)**

---

### Task 3: Update SessionStart hook — mention concepts

The hook should tell the model that concepts exist and that default recall now blends them.

**Files:**
- Modify: `~/.claude/plugins/hari-hive/hooks/SessionStart.md`

**Replace the bash script content:**

```bash
#!/bin/bash
PROJECT_NAME=$(basename "$(pwd)")

echo "<hive-available>"
echo "You have access to hari-hive (AgentSSOT) via MCP tools."
echo "Use hive_recall for semantic search (blends knowledge + synthesized concepts by default)."
echo "Use hive_query for text/tag search. Use hive_ingest to store knowledge."
echo "Concepts are long-term patterns extracted from your knowledge base — they surface automatically in recall."
echo "Current project: ${PROJECT_NAME}. Default namespace: claude-shared."
echo "Fetch context on demand — do NOT pre-load everything."
echo "</hive-available>"
```

Token cost: ~90 tokens (up from ~80). Worth the 10 extra tokens.

---

### Task 4: Fix decay — time-based, not run-based

Current problem: every synthesis run (including manual) decays ALL untouched concepts by -0.05. Manual debugging runs accidentally age your memory. Concepts die in ~15 days even if nothing contradicts them.

Fix: decay based on time since last reinforcement, not per-run. Only decay on scheduled runs, not manual triggers.

**Files:**
- Modify: `api/app/synthesis/reconciler.py`
- Modify: `api/app/synthesis/loop.py`
- Modify: `api/app/main.py`

**Step 1: Make decay time-aware in reconciler.py**

Replace the `decay_stale_concepts` function:

```python
def decay_stale_concepts(
    session: Session,
    namespace: str,
    active_concept_ids: set[UUID],
    decay_rate: float = 0.05,
    min_age_days: int = 7,
) -> int:
    """Reduce confidence of concepts not reinforced recently. Returns count decayed.

    Only decays concepts whose updated_at is older than min_age_days,
    preventing aggressive decay of recently-created or recently-reinforced concepts.
    """
    from datetime import UTC, datetime, timedelta

    cutoff = datetime.now(UTC) - timedelta(days=min_age_days)

    stmt = (
        select(Concept)
        .where(Concept.namespace == namespace)
        .where(~Concept.tags.any("superseded"))
        .where(Concept.confidence > 0.1)
        .where(Concept.updated_at < cutoff)  # only decay old concepts
    )
    all_active = session.scalars(stmt).all()

    decayed = 0
    for concept in all_active:
        if concept.id not in active_concept_ids:
            concept.confidence = max(concept.confidence - decay_rate, 0.0)
            decayed += 1

            if concept.confidence <= 0.1:
                concept.tags = list(set(concept.tags or []) | {"superseded"})
                concept.embedding = None

    if decayed:
        session.commit()
    return decayed
```

**Step 2: Skip decay on manual synthesis runs**

In `api/app/synthesis/loop.py`, add a `skip_decay` parameter to `_run_synthesis_for_namespace`:

```python
def _run_synthesis_for_namespace(
    namespace: str,
    settings,
    llm_provider: LLMProvider,
    embedding_provider: EmbeddingProvider,
    full_resynthesis: bool = False,
    skip_decay: bool = False,          # NEW
) -> dict:
```

And wrap the decay call:

```python
    if not skip_decay:
        decayed = decay_stale_concepts(
            session, namespace, all_touched_ids, settings.synthesis_confidence_decay
        )
        stats["decayed"] = decayed
```

**Step 3: Pass skip_decay=True for manual triggers**

In `api/app/main.py`, the `admin_trigger_synthesis` endpoint:

```python
stats = _run_synthesis_for_namespace(
    namespace=namespace,
    settings=settings,
    llm_provider=llm_provider,
    embedding_provider=embedding_provider,
    full_resynthesis=full,
    skip_decay=True,   # manual runs don't decay
)
```

**Step 4: Rebuild and test**

```bash
docker compose up -d --build api
# Manual synthesis should show decayed_concepts=0
curl -X POST "http://localhost:8088/admin/synthesize?namespace=claude-shared" -H "X-API-Key: $ADMIN_KEY"
```

**Step 5: Commit**

```bash
git add api/app/synthesis/reconciler.py api/app/synthesis/loop.py api/app/main.py
git commit -m "fix: time-based decay, skip decay on manual synthesis"
```

---

### Task 5: Lower clustering thresholds — capture more patterns

Current: `similarity_threshold=0.75`, `min_cluster_size=3`. From 832 items, only 7 concepts were extracted (0.8%). Unique insights that don't repeat 3+ times are invisible.

**Files:**
- Modify: `api/app/settings.py` (change defaults)
- Modify: `.env` (update running config)

**Step 1: Update settings defaults**

```python
# In settings.py, change:
synthesis_similarity_threshold: float = Field(default=0.65, alias="SYNTHESIS_SIMILARITY_THRESHOLD")  # was 0.75
synthesis_min_cluster_size: int = Field(default=2, alias="SYNTHESIS_MIN_CLUSTER_SIZE")  # was 3
```

**Step 2: Update .env**

```bash
# Append or update in .env:
SYNTHESIS_SIMILARITY_THRESHOLD=0.65
SYNTHESIS_MIN_CLUSTER_SIZE=2
```

**Step 3: Run full resynthesis to capture more patterns**

```bash
docker compose up -d --build api
curl -X POST "http://localhost:8088/admin/synthesize?namespace=claude-shared&full=true" -H "X-API-Key: $ADMIN_KEY"
```

Expected: significantly more concepts (target: 20-40 from 832 items).

**Step 4: Commit**

```bash
git add api/app/settings.py
git commit -m "tune: lower clustering thresholds to capture more conceptual patterns"
```

---

### Task 6: Clean up dead code + minor fixes

**Files:**
- Modify: `api/app/settings.py` (remove unused synthesis_batch_size)

**Step 1: Remove unused setting**

Delete:
```python
synthesis_batch_size: int = Field(default=20, alias="SYNTHESIS_BATCH_SIZE")
```

Also remove `SYNTHESIS_BATCH_SIZE` from `.env` if present.

**Step 2: Commit**

```bash
git add api/app/settings.py
git commit -m "clean: remove unused synthesis_batch_size setting"
```

---

## Execution Order

Tasks 1-3 are the critical path (make concepts visible). Task 4 prevents data loss. Task 5 improves quality. Task 6 is cleanup.

Recommended: 1 → 2 → 3 → 4 → 5 → full resynthesis → 6

## Success Criteria

After all tasks:
1. `hive_recall("device routing")` returns a mix of knowledge items AND concepts (blended)
2. `hive_stats` shows concept counts
3. SessionStart mentions concepts exist
4. Manual synthesis shows `decayed_concepts: 0`
5. Full resynthesis produces 20+ concepts (vs current 7)
6. Concepts older than 7 days that aren't reinforced gradually decay; fresh concepts are protected
