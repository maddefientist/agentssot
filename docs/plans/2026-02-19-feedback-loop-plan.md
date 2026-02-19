# Layer 2: Feedback Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a feedback loop so the cortex learns from outcomes — concepts that help get reinforced, concepts flagged wrong get reviewed, and all learning runs on Ollama (zero Claude tokens).

**Architecture:** Two new DB tables (recall_events, concept_feedback) capture implicit and explicit signals. A new `hive_feedback` MCP tool lets operators rate concepts by ID or fuzzy description. SessionEnd fact extraction migrates from Claude to Ollama via a new `/session-complete` endpoint. The reconciler integrates feedback into confidence scoring with 90-day decay grace and concept resurrection.

**Tech Stack:** Python/FastAPI, SQLAlchemy, PostgreSQL, Ollama (qwen3:latest for extraction, qwen3.5:cloud for synthesis), pgvector for fuzzy concept matching.

---

### Task 1: Database Models — recall_events and concept_feedback

**Files:**
- Modify: `api/app/models.py:192-221` (add after Concept class)

**Step 1: Add the FeedbackSignal enum and RecallEvent model**

Add after the Concept class (after line 221):

```python
class FeedbackSignal(str, Enum):
    useful = "useful"
    noted = "noted"
    wrong = "wrong"


class RecallEvent(Base):
    __tablename__ = "recall_events"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    concept_id: Mapped[UUID] = mapped_column(ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False)
    namespace: Mapped[str] = mapped_column(Text, nullable=False)
    session_id: Mapped[str] = mapped_column(Text, nullable=False)
    agent_key: Mapped[str] = mapped_column(Text, nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float] = mapped_column(nullable=False)
    session_completed: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )
```

**Step 2: Add the ConceptFeedback model**

Add after RecallEvent:

```python
class ConceptFeedback(Base):
    __tablename__ = "concept_feedback"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    concept_id: Mapped[UUID] = mapped_column(ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False)
    namespace: Mapped[str] = mapped_column(Text, nullable=False)
    signal: Mapped[FeedbackSignal] = mapped_column(
        Enum(FeedbackSignal, name="feedback_signal", create_type=False,
             values_callable=lambda e: [x.value for x in e]),
        nullable=False,
    )
    agent_key: Mapped[str] = mapped_column(Text, nullable=False)
    session_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )
```

**Step 3: Create the DB tables**

Run from project root:
```bash
docker compose exec -T api python3 -c "
from app.models import Base, RecallEvent, ConceptFeedback
from app.startup import build_engine
from app.settings import Settings
engine = build_engine(Settings())
# Create only the new tables
RecallEvent.__table__.create(engine, checkfirst=True)
ConceptFeedback.__table__.create(engine, checkfirst=True)
print('Tables created')
"
```

Also create the enum type in Postgres first:
```bash
docker compose exec -T db psql -U ssot -d ssot -c "
DO \$\$ BEGIN
    CREATE TYPE feedback_signal AS ENUM ('useful', 'noted', 'wrong');
EXCEPTION WHEN duplicate_object THEN NULL;
END \$\$;
"
```

**Step 4: Verify tables exist**

```bash
docker compose exec -T db psql -U ssot -d ssot -c "\dt recall_events; \dt concept_feedback;"
```

Expected: Both tables listed.

**Step 5: Commit**

```bash
git add api/app/models.py
git commit -m "feat: add recall_events and concept_feedback tables"
```

---

### Task 2: CRUD Functions — Logging and Querying Feedback

**Files:**
- Modify: `api/app/crud.py` (add new functions after `get_namespace_stats` at line ~965)

**Step 1: Add recall event logging function**

```python
def log_recall_events(
    session: Session,
    namespace: str,
    concept_ids: list[UUID],
    session_id: str,
    agent_key: str,
    query_text: str,
    scores: dict[UUID, float],
) -> int:
    """Log recall events for concepts that were surfaced. Returns count logged."""
    from .models import RecallEvent
    count = 0
    for cid in concept_ids:
        session.add(RecallEvent(
            concept_id=cid,
            namespace=namespace,
            session_id=session_id,
            agent_key=agent_key,
            query_text=query_text,
            score=scores.get(cid, 0.0),
        ))
        count += 1
    session.flush()
    return count
```

**Step 2: Add feedback creation with fuzzy match**

```python
def create_concept_feedback(
    session: Session,
    namespace: str,
    signal: str,
    agent_key: str,
    embedding_provider,
    concept_id: UUID | None = None,
    query: str | None = None,
    session_id: str | None = None,
    note: str | None = None,
) -> dict:
    """Create feedback for a concept. Resolves by ID or fuzzy semantic match."""
    from .models import Concept, ConceptFeedback, FeedbackSignal

    if not concept_id and not query:
        raise ValueError("Must provide concept_id or query")

    if concept_id:
        concept = session.get(Concept, concept_id)
        if not concept or concept.namespace != namespace:
            raise ValueError(f"Concept {concept_id} not found in namespace {namespace}")
    else:
        # Fuzzy match: embed query, find closest concept
        query_embedding = embedding_provider.embed(query)
        from pgvector.sqlalchemy import Vector
        score_col = Concept.embedding.cosine_distance(query_embedding).label("score")
        stmt = (
            select(Concept, score_col)
            .where(Concept.namespace == namespace)
            .where(Concept.embedding.is_not(None))
            .where(~Concept.tags.any("superseded"))
            .order_by(score_col)
            .limit(1)
        )
        row = session.execute(stmt).first()
        if not row:
            raise ValueError("No concepts found to match query")
        concept, match_score = row
        concept_id = concept.id

    fb = ConceptFeedback(
        concept_id=concept_id,
        namespace=namespace,
        signal=FeedbackSignal(signal),
        agent_key=agent_key,
        session_id=session_id,
        note=note,
    )
    session.add(fb)

    # If "wrong" signal with a note, also ingest the correction as knowledge
    if signal == "wrong" and note:
        from .models import KnowledgeItem
        session.add(KnowledgeItem(
            namespace=namespace,
            content=f"Correction: {note} (re: concept '{concept.title}')",
            tags=["correction", "operator-feedback"],
            embedding=embedding_provider.embed(note),
        ))

    session.flush()

    return {
        "concept_id": str(concept_id),
        "concept_title": concept.title,
        "signal": signal,
        "confidence": concept.confidence,
    }
```

**Step 3: Add feedback summary query for reconciler**

```python
def get_feedback_summary(
    session: Session,
    namespace: str,
    since: datetime,
) -> dict[UUID, dict]:
    """Get aggregated feedback per concept since a given timestamp.

    Returns: {concept_id: {"useful": N, "noted": N, "wrong": N, "implicit_recalls": N}}
    """
    from .models import ConceptFeedback, FeedbackSignal, RecallEvent
    from collections import defaultdict

    summary: dict[UUID, dict] = defaultdict(lambda: {
        "useful": 0, "noted": 0, "wrong": 0, "implicit_recalls": 0, "wrong_notes": []
    })

    # Explicit feedback
    fb_stmt = (
        select(ConceptFeedback)
        .where(ConceptFeedback.namespace == namespace)
        .where(ConceptFeedback.created_at >= since)
    )
    for fb in session.scalars(fb_stmt):
        summary[fb.concept_id][fb.signal.value] += 1
        if fb.signal == FeedbackSignal.wrong and fb.note:
            summary[fb.concept_id]["wrong_notes"].append(fb.note)

    # Implicit recalls (session completed)
    re_stmt = (
        select(RecallEvent.concept_id, func.count(func.distinct(RecallEvent.session_id)))
        .where(RecallEvent.namespace == namespace)
        .where(RecallEvent.session_completed == True)
        .where(RecallEvent.created_at >= since)
        .group_by(RecallEvent.concept_id)
    )
    for cid, count in session.execute(re_stmt):
        summary[cid]["implicit_recalls"] = count

    return dict(summary)
```

**Step 4: Add session completion marker**

```python
def mark_session_completed(session: Session, session_id: str) -> int:
    """Mark all recall events for a session as completed. Returns count updated."""
    from .models import RecallEvent
    stmt = (
        update(RecallEvent)
        .where(RecallEvent.session_id == session_id)
        .where(RecallEvent.session_completed == False)
        .values(session_completed=True)
    )
    result = session.execute(stmt)
    session.flush()
    return result.rowcount
```

**Step 5: Commit**

```bash
git add api/app/crud.py
git commit -m "feat: CRUD functions for recall events and concept feedback"
```

---

### Task 3: Integrate Recall Event Logging into recall()

**Files:**
- Modify: `api/app/crud.py:349` (the `recall()` function)
- Modify: `api/app/schemas.py:99` (RecallRequest — add session_id/agent_key fields)

**Step 1: Add session tracking fields to RecallRequest**

In `api/app/schemas.py`, add to the RecallRequest class:

```python
session_id: str | None = None
agent_key: str | None = None  # populated by the API layer from auth context
```

**Step 2: Add recall event logging at the end of the recall() function**

In `api/app/crud.py`, at the end of the `recall()` function, just before the final return statement, add concept recall event logging:

```python
    # Log recall events for concepts that were surfaced
    if payload.session_id:
        concept_items = [item for item in final_items if item.get("scope") == "concepts"]
        if concept_items:
            concept_ids = [UUID(item["id"]) for item in concept_items]
            scores_map = {UUID(item["id"]): item["score"] for item in concept_items}
            log_recall_events(
                session=session,
                namespace=payload.namespace,
                concept_ids=concept_ids,
                session_id=payload.session_id,
                agent_key=payload.agent_key or "unknown",
                query_text=payload.query_text or "",
                scores=scores_map,
            )
```

**Step 3: Commit**

```bash
git add api/app/crud.py api/app/schemas.py
git commit -m "feat: log recall events when concepts are surfaced"
```

---

### Task 4: API Endpoints — feedback and session-complete

**Files:**
- Modify: `api/app/main.py` (add endpoints after `/cortex/data` route, around line 170)
- Modify: `api/app/schemas.py` (add request/response schemas)

**Step 1: Add Pydantic schemas**

In `api/app/schemas.py`, add:

```python
class FeedbackRequest(BaseModel):
    signal: Literal["useful", "noted", "wrong"]
    concept_id: str | None = None
    query: str | None = None
    note: str | None = None
    session_id: str | None = None

class FeedbackResponse(BaseModel):
    concept_id: str
    concept_title: str
    signal: str
    confidence: float

class SessionCompleteRequest(BaseModel):
    session_id: str
    conversation_summary: str
    recalled_concept_ids: list[str] = []

class SessionCompleteResponse(BaseModel):
    session_id: str
    facts_extracted: int
    recall_events_completed: int
```

**Step 2: Add POST /feedback endpoint**

In `api/app/main.py`, add:

```python
@app.post("/feedback", response_model=schemas.FeedbackResponse)
def submit_feedback(
    payload: schemas.FeedbackRequest,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    namespace = auth.namespace or "claude-shared"
    ensure_namespace_access(auth, namespace, {ApiRole.reader.value, ApiRole.writer.value, ApiRole.admin.value})
    result = crud.create_concept_feedback(
        session=session,
        namespace=namespace,
        signal=payload.signal,
        agent_key=auth.key_name,
        embedding_provider=app.state.embedding_provider,
        concept_id=UUID(payload.concept_id) if payload.concept_id else None,
        query=payload.query,
        session_id=payload.session_id,
        note=payload.note,
    )
    return schemas.FeedbackResponse(**result)
```

**Step 3: Add POST /session-complete endpoint (Ollama-powered extraction)**

```python
@app.post("/session-complete", response_model=schemas.SessionCompleteResponse)
def session_complete(
    payload: schemas.SessionCompleteRequest,
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    namespace = auth.namespace or "claude-shared"
    ensure_namespace_access(auth, namespace, {ApiRole.writer.value, ApiRole.admin.value})

    # 1. Mark recall events as session completed
    completed_count = crud.mark_session_completed(session, payload.session_id)

    # 2. Extract facts via Ollama (zero Claude tokens)
    llm = app.state.llm_provider
    extraction_prompt = (
        "Extract 3-5 key facts from this conversation summary. "
        "Return each fact on its own line. Focus on: decisions made, bugs fixed, "
        "patterns learned, architecture changes. Skip routine actions.\n\n"
        f"Summary:\n{payload.conversation_summary}"
    )
    try:
        # Use local model for fast extraction
        raw = llm.generate(extraction_prompt, model_override=app.state.settings.synthesis_fallback_model)
        facts = [line.strip() for line in raw.strip().split("\n") if line.strip() and len(line.strip()) > 10]
    except Exception:
        facts = []

    # 3. Ingest extracted facts
    device_name = auth.key_name.replace("device-", "").replace("-writer", "") if auth.key_name else "unknown"
    for fact in facts:
        embedding = app.state.embedding_provider.embed(fact)
        session.add(models.KnowledgeItem(
            namespace=namespace,
            content=fact,
            tags=["session-extract", f"device-{device_name}", "auto-extracted"],
            embedding=embedding,
        ))
    session.flush()

    return schemas.SessionCompleteResponse(
        session_id=payload.session_id,
        facts_extracted=len(facts),
        recall_events_completed=completed_count,
    )
```

**Step 4: Verify endpoints respond**

```bash
# Rebuild
docker compose up -d --build api

# Test feedback endpoint
curl -s -X POST http://YOUR_HOST:8088/feedback \
  -H "X-API-Key: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"signal":"useful","query":"docker deployment","session_id":"test-1"}' | python3 -m json.tool
```

**Step 5: Commit**

```bash
git add api/app/main.py api/app/schemas.py
git commit -m "feat: feedback and session-complete API endpoints"
```

---

### Task 5: MCP Tools — hive_feedback and hive_session_end

**Files:**
- Modify: `/home/hari/.claude/plugins/hari-hive/mcp_server.py` (add after `hive_dedup` at line ~372)

**Step 1: Add hive_feedback tool**

```python
@server.tool()
async def hive_feedback(
    signal: str,
    concept_id: str = "",
    query: str = "",
    note: str = "",
    session_id: str = "",
) -> str:
    """Rate a concept: 'useful' (helped with task), 'noted' (good reminder), or 'wrong' (outdated/incorrect).
    Provide concept_id for direct reference, or query for fuzzy semantic match.
    Add a note for corrections (especially with 'wrong' signal)."""
    if not concept_id and not query:
        return "Error: provide concept_id or query to identify the concept"
    if signal not in ("useful", "noted", "wrong"):
        return "Error: signal must be 'useful', 'noted', or 'wrong'"

    body = {"signal": signal, "session_id": session_id or None, "note": note or None}
    if concept_id:
        body["concept_id"] = concept_id
    else:
        body["query"] = query

    data = await _post("/feedback", body)
    if "error" in data:
        return f"Feedback error: {data['error']}"
    return (
        f"Feedback recorded: {data['signal']} for '{data['concept_title']}' "
        f"(confidence: {data['confidence']:.2f})"
    )
```

**Step 2: Add hive_session_end tool**

```python
@server.tool()
async def hive_session_end(
    conversation_summary: str,
    session_id: str = "",
) -> str:
    """End-of-session processing: extracts facts via Ollama (zero Claude tokens) and marks recall events complete.
    Call this at the end of a session with a brief summary of what was accomplished."""
    if not conversation_summary.strip():
        return "Error: provide a conversation summary"

    body = {
        "session_id": session_id or f"session-{__import__('time').time_ns()}",
        "conversation_summary": conversation_summary,
        "recalled_concept_ids": [],
    }
    data = await _post("/session-complete", body)
    if "error" in data:
        return f"Session-end error: {data['error']}"
    return (
        f"Session complete: {data['facts_extracted']} facts extracted (via Ollama), "
        f"{data['recall_events_completed']} recall events marked complete"
    )
```

**Step 3: Verify tools load**

```bash
# Restart Claude Code or check MCP tool list
# The new tools should appear as hive_feedback and hive_session_end
```

**Step 4: Commit**

```bash
git add ~/.claude/plugins/hari-hive/mcp_server.py
git commit -m "feat: hive_feedback and hive_session_end MCP tools"
```

---

### Task 6: Update SessionEnd Hook to Use Ollama

**Files:**
- Modify: `/home/hari/.claude/plugins/hari-hive/hooks/SessionEnd.md`

**Step 1: Rewrite SessionEnd hook**

Replace the entire content:

```markdown
---
name: SessionEnd
description: End-of-session fact extraction via Ollama (zero Claude tokens)
enabled: true
---

# hari-hive Session End

<hive-session-end>
Summarize this session in 2-3 sentences (what was accomplished, key decisions, problems solved).
Then call hive_session_end with that summary. This extracts facts via Ollama — no extra Claude tokens.
</hive-session-end>
```

**Step 2: Commit**

```bash
git add ~/.claude/plugins/hari-hive/hooks/SessionEnd.md
git commit -m "feat: migrate SessionEnd extraction to Ollama"
```

---

### Task 7: Reconciler Integration — Feedback-Aware Confidence

**Files:**
- Modify: `api/app/synthesis/reconciler.py:110` (update `decay_stale_concepts`)
- Modify: `api/app/synthesis/loop.py:103` (integrate feedback into synthesis cycle)
- Modify: `api/app/settings.py:43-50` (update decay defaults)

**Step 1: Update settings defaults**

In `api/app/settings.py`, update:

```python
synthesis_confidence_decay: float = Field(default=0.02, json_schema_extra={"env": "SYNTHESIS_CONFIDENCE_DECAY"})
synthesis_decay_grace_days: int = Field(default=90, json_schema_extra={"env": "SYNTHESIS_DECAY_GRACE_DAYS"})
synthesis_decay_floor: float = Field(default=0.15, json_schema_extra={"env": "SYNTHESIS_DECAY_FLOOR"})
synthesis_feedback_protection_days: int = Field(default=180, json_schema_extra={"env": "SYNTHESIS_FEEDBACK_PROTECTION_DAYS"})
```

**Step 2: Add feedback processing to reconciler**

Add a new function in `api/app/synthesis/reconciler.py`:

```python
def apply_feedback_signals(
    session: Session,
    namespace: str,
    since: datetime,
    feedback_protection_days: int = 180,
) -> tuple[set[UUID], int]:
    """Apply feedback signals to concept confidence. Returns (protected_ids, adjustments_made)."""
    from .models import Concept
    from app.crud import get_feedback_summary
    from datetime import UTC, timedelta

    summary = get_feedback_summary(session, namespace, since)
    protection_cutoff = datetime.now(UTC) - timedelta(days=feedback_protection_days)
    protected_ids: set[UUID] = set()
    adjustments = 0

    for concept_id, signals in summary.items():
        concept = session.get(Concept, concept_id)
        if not concept or "superseded" in (concept.tags or []):
            continue

        delta = 0.0

        # Explicit signals (per-session capped)
        useful = min(signals["useful"], 2)  # cap +0.30
        delta += useful * 0.15

        noted = min(signals["noted"], 2)  # cap +0.10
        delta += noted * 0.05

        # Implicit recalls
        implicit = min(signals["implicit_recalls"], 5)  # cap +0.10
        delta += implicit * 0.02

        # Apply confidence adjustment
        if delta > 0:
            concept.confidence = min(1.0, concept.confidence + delta)
            concept.updated_at = datetime.now(UTC)
            adjustments += 1

        # Protection: any positive signal protects from decay
        if signals["useful"] > 0 or signals["noted"] > 0 or signals["implicit_recalls"] > 0:
            protected_ids.add(concept_id)

        # Wrong signals: tag as contested for LLM review
        if signals["wrong"] > 0 and "contested" not in (concept.tags or []):
            concept.tags = list(concept.tags or []) + ["contested"]

    # Also protect concepts with any positive feedback in protection window
    from app.models import ConceptFeedback, FeedbackSignal
    recent_positive = (
        select(func.distinct(ConceptFeedback.concept_id))
        .where(ConceptFeedback.namespace == namespace)
        .where(ConceptFeedback.signal.in_([FeedbackSignal.useful, FeedbackSignal.noted]))
        .where(ConceptFeedback.created_at >= protection_cutoff)
    )
    for (cid,) in session.execute(recent_positive):
        protected_ids.add(cid)

    session.flush()
    return protected_ids, adjustments
```

**Step 3: Update decay_stale_concepts to accept protected IDs and new defaults**

In `api/app/synthesis/reconciler.py`, update the `decay_stale_concepts` function signature and body:

```python
def decay_stale_concepts(
    session: Session,
    namespace: str,
    active_concept_ids: set[UUID],
    decay_rate: float = 0.02,
    min_age_days: int = 90,
    decay_floor: float = 0.15,
    protected_ids: set[UUID] | None = None,
) -> int:
    from datetime import UTC, datetime, timedelta
    cutoff = datetime.now(UTC) - timedelta(days=min_age_days)
    protected = protected_ids or set()

    stmt = (
        select(Concept)
        .where(Concept.namespace == namespace)
        .where(~Concept.tags.any("superseded"))
        .where(Concept.confidence > decay_floor)
        .where(Concept.updated_at < cutoff)
    )
    decayed = 0
    for concept in session.scalars(stmt):
        if concept.id in active_concept_ids or concept.id in protected:
            continue
        concept.confidence = max(decay_floor, concept.confidence - decay_rate)
        if concept.confidence <= decay_floor:
            concept.tags = list(concept.tags or []) + ["dormant"]
        decayed += 1

    session.flush()
    return decayed
```

**Step 4: Add resurrection logic**

Add to reconciler.py:

```python
def resurrect_concept(session: Session, concept_id: UUID) -> bool:
    """Revive a dormant concept to confidence 0.5. Returns True if resurrected."""
    concept = session.get(Concept, concept_id)
    if not concept:
        return False
    if concept.confidence < 0.5 and "dormant" in (concept.tags or []):
        concept.confidence = 0.5
        concept.tags = [t for t in concept.tags if t != "dormant"]
        concept.updated_at = datetime.now(UTC)
        session.flush()
        return True
    return False
```

**Step 5: Integrate feedback into synthesis loop**

In `api/app/synthesis/loop.py`, inside `_run_synthesis_for_namespace`, add feedback processing before the decay step:

```python
    # --- Feedback integration (Layer 2) ---
    from app.synthesis.reconciler import apply_feedback_signals
    from datetime import UTC, datetime, timedelta

    last_synthesis_time = datetime.now(UTC) - timedelta(days=1)  # default to last 24h
    protected_ids, feedback_adjustments = apply_feedback_signals(
        session, namespace, since=last_synthesis_time,
        feedback_protection_days=settings.synthesis_feedback_protection_days,
    )
    stats["feedback_adjustments"] = feedback_adjustments

    # --- Decay (with feedback protection) ---
    if not skip_decay:
        decayed = decay_stale_concepts(
            session, namespace, all_touched_ids,
            decay_rate=settings.synthesis_confidence_decay,
            min_age_days=settings.synthesis_decay_grace_days,
            decay_floor=settings.synthesis_decay_floor,
            protected_ids=protected_ids,
        )
        stats["decayed"] = decayed
```

**Step 6: Commit**

```bash
git add api/app/synthesis/reconciler.py api/app/synthesis/loop.py api/app/settings.py
git commit -m "feat: feedback-aware reconciler with 90-day decay and resurrection"
```

---

### Task 8: Wire Resurrection into Feedback

**Files:**
- Modify: `api/app/crud.py` (update `create_concept_feedback` to trigger resurrection)

**Step 1: Add resurrection trigger**

In the `create_concept_feedback` function, after creating the feedback record, add:

```python
    # Resurrect dormant concepts on positive feedback
    if signal in ("useful", "noted") and concept.confidence < 0.5:
        from app.synthesis.reconciler import resurrect_concept
        resurrect_concept(session, concept_id)
        concept = session.get(Concept, concept_id)  # refresh
```

**Step 2: Commit**

```bash
git add api/app/crud.py
git commit -m "feat: resurrect dormant concepts on positive feedback"
```

---

### Task 9: Update MCP hive_recall to Pass Session Context

**Files:**
- Modify: `/home/hari/.claude/plugins/hari-hive/mcp_server.py` (update `hive_recall`)

**Step 1: Add session_id parameter to hive_recall**

Update the `hive_recall` function signature to include session_id:

```python
@server.tool()
async def hive_recall(
    query: str,
    namespace: str = "",
    scope: str = "all",
    top_k: int = 5,
    session_id: str = "",
) -> str:
```

And update the body dict to include it:

```python
    body = {
        "query_text": query,
        "namespace": ns,
        "scope": scope,
        "top_k": top_k,
        "session_id": session_id or None,
    }
```

**Step 2: Commit**

```bash
git add ~/.claude/plugins/hari-hive/mcp_server.py
git commit -m "feat: pass session context through hive_recall for event tracking"
```

---

### Task 10: Integration Test — Full Feedback Cycle

**Step 1: Rebuild and verify**

```bash
docker compose -f /opt/agentssot/docker-compose.yml up -d --build api
sleep 5
curl -s http://YOUR_HOST:8088/health | python3 -m json.tool
```

**Step 2: Test explicit feedback (by query)**

```bash
ADMIN_KEY=$(jq -r .admin_api_key ~/.claude/agentssot/local/agent.json)
curl -s -X POST http://YOUR_HOST:8088/feedback \
  -H "X-API-Key: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"signal":"useful","query":"docker deployment patterns","session_id":"integration-test-1"}' \
  | python3 -m json.tool
```

Expected: Returns matched concept title with updated confidence.

**Step 3: Test "wrong" feedback with correction**

```bash
curl -s -X POST http://YOUR_HOST:8088/feedback \
  -H "X-API-Key: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"signal":"wrong","query":"kubernetes orchestration","note":"We use Docker Compose, not Kubernetes","session_id":"integration-test-1"}' \
  | python3 -m json.tool
```

Expected: Concept tagged "contested", correction ingested as knowledge item.

**Step 4: Test session-complete endpoint**

```bash
curl -s -X POST http://YOUR_HOST:8088/session-complete \
  -H "X-API-Key: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"integration-test-1","conversation_summary":"Tested the feedback loop. Added recall event tracking and concept feedback with three signals. Migrated SessionEnd to Ollama.","recalled_concept_ids":[]}' \
  | python3 -m json.tool
```

Expected: facts_extracted > 0, recall_events_completed >= 0.

**Step 5: Trigger synthesis and verify feedback integration**

```bash
curl -s -X POST "http://YOUR_HOST:8088/admin/synthesize?namespace=claude-shared" \
  -H "X-API-Key: $ADMIN_KEY" | python3 -m json.tool
```

Expected: Response includes `feedback_adjustments` field.

**Step 6: Commit integration test notes**

```bash
git add -A
git commit -m "feat: Layer 2 feedback loop complete — three signals, Ollama extraction, 90-day decay"
```
