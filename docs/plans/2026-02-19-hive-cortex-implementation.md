# Hive Cortex Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a conceptual memory layer that periodically synthesizes accumulated knowledge into evolving mental models, relationships, and principles using a local LLM (Qwen3 30B-A3B).

**Architecture:** New `Concept` model + `synthesis` package. Daily background loop gathers recent knowledge, clusters semantically, feeds to Qwen3 30B for synthesis, reconciles with existing concepts (versioned evolution), embeds results, and stores them. Concepts blend into existing `/recall` pipeline with zero caller changes.

**Tech Stack:** SQLAlchemy (Concept model), pgvector (embeddings), Ollama Qwen3 30B-A3B (synthesis LLM), existing embedding/reranker providers.

---

### Task 1: Database Schema — Concept Table

**Files:**
- Modify: `db/init/001_init.sql` (append concept table DDL)
- Modify: `api/app/models.py` (add enums + Concept model)
- Modify: `api/app/startup.py` (add migration for existing DBs)

**Step 1: Add enum types and concept table to init SQL**

Append to the bottom of `db/init/001_init.sql` (before the final INSERT):

```sql
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'concept_type') THEN
        CREATE TYPE concept_type AS ENUM ('mental_model', 'relationship', 'principle');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'concept_scope') THEN
        CREATE TYPE concept_scope AS ENUM ('global', 'project', 'device');
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS concepts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL REFERENCES namespaces(name) ON DELETE CASCADE,
    type concept_type NOT NULL,
    scope concept_scope NOT NULL DEFAULT 'global',
    scope_ref TEXT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    evidence_ids UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    confidence FLOAT NOT NULL DEFAULT 0.5,
    version INTEGER NOT NULL DEFAULT 1,
    parent_id UUID REFERENCES concepts(id) ON DELETE SET NULL,
    tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_concepts_namespace ON concepts(namespace);
CREATE INDEX IF NOT EXISTS idx_concepts_type ON concepts(type);
CREATE INDEX IF NOT EXISTS idx_concepts_scope ON concepts(scope, scope_ref);
CREATE INDEX IF NOT EXISTS idx_concepts_tags_gin ON concepts USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_concepts_parent ON concepts(parent_id);
CREATE INDEX IF NOT EXISTS idx_concepts_confidence ON concepts(confidence);

DROP TRIGGER IF EXISTS trg_concepts_set_updated_at ON concepts;
CREATE TRIGGER trg_concepts_set_updated_at
BEFORE UPDATE ON concepts
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();
```

**Step 2: Add SQLAlchemy enums and Concept model to `api/app/models.py`**

Add after the `EventType` enum (line ~51):

```python
class ConceptType(str, enum.Enum):
    mental_model = "mental_model"
    relationship = "relationship"
    principle = "principle"


class ConceptScope(str, enum.Enum):
    global_ = "global"
    project = "project"
    device = "device"
```

Add the Concept model after the `Event` model (after line 178):

```python
class Concept(Base):
    __tablename__ = "concepts"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    namespace: Mapped[str] = mapped_column(Text, ForeignKey("namespaces.name", ondelete="CASCADE"), nullable=False)
    type: Mapped[ConceptType] = mapped_column(
        Enum(ConceptType, name="concept_type", create_type=False), nullable=False
    )
    scope: Mapped[ConceptScope] = mapped_column(
        Enum(ConceptScope, name="concept_scope", create_type=False),
        nullable=False,
        default=ConceptScope.global_,
        server_default="global",
    )
    scope_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)), nullable=False, default=list, server_default=text("ARRAY[]::UUID[]")
    )
    confidence: Mapped[float] = mapped_column(nullable=False, default=0.5, server_default=text("0.5"))
    version: Mapped[int] = mapped_column(nullable=False, default=1, server_default=text("1"))
    parent_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("concepts.id", ondelete="SET NULL"), nullable=True
    )
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list, server_default=text("ARRAY[]::TEXT[]"))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.embedding_dim), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
```

**Step 3: Add startup migration in `api/app/startup.py`**

Add a new function `_ensure_concepts_table` and call it from `initialize_system`:

```python
def _ensure_concepts_table(session) -> None:
    """Create concepts table if it doesn't exist (migration for existing DBs)."""
    try:
        session.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'concept_type') THEN
                    CREATE TYPE concept_type AS ENUM ('mental_model', 'relationship', 'principle');
                END IF;
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'concept_scope') THEN
                    CREATE TYPE concept_scope AS ENUM ('global', 'project', 'device');
                END IF;
            END $$
        """))
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS concepts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                namespace TEXT NOT NULL REFERENCES namespaces(name) ON DELETE CASCADE,
                type concept_type NOT NULL,
                scope concept_scope NOT NULL DEFAULT 'global',
                scope_ref TEXT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                evidence_ids UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
                confidence FLOAT NOT NULL DEFAULT 0.5,
                version INTEGER NOT NULL DEFAULT 1,
                parent_id UUID REFERENCES concepts(id) ON DELETE SET NULL,
                tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                embedding VECTOR(:dim),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """), {"dim": settings.embedding_dim})
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_concepts_namespace ON concepts(namespace);
            CREATE INDEX IF NOT EXISTS idx_concepts_type ON concepts(type);
            CREATE INDEX IF NOT EXISTS idx_concepts_scope ON concepts(scope, scope_ref);
            CREATE INDEX IF NOT EXISTS idx_concepts_tags_gin ON concepts USING GIN(tags);
            CREATE INDEX IF NOT EXISTS idx_concepts_parent ON concepts(parent_id);
            CREATE INDEX IF NOT EXISTS idx_concepts_confidence ON concepts(confidence);
        """))
        # Add updated_at trigger
        session.execute(text("""
            DROP TRIGGER IF EXISTS trg_concepts_set_updated_at ON concepts;
            CREATE TRIGGER trg_concepts_set_updated_at
            BEFORE UPDATE ON concepts
            FOR EACH ROW
            EXECUTE FUNCTION set_updated_at();
        """))
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.warning("concepts table creation skipped: %s", exc)
```

In `initialize_system`, add `_ensure_concepts_table(session)` after `_ensure_enrollment_tokens_table(session)`.

Also add `Concept` to the import list and ensure `_ensure_embedding_dim` also handles the concepts table vector column.

**Step 4: Rebuild and verify**

Run: `cd /opt/agentssot && docker compose up -d --build api`
Verify: `docker compose exec api python3 -c "from app.models import Concept; print('OK')"`
Verify table exists: `docker compose exec db psql -U ssot -d ssot -c '\d concepts'`

**Step 5: Commit**

```bash
git add db/init/001_init.sql api/app/models.py api/app/startup.py
git commit -m "feat: add Concept table for conceptual memory layer"
```

---

### Task 2: Pydantic Schemas for Concepts

**Files:**
- Modify: `api/app/schemas.py`

**Step 1: Add concept schemas to `api/app/schemas.py`**

Add after the `DedupResponse` class (around line 230):

```python
class ConceptOut(BaseModel):
    id: str
    namespace: str
    type: Literal["mental_model", "relationship", "principle"]
    scope: Literal["global", "project", "device"]
    scope_ref: str | None = None
    title: str
    content: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float
    version: int
    parent_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ConceptListResponse(BaseModel):
    namespace: str
    total: int
    concepts: list[ConceptOut]


class ConceptDetailResponse(ConceptOut):
    history: list[ConceptOut] = Field(default_factory=list)


class SynthesisRunResponse(BaseModel):
    namespace: str
    new_concepts: int
    updated_concepts: int
    decayed_concepts: int
```

Also update `RecallRequest` to accept `scope: "concepts"` and `RecallItem` to include concept metadata:

Update `RecallRequest.scope` from:
```python
scope: Literal["knowledge", "requirements", "events"] = "knowledge"
```
To:
```python
scope: Literal["knowledge", "requirements", "events", "concepts"] = "knowledge"
```

Update `RecallItem.scope` similarly:
```python
scope: Literal["knowledge", "requirements", "events", "concepts"]
```

Update `RecallResponse.scope` similarly:
```python
scope: Literal["knowledge", "requirements", "events", "concepts"]
```

Add concept-specific optional fields to `RecallItem`:
```python
concept_type: Literal["mental_model", "relationship", "principle"] | None = None
confidence: float | None = None
```

Also update `NamespaceStatsResponse` to include concepts:
```python
class NamespaceStatsResponse(BaseModel):
    namespace: str
    entities: int
    knowledge_items: ItemCountDetail
    requirements: ItemCountDetail
    events: ItemCountDetail
    concepts: ItemCountDetail | None = None
```

**Step 2: Commit**

```bash
git add api/app/schemas.py
git commit -m "feat: add Pydantic schemas for concepts and update recall scope"
```

---

### Task 3: Settings — Synthesis Configuration

**Files:**
- Modify: `api/app/settings.py`
- Modify: `docker-compose.yml`

**Step 1: Add synthesis settings to `api/app/settings.py`**

Add after the compaction settings (around line 41):

```python
    # Synthesis (conceptual memory)
    synthesis_enabled: bool = Field(default=False, alias="SYNTHESIS_ENABLED")
    synthesis_model: str = Field(default="qwen3.5:cloud", alias="SYNTHESIS_MODEL")
    synthesis_fallback_model: str = Field(default="qwen3:latest", alias="SYNTHESIS_FALLBACK_MODEL")
    synthesis_schedule_hour: int = Field(default=3, alias="SYNTHESIS_SCHEDULE_HOUR")
    synthesis_batch_size: int = Field(default=20, alias="SYNTHESIS_BATCH_SIZE")
    synthesis_similarity_threshold: float = Field(default=0.75, alias="SYNTHESIS_SIMILARITY_THRESHOLD")
    synthesis_min_cluster_size: int = Field(default=3, alias="SYNTHESIS_MIN_CLUSTER_SIZE")
    synthesis_confidence_decay: float = Field(default=0.05, alias="SYNTHESIS_CONFIDENCE_DECAY")
```

Add a property:
```python
    @property
    def effective_synthesis_enabled(self) -> bool:
        return self.synthesis_enabled and self.llm_provider != "none"
```

**Step 2: No docker-compose.yml changes needed**

The `api` service already uses `env_file: .env` (see `docker-compose.yml:24`), so all `SYNTHESIS_*` env vars are automatically loaded from `.env` by pydantic-settings. No need to add them to the `environment:` block.

**Step 3: Commit**

```bash
git add api/app/settings.py
git commit -m "feat: add synthesis configuration settings"
```

---

### Task 4: LLM Provider — Add `synthesize_concepts` Method

**Files:**
- Modify: `api/app/llm/base.py`
- Modify: `api/app/llm/ollama_provider.py`
- Modify: `api/app/llm/__init__.py`

**Step 1: Add `synthesize_concepts` to base LLM provider**

In `api/app/llm/base.py`, add a new method to `LLMProvider`:

```python
    def synthesize_concepts(self, facts: str, existing_concepts: str, model_override: str | None = None, fallback_model: str | None = None) -> str:
        """Synthesize conceptual knowledge from facts. Uses model_override if set,
        falls back to fallback_model if primary fails, then to self.model."""
        raise NotImplementedError
```

**Step 2: Implement in Ollama provider**

In `api/app/llm/ollama_provider.py`, add a `model_override` parameter support and the `synthesize_concepts` method. The synthesis method needs a longer timeout (120s) since the 30B model is slower:

```python
    def synthesize_concepts(
        self,
        facts: str,
        existing_concepts: str,
        model_override: str | None = None,
        fallback_model: str | None = None,
    ) -> str:
        if not self.is_available:
            raise LLMProviderError(self.unavailable_reason or "Ollama LLM provider unavailable")

        model = model_override or self.model

        system_prompt = """You are a knowledge synthesis engine. Review the provided facts and identify conceptual patterns.

For each concept you identify, output a JSON object on its own line with these fields:
- type: "mental_model" | "relationship" | "principle"
- scope: "global" | "project" | "device"
- scope_ref: project slug or device name if scoped, null if global
- title: concise label (under 80 chars)
- content: full description (2-5 sentences)
- confidence: 0.0-1.0 based on evidence strength
- matches_existing_id: UUID of existing concept if this reinforces/contradicts one, null otherwise
- is_contradiction: true if this contradicts the matched existing concept, false if reinforcement

Output ONLY valid JSON lines (one JSON object per line, no markdown, no commentary).
If no meaningful concepts can be extracted, output an empty line.

Rules:
- Only extract concepts with clear evidence from multiple facts
- Prefer specific, actionable knowledge over vague generalizations
- Relationships should name specific entities (projects, hosts, tools)
- Principles should be testable and falsifiable
- Mental models should describe observable patterns"""

        user_content = f"=== NEW FACTS ===\n{facts}"
        if existing_concepts.strip():
            user_content += f"\n\n=== EXISTING CONCEPTS ===\n{existing_concepts}"

        payload = {
            "model": model,
            "stream": False,
            "options": {"num_ctx": 8192},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }

        url = f"{self.base_url}/api/chat"

        try:
            response = httpx.post(url, json=payload, timeout=120)
            if response.status_code >= 400:
                raise LLMProviderError(f"Ollama synthesis failed with {response.status_code}: {response.text[:400]}")
        except Exception as exc:
            # Fallback to local model if cloud model fails
            if fallback_model and model != fallback_model:
                import logging
                logging.getLogger("agentssot.llm").warning(
                    "synthesis model %s failed, falling back to %s: %s", model, fallback_model, exc
                )
                payload["model"] = fallback_model
                try:
                    response = httpx.post(url, json=payload, timeout=120)
                    if response.status_code >= 400:
                        raise LLMProviderError(f"Fallback model also failed: {response.status_code}")
                except Exception as fallback_exc:
                    raise LLMProviderError(f"Both {model} and {fallback_model} failed: {fallback_exc}") from fallback_exc
            else:
                raise LLMProviderError(f"Ollama synthesis request failed: {exc}") from exc

        data = response.json()
        content = data.get("message", {}).get("content")
        if not isinstance(content, str):
            raise LLMProviderError("Ollama synthesis response format was unexpected")
        return content.strip()
```

**Step 3: Update DisabledLLMProvider in `__init__.py`**

Add the method override:
```python
    def synthesize_concepts(self, facts: str, existing_concepts: str, model_override: str | None = None, fallback_model: str | None = None) -> str:
        raise LLMProviderError(self.unavailable_reason or "LLM provider disabled")
```

**Step 4: Commit**

```bash
git add api/app/llm/
git commit -m "feat: add synthesize_concepts method to LLM providers"
```

---

### Task 5: Synthesis Package — Clustering

**Files:**
- Create: `api/app/synthesis/__init__.py`
- Create: `api/app/synthesis/clustering.py`

**Step 1: Create the synthesis package**

Create `api/app/synthesis/__init__.py`:
```python
from .clustering import cluster_items
from .loop import synthesis_loop
from .reconciler import reconcile_concepts
from .synthesizer import run_synthesis_batch

__all__ = [
    "cluster_items",
    "synthesis_loop",
    "reconcile_concepts",
    "run_synthesis_batch",
]
```

**Step 2: Implement clustering in `api/app/synthesis/clustering.py`**

This groups knowledge items by semantic similarity using their existing embeddings:

```python
import logging
from collections import defaultdict
from uuid import UUID

import numpy as np

logger = logging.getLogger("agentssot.synthesis.clustering")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def cluster_items(
    items: list[dict],
    similarity_threshold: float = 0.75,
    min_cluster_size: int = 3,
) -> list[list[dict]]:
    """Cluster items by embedding similarity using greedy agglomerative approach.

    Each item dict must have 'id', 'content', 'embedding' (list[float]).
    Returns list of clusters, each cluster is a list of items.
    Only returns clusters with >= min_cluster_size items.
    """
    if not items:
        return []

    # Filter items with valid embeddings
    embedded = [it for it in items if it.get("embedding")]
    if len(embedded) < min_cluster_size:
        return []

    # Simple greedy clustering: assign each item to the first cluster
    # whose centroid is within threshold, or start a new cluster
    clusters: list[list[dict]] = []
    centroids: list[np.ndarray] = []

    for item in embedded:
        emb = np.array(item["embedding"])
        assigned = False

        for i, centroid in enumerate(centroids):
            sim = float(np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-9))
            if sim >= similarity_threshold:
                clusters[i].append(item)
                # Update centroid as running average
                n = len(clusters[i])
                centroids[i] = centroid * ((n - 1) / n) + emb * (1 / n)
                assigned = True
                break

        if not assigned:
            clusters.append([item])
            centroids.append(emb.copy())

    # Filter by minimum cluster size
    result = [c for c in clusters if len(c) >= min_cluster_size]
    logger.info(
        "clustering complete",
        extra={"total_items": len(embedded), "clusters_formed": len(result), "threshold": similarity_threshold},
    )
    return result
```

**Step 3: Commit**

```bash
git add api/app/synthesis/
git commit -m "feat: add synthesis package with semantic clustering"
```

---

### Task 6: Synthesis Package — Synthesizer

**Files:**
- Create: `api/app/synthesis/synthesizer.py`

**Step 1: Implement the synthesizer**

This formats clusters for the LLM and parses the JSON response:

```python
import json
import logging
from uuid import UUID

from ..llm import LLMProvider, LLMProviderError

logger = logging.getLogger("agentssot.synthesis.synthesizer")


def _format_facts(items: list[dict]) -> str:
    lines = []
    for i, item in enumerate(items, 1):
        tags_str = f" [{', '.join(item.get('tags', []))}]" if item.get("tags") else ""
        source = item.get("source", "")
        source_str = f" (source: {source})" if source else ""
        lines.append(f"{i}. {item['content'][:600]}{tags_str}{source_str}")
    return "\n".join(lines)


def _format_existing_concepts(concepts: list[dict]) -> str:
    if not concepts:
        return ""
    lines = []
    for c in concepts:
        lines.append(
            f"ID: {c['id']} | Type: {c['type']} | Title: {c['title']} | "
            f"Confidence: {c['confidence']:.2f}\n  {c['content'][:400]}"
        )
    return "\n\n".join(lines)


def _parse_synthesis_response(raw: str) -> list[dict]:
    """Parse LLM output as JSON lines. Tolerant of markdown fences and extra text."""
    results = []

    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    for line in cleaned.split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            # Validate required fields
            if not all(k in obj for k in ("type", "title", "content", "confidence")):
                logger.warning("skipping concept missing required fields: %s", list(obj.keys()))
                continue
            if obj["type"] not in ("mental_model", "relationship", "principle"):
                logger.warning("skipping concept with unknown type: %s", obj["type"])
                continue
            results.append(obj)
        except json.JSONDecodeError:
            logger.debug("skipping non-JSON line in synthesis output")
            continue

    return results


def run_synthesis_batch(
    cluster_items: list[dict],
    existing_concepts: list[dict],
    llm_provider: LLMProvider,
    synthesis_model: str,
    fallback_model: str | None = None,
) -> list[dict]:
    """Run synthesis on a cluster of items, returning parsed concept proposals.

    Returns list of dicts with keys: type, scope, scope_ref, title, content,
    confidence, matches_existing_id, is_contradiction, evidence_item_ids.
    """
    facts_text = _format_facts(cluster_items)
    concepts_text = _format_existing_concepts(existing_concepts)

    try:
        raw_output = llm_provider.synthesize_concepts(
            facts=facts_text,
            existing_concepts=concepts_text,
            model_override=synthesis_model,
            fallback_model=fallback_model,
        )
    except LLMProviderError:
        logger.warning("synthesis LLM call failed for cluster", exc_info=True)
        return []

    proposals = _parse_synthesis_response(raw_output)

    # Attach evidence IDs from the cluster items
    evidence_ids = [str(item["id"]) for item in cluster_items if item.get("id")]
    for proposal in proposals:
        proposal.setdefault("scope", "global")
        proposal.setdefault("scope_ref", None)
        proposal.setdefault("matches_existing_id", None)
        proposal.setdefault("is_contradiction", False)
        proposal["evidence_item_ids"] = evidence_ids

    logger.info("synthesis batch produced %d concept proposals", len(proposals))
    return proposals
```

**Step 2: Commit**

```bash
git add api/app/synthesis/synthesizer.py
git commit -m "feat: add concept synthesizer with LLM prompt and JSON parsing"
```

---

### Task 7: Synthesis Package — Reconciler

**Files:**
- Create: `api/app/synthesis/reconciler.py`

**Step 1: Implement the reconciler**

This matches proposals to existing concepts and handles create/update/version logic:

```python
import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from ..embeddings import EmbeddingProvider, EmbeddingProviderError
from ..models import Concept, ConceptScope, ConceptType

logger = logging.getLogger("agentssot.synthesis.reconciler")

_SCOPE_MAP = {"global": ConceptScope.global_, "project": ConceptScope.project, "device": ConceptScope.device}
_TYPE_MAP = {"mental_model": ConceptType.mental_model, "relationship": ConceptType.relationship, "principle": ConceptType.principle}


def reconcile_concepts(
    session: Session,
    namespace: str,
    proposals: list[dict],
    embedding_provider: EmbeddingProvider,
    embedding_provider_kind: str,
    embedding_dim: int,
) -> dict:
    """Reconcile synthesis proposals against existing concepts.

    Returns: {"new": int, "updated": int}
    """
    new_count = 0
    updated_count = 0

    for proposal in proposals:
        matched_id = proposal.get("matches_existing_id")
        is_contradiction = proposal.get("is_contradiction", False)

        # Generate embedding for the concept content
        concept_embedding = None
        if embedding_provider_kind != "none" and embedding_provider.is_available:
            try:
                text_to_embed = f"{proposal['title']}\n{proposal['content']}"
                concept_embedding = embedding_provider.embed_text(text_to_embed)
            except EmbeddingProviderError:
                logger.warning("failed to embed concept, storing without embedding")

        if matched_id:
            # Try to find the existing concept
            try:
                existing = session.scalar(
                    select(Concept).where(
                        and_(Concept.id == UUID(matched_id), Concept.namespace == namespace)
                    )
                )
            except (ValueError, Exception):
                existing = None

            if existing and not is_contradiction:
                # Reinforcement: update in place
                existing.confidence = min(existing.confidence + 0.1, 1.0)
                new_evidence = [UUID(eid) for eid in proposal.get("evidence_item_ids", []) if eid]
                existing.evidence_ids = list(set(existing.evidence_ids or []) | set(new_evidence))
                existing.version += 1
                if proposal.get("content") and len(proposal["content"]) > len(existing.content):
                    existing.content = proposal["content"]
                    existing.title = proposal.get("title", existing.title)
                    if concept_embedding:
                        existing.embedding = concept_embedding
                updated_count += 1
                continue

            elif existing and is_contradiction:
                # Contradiction: supersede old, create new version
                existing.confidence = max(existing.confidence - 0.2, 0.0)
                existing.tags = list(set(existing.tags or []) | {"superseded"})
                existing.embedding = None  # Remove from recall index

                new_concept = Concept(
                    namespace=namespace,
                    type=_TYPE_MAP[proposal["type"]],
                    scope=_SCOPE_MAP.get(proposal.get("scope", "global"), ConceptScope.global_),
                    scope_ref=proposal.get("scope_ref"),
                    title=proposal["title"],
                    content=proposal["content"],
                    evidence_ids=[UUID(eid) for eid in proposal.get("evidence_item_ids", []) if eid],
                    confidence=proposal.get("confidence", 0.5),
                    version=existing.version + 1,
                    parent_id=existing.id,
                    tags=list(proposal.get("tags", [])),
                    embedding=concept_embedding,
                )
                session.add(new_concept)
                new_count += 1
                continue

        # New concept (no match or match not found)
        new_concept = Concept(
            namespace=namespace,
            type=_TYPE_MAP[proposal["type"]],
            scope=_SCOPE_MAP.get(proposal.get("scope", "global"), ConceptScope.global_),
            scope_ref=proposal.get("scope_ref"),
            title=proposal["title"],
            content=proposal["content"],
            evidence_ids=[UUID(eid) for eid in proposal.get("evidence_item_ids", []) if eid],
            confidence=proposal.get("confidence", 0.5),
            version=1,
            parent_id=None,
            tags=list(proposal.get("tags", [])),
            embedding=concept_embedding,
        )
        session.add(new_concept)
        new_count += 1

    session.commit()
    return {"new": new_count, "updated": updated_count}


def decay_stale_concepts(
    session: Session,
    namespace: str,
    active_concept_ids: set[UUID],
    decay_rate: float = 0.05,
) -> int:
    """Reduce confidence of concepts not reinforced this cycle. Returns count decayed."""
    stmt = (
        select(Concept)
        .where(Concept.namespace == namespace)
        .where(~Concept.tags.any("superseded"))
        .where(Concept.confidence > 0.1)
    )
    all_active = session.scalars(stmt).all()

    decayed = 0
    for concept in all_active:
        if concept.id not in active_concept_ids:
            concept.confidence = max(concept.confidence - decay_rate, 0.0)
            decayed += 1

            # Auto-archive if below threshold for too long
            if concept.confidence <= 0.1:
                concept.tags = list(set(concept.tags or []) | {"superseded"})
                concept.embedding = None

    if decayed:
        session.commit()
    return decayed
```

**Step 2: Commit**

```bash
git add api/app/synthesis/reconciler.py
git commit -m "feat: add concept reconciler with versioned evolution and decay"
```

---

### Task 8: Synthesis Package — Background Loop

**Files:**
- Create: `api/app/synthesis/loop.py`
- Modify: `api/app/background.py`
- Modify: `api/app/main.py`

**Step 1: Implement the synthesis loop in `api/app/synthesis/loop.py`**

```python
import asyncio
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from ..db import SessionLocal
from ..embeddings import EmbeddingProvider
from ..llm import LLMProvider
from ..models import Concept, KnowledgeItem, Namespace
from .clustering import cluster_items
from .reconciler import decay_stale_concepts, reconcile_concepts
from .synthesizer import run_synthesis_batch

logger = logging.getLogger("agentssot.synthesis.loop")


def _gather_recent_knowledge(
    session: Session,
    namespace: str,
    since: datetime,
    limit: int = 500,
) -> list[dict]:
    """Fetch knowledge items created since last synthesis run."""
    stmt = (
        select(KnowledgeItem)
        .where(KnowledgeItem.namespace == namespace)
        .where(KnowledgeItem.created_at > since)
        .where(KnowledgeItem.embedding.is_not(None))
        .order_by(KnowledgeItem.created_at.desc())
        .limit(limit)
    )
    rows = session.scalars(stmt).all()
    return [
        {
            "id": item.id,
            "content": item.content,
            "embedding": list(item.embedding) if item.embedding else None,
            "tags": list(item.tags or []),
            "source": item.source,
            "created_at": item.created_at,
        }
        for item in rows
    ]


def _get_active_concepts(session: Session, namespace: str) -> list[dict]:
    """Load all non-superseded concepts for reconciliation."""
    stmt = (
        select(Concept)
        .where(Concept.namespace == namespace)
        .where(~Concept.tags.any("superseded"))
    )
    rows = session.scalars(stmt).all()
    return [
        {
            "id": str(c.id),
            "type": c.type.value,
            "scope": c.scope.value if hasattr(c.scope, "value") else str(c.scope),
            "title": c.title,
            "content": c.content,
            "confidence": c.confidence,
            "embedding": list(c.embedding) if c.embedding else None,
        }
        for c in rows
    ]


def _find_related_concepts(
    cluster_items_list: list[dict],
    all_concepts: list[dict],
    threshold: float = 0.6,
    max_related: int = 5,
) -> list[dict]:
    """Find existing concepts related to a cluster by embedding similarity."""
    import numpy as np

    if not all_concepts or not cluster_items_list:
        return []

    # Compute cluster centroid
    embeddings = [it["embedding"] for it in cluster_items_list if it.get("embedding")]
    if not embeddings:
        return []
    centroid = np.mean(embeddings, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        return []

    scored = []
    for concept in all_concepts:
        if not concept.get("embedding"):
            continue
        c_emb = np.array(concept["embedding"])
        sim = float(np.dot(centroid, c_emb) / (centroid_norm * np.linalg.norm(c_emb) + 1e-9))
        if sim >= threshold:
            scored.append((sim, concept))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:max_related]]


def _run_synthesis_for_namespace(
    namespace: str,
    settings,
    llm_provider: LLMProvider,
    embedding_provider: EmbeddingProvider,
    full_resynthesis: bool = False,
) -> dict:
    """Run one synthesis cycle for a namespace. Returns stats dict.
    When full_resynthesis=True, examines ALL knowledge (not just recent).
    Useful after upgrading the synthesis model.
    """
    stats = {"namespace": namespace, "new": 0, "updated": 0, "decayed": 0, "clusters": 0}

    with SessionLocal() as session:
        if full_resynthesis:
            # Look at everything — epoch start
            since = datetime(2020, 1, 1, tzinfo=UTC)
            logger.info("full resynthesis requested", extra={"namespace": namespace})
        else:
            # Determine time window: last 24 hours (or since last concept update)
            latest_concept = session.scalar(
                select(func.max(Concept.updated_at)).where(Concept.namespace == namespace)
            )
            since = latest_concept or (datetime.now(UTC) - timedelta(days=7))
            # Always look back at least 24 hours
            min_since = datetime.now(UTC) - timedelta(hours=24)
            if since > min_since:
                since = min_since

        gather_limit = 5000 if full_resynthesis else 500
        items = _gather_recent_knowledge(session, namespace, since, limit=gather_limit)
        if not items:
            logger.info("no new knowledge to synthesize", extra={"namespace": namespace})
            return stats

        logger.info(
            "gathered items for synthesis",
            extra={"namespace": namespace, "item_count": len(items)},
        )

        # Cluster
        clusters = cluster_items(
            items,
            similarity_threshold=settings.synthesis_similarity_threshold,
            min_cluster_size=settings.synthesis_min_cluster_size,
        )
        stats["clusters"] = len(clusters)

        if not clusters:
            logger.info("no clusters formed", extra={"namespace": namespace})
            return stats

        # Load existing concepts for reconciliation
        all_concepts = _get_active_concepts(session, namespace)

        # Process each cluster
        all_touched_ids = set()
        for cluster in clusters:
            related = _find_related_concepts(cluster, all_concepts)

            proposals = run_synthesis_batch(
                cluster_items=cluster,
                existing_concepts=related,
                llm_provider=llm_provider,
                synthesis_model=settings.synthesis_model,
                fallback_model=settings.synthesis_fallback_model,
            )

            if proposals:
                result = reconcile_concepts(
                    session=session,
                    namespace=namespace,
                    proposals=proposals,
                    embedding_provider=embedding_provider,
                    embedding_provider_kind=settings.embedding_provider,
                    embedding_dim=settings.embedding_dim,
                )
                stats["new"] += result["new"]
                stats["updated"] += result["updated"]

                # Track touched concept IDs for decay calculation
                for p in proposals:
                    if p.get("matches_existing_id"):
                        try:
                            from uuid import UUID
                            all_touched_ids.add(UUID(p["matches_existing_id"]))
                        except ValueError:
                            pass

        # Decay concepts not reinforced
        decayed = decay_stale_concepts(
            session, namespace, all_touched_ids, settings.synthesis_confidence_decay
        )
        stats["decayed"] = decayed

    return stats


async def synthesis_loop(app) -> None:
    """Daily synthesis loop. Runs at the configured hour."""
    settings = app.state.settings

    while True:
        try:
            # Calculate sleep until next scheduled hour
            now = datetime.now(UTC)
            target_hour = settings.synthesis_schedule_hour
            next_run = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            sleep_seconds = (next_run - now).total_seconds()

            logger.info(
                "synthesis loop sleeping until next run",
                extra={"next_run": next_run.isoformat(), "sleep_seconds": int(sleep_seconds)},
            )
            await asyncio.sleep(sleep_seconds)

            # Run synthesis for all namespaces
            llm_provider = app.state.llm_provider
            embedding_provider = app.state.embedding_provider

            if not llm_provider.is_available:
                logger.warning("synthesis skipped; LLM provider unavailable")
                continue

            with SessionLocal() as session:
                namespaces = [ns.name for ns in session.scalars(select(Namespace)).all()]

            for ns in namespaces:
                try:
                    stats = _run_synthesis_for_namespace(
                        namespace=ns,
                        settings=settings,
                        llm_provider=llm_provider,
                        embedding_provider=embedding_provider,
                    )
                    logger.info("synthesis complete", extra=stats)
                except Exception:
                    logger.exception("synthesis failed for namespace", extra={"namespace": ns})

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("synthesis loop iteration failed")
            await asyncio.sleep(3600)  # Wait an hour before retrying on unexpected failure
```

**Step 2: Register synthesis loop in `api/app/main.py`**

In the `lifespan` function, after the compaction task setup (around line 51), add:

```python
    from .synthesis import synthesis_loop as _synthesis_loop

    synthesis_task = None
    if settings.effective_synthesis_enabled:
        synthesis_task = asyncio.create_task(_synthesis_loop(app), name="synthesis-loop")
        logger.info("background synthesis loop started (hour=%d)", settings.synthesis_schedule_hour)
    else:
        logger.info("background synthesis loop disabled")

    app.state.synthesis_task = synthesis_task
```

In the yield teardown section, add:

```python
    if synthesis_task:
        synthesis_task.cancel()
        try:
            await synthesis_task
        except asyncio.CancelledError:
            pass
```

Also update the `/health` endpoint to include synthesis status:

```python
        "synthesis_enabled": settings.effective_synthesis_enabled,
```

**Step 3: Commit**

```bash
git add api/app/synthesis/ api/app/background.py api/app/main.py
git commit -m "feat: add daily synthesis background loop"
```

---

### Task 9: CRUD — Concept Recall Integration

**Files:**
- Modify: `api/app/crud.py`

**Step 1: Add concept recall to the `recall` function**

In `api/app/crud.py`, add the Concept import at line 13:

```python
from .models import ApiKey, ApiRole, Concept, ConceptScope, ConceptType, EnrollmentToken, Entity, EntityType, Event, EventType, KnowledgeItem, Namespace, Requirement
```

Add a new `scope == "concepts"` branch in the `recall` function, after the events branch (around line 474):

```python
    if payload.scope == "concepts":
        score = Concept.embedding.cosine_distance(query_embedding).label("score")
        stmt = (
            select(Concept, score)
            .where(Concept.namespace == payload.namespace)
            .where(Concept.embedding.is_not(None))
            .where(~Concept.tags.any("superseded"))
            .order_by(score)
            .limit(candidate_k)
        )

        rows = session.execute(stmt).all()
        items = [
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
            for item, score_value in rows
        ]
        return _apply_reranker(payload.query_text, items, top_k, reranker_provider)
```

**Important:** The events branch in `crud.py:recall()` (line ~449) is currently a bare fallthrough (no `if`/`elif`). You must:
1. Change the events block from a bare fallthrough to `if payload.scope == "events":`
2. Add the concepts block as `elif payload.scope == "concepts":` right after
3. Add a final `else: raise HTTPException(400, "Unknown scope")` at the end

Concrete edit — replace this (around line 449):
```python
    score = Event.embedding.cosine_distance(query_embedding).label("score")
```
With:
```python
    if payload.scope == "events":
        score = Event.embedding.cosine_distance(query_embedding).label("score")
```
And indent the rest of the events block under that `if`. Then add the concepts elif block after the events return statement.

**Step 2: Add concept CRUD functions**

Add at the end of `crud.py`:

```python
def list_concepts(
    session: Session,
    namespace: str,
    concept_type: str | None = None,
    scope: str | None = None,
    include_superseded: bool = False,
    limit: int = 50,
) -> list[dict]:
    """List concepts for a namespace with optional filters."""
    ensure_namespace_exists(session, namespace)

    stmt = select(Concept).where(Concept.namespace == namespace)
    if not include_superseded:
        stmt = stmt.where(~Concept.tags.any("superseded"))
    if concept_type:
        stmt = stmt.where(Concept.type == ConceptType(concept_type))
    if scope:
        stmt = stmt.where(Concept.scope == ConceptScope(scope))
    stmt = stmt.order_by(Concept.confidence.desc()).limit(min(max(limit, 1), 200))

    rows = session.scalars(stmt).all()
    return [
        {
            "id": str(c.id),
            "namespace": c.namespace,
            "type": c.type.value,
            "scope": c.scope.value if hasattr(c.scope, "value") else str(c.scope),
            "scope_ref": c.scope_ref,
            "title": c.title,
            "content": c.content,
            "evidence_ids": [str(eid) for eid in (c.evidence_ids or [])],
            "confidence": c.confidence,
            "version": c.version,
            "parent_id": str(c.parent_id) if c.parent_id else None,
            "tags": list(c.tags or []),
            "created_at": c.created_at,
            "updated_at": c.updated_at,
        }
        for c in rows
    ]


def get_concept_with_history(session: Session, namespace: str, concept_id: str) -> dict | None:
    """Get a concept and its version history chain."""
    ensure_namespace_exists(session, namespace)
    try:
        uid = UUID(concept_id)
    except ValueError:
        return None

    concept = session.scalar(
        select(Concept).where(and_(Concept.id == uid, Concept.namespace == namespace))
    )
    if not concept:
        return None

    def _to_dict(c):
        return {
            "id": str(c.id),
            "namespace": c.namespace,
            "type": c.type.value,
            "scope": c.scope.value if hasattr(c.scope, "value") else str(c.scope),
            "scope_ref": c.scope_ref,
            "title": c.title,
            "content": c.content,
            "evidence_ids": [str(eid) for eid in (c.evidence_ids or [])],
            "confidence": c.confidence,
            "version": c.version,
            "parent_id": str(c.parent_id) if c.parent_id else None,
            "tags": list(c.tags or []),
            "created_at": c.created_at,
            "updated_at": c.updated_at,
        }

    result = _to_dict(concept)

    # Walk parent chain for history
    history = []
    current = concept
    while current.parent_id:
        parent = session.scalar(
            select(Concept).where(Concept.id == current.parent_id)
        )
        if not parent:
            break
        history.append(_to_dict(parent))
        current = parent

    result["history"] = history
    return result
```

**Step 3: Update `get_namespace_stats` to include concept counts**

Add concept counting to the stats function:

```python
    concept_total = session.scalar(
        select(func.count()).select_from(Concept).where(Concept.namespace == namespace)
    ) or 0
    concept_embedded = session.scalar(
        select(func.count()).select_from(Concept).where(
            and_(Concept.namespace == namespace, Concept.embedding.is_not(None))
        )
    ) or 0
```

And add to the return dict:
```python
    "concepts": {"total": concept_total, "embedded": concept_embedded},
```

**Step 4: Commit**

```bash
git add api/app/crud.py
git commit -m "feat: integrate concepts into recall pipeline and add concept CRUD"
```

---

### Task 10: API Endpoints for Concepts

**Files:**
- Modify: `api/app/main.py`

**Step 1: Add concept endpoints to `api/app/main.py`**

Add these endpoints after the existing admin endpoints:

```python
# ── Concepts ───────────────────────────────────────────────────────


@app.get("/concepts", response_model=schemas.ConceptListResponse)
def list_concepts(
    namespace: str = Query(default="default"),
    type: str | None = Query(default=None, alias="concept_type"),
    scope: str | None = Query(default=None),
    include_superseded: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    ensure_namespace_access(auth, namespace, {ApiRole.reader.value, ApiRole.writer.value, ApiRole.admin.value})
    concepts = crud.list_concepts(session, namespace, concept_type=type, scope=scope, include_superseded=include_superseded, limit=limit)
    return schemas.ConceptListResponse(namespace=namespace, total=len(concepts), concepts=concepts)


@app.get("/concepts/{concept_id}", response_model=schemas.ConceptDetailResponse)
def get_concept(
    concept_id: str,
    namespace: str = Query(default="default"),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    ensure_namespace_access(auth, namespace, {ApiRole.reader.value, ApiRole.writer.value, ApiRole.admin.value})
    result = crud.get_concept_with_history(session, namespace, concept_id)
    if not result:
        raise HTTPException(status_code=404, detail="Concept not found")
    return schemas.ConceptDetailResponse(**result)


@app.post("/admin/synthesize", response_model=schemas.SynthesisRunResponse)
def admin_trigger_synthesis(
    namespace: str = Query(default="default"),
    full: bool = Query(default=False, description="Re-synthesize ALL knowledge, not just recent. Use after model upgrades."),
    auth: AuthContext = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Manually trigger a synthesis run for a namespace. Admin only.
    Set full=true to re-examine all knowledge (useful after upgrading the synthesis model).
    When full=true, existing concepts are kept but may be superseded by better versions.
    """
    require_admin(auth)
    ensure_namespace_access(auth, namespace, {ApiRole.admin.value})

    from .synthesis.loop import _run_synthesis_for_namespace

    stats = _run_synthesis_for_namespace(
        namespace=namespace,
        settings=app.state.settings,
        llm_provider=app.state.llm_provider,
        embedding_provider=app.state.embedding_provider,
        full_resynthesis=full,
    )
    return schemas.SynthesisRunResponse(
        namespace=namespace,
        new_concepts=stats["new"],
        updated_concepts=stats["updated"],
        decayed_concepts=stats["decayed"],
    )
```

**Step 2: Commit**

```bash
git add api/app/main.py
git commit -m "feat: add concept list/detail/manual-synthesis endpoints"
```

---

### Task 11: Dependencies + .env Updates

**Files:**
- Modify: `api/requirements.txt` (add numpy)
- Modify: `.env` (add synthesis vars — use `docker compose exec api env` to read current values, NOT the Read tool)

**Step 1: Add numpy to `api/requirements.txt`**

The file is at `/opt/agentssot/api/requirements.txt`. Append this line:
```
numpy>=1.26.0,<2.0
```

**Step 2: Add synthesis env vars to `.env`**

IMPORTANT: The `.env` file cannot be read by the Read tool. Use `docker compose exec api env | grep -i synth` to check current state. Use the Bash tool with `echo` to append:

```bash
cat >> /opt/agentssot/.env << 'EOF'

# Synthesis (conceptual memory / Hive Cortex)
SYNTHESIS_ENABLED=true
SYNTHESIS_MODEL=qwen3.5:cloud
SYNTHESIS_FALLBACK_MODEL=qwen3:latest
SYNTHESIS_SCHEDULE_HOUR=3
EOF
```

**Step 3: Rebuild and test**

```bash
cd /opt/agentssot && docker compose up -d --build api
```

Wait 20s for healthcheck, then verify:
```bash
curl -s http://localhost:8088/health | python3 -m json.tool
```

Expected: `"synthesis_enabled": true` in the response.

**Step 4: Commit**

```bash
git add api/requirements.txt
git commit -m "feat: add numpy dependency and enable synthesis"
```

NOTE: Do NOT `git add .env` — it contains secrets.

---

### Task 12: Verify Synthesis Models in Ollama

**Step 1: Verify qwen3.5:cloud is available**

`qwen3.5:cloud` is an Ollama cloud-routed model. Test it:

```bash
curl -s http://localhost:11434/api/chat -d '{
  "model": "qwen3.5:cloud",
  "stream": false,
  "messages": [{"role": "user", "content": "Say hello in one word"}]
}' | python3 -m json.tool
```

If this fails (e.g. model not found), pull it:
```bash
ollama pull qwen3.5:cloud
```

**Step 2: Verify fallback model (qwen3:latest) is available**

This should already be loaded (it's the existing compaction model):
```bash
ollama list | grep qwen3
```

**Step 3: Test both models respond**

```bash
# Primary (cloud)
curl -s http://localhost:11434/api/chat -d '{"model":"qwen3.5:cloud","stream":false,"messages":[{"role":"user","content":"Reply OK"}]}' | python3 -c "import sys,json; print(json.load(sys.stdin)['message']['content'][:50])"

# Fallback (local)
curl -s http://localhost:11434/api/chat -d '{"model":"qwen3:latest","stream":false,"messages":[{"role":"user","content":"Reply OK"}]}' | python3 -c "import sys,json; print(json.load(sys.stdin)['message']['content'][:50])"
```

---

### Task 13: Integration Test — Manual Synthesis Run

**Step 1: Trigger a manual synthesis via API**

```bash
# Get admin API key from agent.json
ADMIN_KEY=$(python3 -c "import json; print(json.load(open('/home/hari/.claude/agentssot/local/admin.json'))['admin_api_key'])")

# Trigger synthesis on claude-shared namespace
curl -s -X POST "http://localhost:8088/admin/synthesize?namespace=claude-shared" \
  -H "X-API-Key: $ADMIN_KEY" | python3 -m json.tool
```

**Step 2: Verify concepts were created**

```bash
curl -s "http://localhost:8088/concepts?namespace=claude-shared" \
  -H "X-API-Key: $ADMIN_KEY" | python3 -m json.tool
```

**Step 3: Test concept recall blending**

```bash
curl -s -X POST http://localhost:8088/recall \
  -H "X-API-Key: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"namespace":"claude-shared","scope":"concepts","query_text":"Docker deployment patterns"}' \
  | python3 -m json.tool
```

**Step 4: Check stats include concepts**

```bash
curl -s "http://localhost:8088/admin/stats?namespace=claude-shared" \
  -H "X-API-Key: $ADMIN_KEY" | python3 -m json.tool
```

**Step 5: Commit final state**

```bash
git add -A
git commit -m "feat: hive cortex v1 — conceptual memory layer complete"
```

---

### Task 14: Update startup.py — Ensure embedding dim covers concepts table

**Files:**
- Modify: `api/app/startup.py`

**Step 1: Add concepts table to `_ensure_embedding_dim`**

In the `_ensure_embedding_dim` function, update the SQL query to include `'concepts'` in the table list:

```python
WHERE c.relname IN ('knowledge_items', 'requirements', 'events', 'concepts')
```

And add the ALTER TABLE for concepts:
```python
session.execute(text(f"ALTER TABLE concepts ALTER COLUMN embedding TYPE VECTOR({desired})"))
```

Also drop any HNSW index on concepts if present:
```python
session.execute(text("DROP INDEX IF EXISTS idx_concepts_embedding_hnsw"))
```

**Step 2: Commit**

```bash
git add api/app/startup.py
git commit -m "fix: include concepts table in embedding dimension migration"
```
