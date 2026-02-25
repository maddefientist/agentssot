from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class EntityIn(BaseModel):
    slug: str
    type: Literal["project", "person", "agent", "document", "integration", "other"]
    name: str
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RequirementIn(BaseModel):
    project_slug: str | None = None
    owner_entity_slug: str | None = None
    title: str
    body: str | None = None
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    status: Literal["draft", "proposed", "in_progress", "blocked", "done", "archived"] = "draft"
    context_snippet: str | None = None
    tags: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None


class KnowledgeItemIn(BaseModel):
    project_slug: str | None = None
    entity_slug: str | None = None
    content: str
    source: str | None = None
    source_ref: str | None = None
    tags: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None


class EventIn(BaseModel):
    project_slug: str | None = None
    agent_slug: str | None = None
    type: Literal["note", "decision", "directive", "action", "result", "error"] = "note"
    title: str
    body: str | None = None
    context_snippet: str | None = None
    session_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None


class IngestRequest(BaseModel):
    namespace: str = "default"
    entities: list[EntityIn] = Field(default_factory=list)
    requirements: list[RequirementIn] = Field(default_factory=list)
    knowledge_items: list[KnowledgeItemIn] = Field(default_factory=list)
    events: list[EventIn] = Field(default_factory=list)


class IngestResponse(BaseModel):
    namespace: str
    counts: dict[str, int]


class QueryRecord(BaseModel):
    id: str
    kind: Literal["entity", "requirement", "knowledge_item", "event"]
    title: str
    snippet: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class QueryResponse(BaseModel):
    namespace: str
    total: int
    results: list[QueryRecord]


class RecallRequest(BaseModel):
    namespace: str = "default"
    scope: Literal["knowledge", "requirements", "events", "concepts", "all"] = "knowledge"
    query_text: str | None = None
    query_embedding: list[float] | None = None
    top_k: int | None = None
    project_slug: str | None = None
    entity_slug: str | None = None
    session_id: str | None = None
    agent_key: str | None = None


class RecallItem(BaseModel):
    id: str
    scope: Literal["knowledge", "requirements", "events", "concepts"]
    score: float
    reranker_score: float | None = None
    snippet: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    concept_type: Literal["mental_model", "relationship", "principle", "skill"] | None = None
    confidence: float | None = None
    trigger: str | None = None
    action: str | None = None
    success_hint: str | None = None


class RecallResponse(BaseModel):
    namespace: str
    scope: Literal["knowledge", "requirements", "events", "concepts", "all"]
    top_k: int
    items: list[RecallItem]


class SummarizeClearRequest(BaseModel):
    namespace: str = "default"
    session_id: str
    project_slug: str | None = None
    max_events: int = 500


class SummarizeClearResponse(BaseModel):
    namespace: str
    session_id: str
    archived_events: int
    summary_knowledge_item_id: str


class NamespaceCreateRequest(BaseModel):
    name: str


class NamespaceCreateResponse(BaseModel):
    name: str
    created_at: datetime


class ApiKeyCreateRequest(BaseModel):
    name: str
    role: Literal["reader", "writer", "admin"]
    namespaces: list[str] = Field(default_factory=lambda: ["default"])


class ApiKeyCreateResponse(BaseModel):
    id: str
    name: str
    role: Literal["reader", "writer", "admin"]
    namespaces: list[str]
    is_active: bool
    created_at: datetime
    api_key: str


class ApiKeyListItem(BaseModel):
    id: str
    name: str
    role: Literal["reader", "writer", "admin"]
    namespaces: list[str]
    is_active: bool
    created_at: datetime
    key_preview: str


class DeleteItemsRequest(BaseModel):
    namespace: str = "default"
    ids: list[str] = Field(min_length=1, max_length=100)


class DeleteItemsResponse(BaseModel):
    namespace: str
    deleted: int


class EnrollmentTokenCreateRequest(BaseModel):
    role: Literal["reader", "writer"] = "writer"
    namespaces: list[str] = Field(default_factory=lambda: ["default"])
    name_hint: str | None = None
    max_uses: int = Field(default=1, ge=1, le=100)
    expires_in_hours: int | None = Field(default=None, ge=1, le=8760)


class EnrollmentTokenCreateResponse(BaseModel):
    id: str
    token: str
    role: Literal["reader", "writer"]
    namespaces: list[str]
    name_hint: str | None
    max_uses: int
    expires_at: datetime | None


class EnrollmentTokenListItem(BaseModel):
    id: str
    role: Literal["reader", "writer"]
    namespaces: list[str]
    name_hint: str | None
    max_uses: int
    times_used: int
    expires_at: datetime | None
    is_active: bool
    created_at: datetime


class EnrollRequest(BaseModel):
    token: str
    name: str = Field(min_length=1, max_length=100)


class EnrollResponse(BaseModel):
    api_key: str
    name: str
    role: Literal["reader", "writer", "admin"]
    namespaces: list[str]


class BackfillEmbeddingsRequest(BaseModel):
    namespace: str = "default"
    scope: Literal["knowledge", "requirements", "events", "concepts"] = "knowledge"
    limit: int = 500
    batch_size: int = 50
    dry_run: bool = False


class BackfillEmbeddingsResponse(BaseModel):
    namespace: str
    scope: Literal["knowledge", "requirements", "events", "concepts"]
    updated: int
    skipped: int
    dry_run: bool


class DedupRequest(BaseModel):
    namespace: str = "default"
    dry_run: bool = True


class DedupResponse(BaseModel):
    namespace: str
    duplicate_groups: int
    deleted: int
    dry_run: bool


class ConceptOut(BaseModel):
    id: str
    namespace: str
    type: Literal["mental_model", "relationship", "principle", "skill"]
    scope: Literal["global", "project", "device"]
    scope_ref: str | None = None
    title: str
    content: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float
    version: int
    parent_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    trigger: str | None = None
    action: str | None = None
    success_hint: str | None = None
    confirming_agents: list[str] = Field(default_factory=list)
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
    feedback_adjustments: int = 0


class ItemCountDetail(BaseModel):
    total: int
    embedded: int


class NamespaceStatsResponse(BaseModel):
    namespace: str
    entities: int
    knowledge_items: ItemCountDetail
    requirements: ItemCountDetail
    events: ItemCountDetail
    concepts: ItemCountDetail | None = None


class AutoEnrollRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    passphrase: str = ""


class AgentConfig(BaseModel):
    base_url: str
    api_key: str
    device_name: str
    default_namespace: str
    default_scope: str
    namespaces: list[str]


class AutoEnrollResponse(BaseModel):
    api_key: str
    name: str
    role: Literal["reader", "writer", "admin"]
    namespaces: list[str]
    agent_config: AgentConfig


class AgentProfileResponse(BaseModel):
    agent_key: str
    namespace: str
    device_name: str | None = None
    model_hint: str | None = None
    strengths: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    total_recalls: int = 0
    total_feedback: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class FeedbackRequest(BaseModel):
    signal: Literal["useful", "noted", "wrong"]
    concept_id: str | None = None
    knowledge_item_id: str | None = None
    query: str | None = None
    note: str | None = None
    session_id: str | None = None
    agent_key: str | None = None


class FeedbackResponse(BaseModel):
    concept_id: str = ""
    concept_title: str = ""
    knowledge_item_id: str = ""
    signal: str
    confidence: float = 0.0
    strength: float = 0.0


class SessionCompleteRequest(BaseModel):
    session_id: str
    conversation_summary: str
    recalled_concept_ids: list[str] = []
    agent_key: str | None = None


class SessionCompleteResponse(BaseModel):
    session_id: str
    facts_extracted: int
    recall_events_completed: int
