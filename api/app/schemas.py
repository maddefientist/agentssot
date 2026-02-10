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
    scope: Literal["knowledge", "requirements", "events"] = "knowledge"
    query_text: str | None = None
    query_embedding: list[float] | None = None
    top_k: int | None = None
    project_slug: str | None = None
    entity_slug: str | None = None


class RecallItem(BaseModel):
    id: str
    scope: Literal["knowledge", "requirements", "events"]
    score: float
    reranker_score: float | None = None
    snippet: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class RecallResponse(BaseModel):
    namespace: str
    scope: Literal["knowledge", "requirements", "events"]
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


class BackfillEmbeddingsRequest(BaseModel):
    namespace: str = "default"
    scope: Literal["knowledge", "requirements", "events"] = "knowledge"
    limit: int = 500
    batch_size: int = 50
    dry_run: bool = False


class BackfillEmbeddingsResponse(BaseModel):
    namespace: str
    scope: Literal["knowledge", "requirements", "events"]
    updated: int
    skipped: int
    dry_run: bool
