import enum
from datetime import datetime
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Boolean, DateTime, Enum, ForeignKey, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .settings import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    pass


class EntityType(str, enum.Enum):
    project = "project"
    person = "person"
    agent = "agent"
    document = "document"
    integration = "integration"
    other = "other"


class RequirementPriority(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class RequirementStatus(str, enum.Enum):
    draft = "draft"
    proposed = "proposed"
    in_progress = "in_progress"
    blocked = "blocked"
    done = "done"
    archived = "archived"


class EventType(str, enum.Enum):
    note = "note"
    decision = "decision"
    directive = "directive"
    action = "action"
    result = "result"
    error = "error"


class ApiRole(str, enum.Enum):
    reader = "reader"
    writer = "writer"
    admin = "admin"


class Namespace(Base):
    __tablename__ = "namespaces"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    name: Mapped[str] = mapped_column(Text, nullable=False)
    key_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[ApiRole] = mapped_column(Enum(ApiRole, name="api_role", create_type=False), nullable=False)
    namespaces: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list, server_default=text("ARRAY['default']::TEXT[]")
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("namespace", "slug", name="uq_entities_namespace_slug"),)

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    namespace: Mapped[str] = mapped_column(Text, ForeignKey("namespaces.name", ondelete="CASCADE"), nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[EntityType] = mapped_column(Enum(EntityType, name="entity_type", create_type=False), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Requirement(Base):
    __tablename__ = "requirements"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    namespace: Mapped[str] = mapped_column(Text, ForeignKey("namespaces.name", ondelete="CASCADE"), nullable=False)
    project_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("entities.id", ondelete="SET NULL"))
    owner_entity_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("entities.id", ondelete="SET NULL")
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str | None] = mapped_column(Text)
    priority: Mapped[RequirementPriority] = mapped_column(
        Enum(RequirementPriority, name="requirement_priority", create_type=False),
        nullable=False,
        default=RequirementPriority.medium,
        server_default=RequirementPriority.medium.value,
    )
    status: Mapped[RequirementStatus] = mapped_column(
        Enum(RequirementStatus, name="requirement_status", create_type=False),
        nullable=False,
        default=RequirementStatus.draft,
        server_default=RequirementStatus.draft.value,
    )
    context_snippet: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list, server_default=text("ARRAY[]::TEXT[]"))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.embedding_dim), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class KnowledgeItem(Base):
    __tablename__ = "knowledge_items"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    namespace: Mapped[str] = mapped_column(Text, ForeignKey("namespaces.name", ondelete="CASCADE"), nullable=False)
    project_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("entities.id", ondelete="SET NULL"))
    entity_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("entities.id", ondelete="SET NULL"))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(Text)
    source_ref: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list, server_default=text("ARRAY[]::TEXT[]"))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.embedding_dim), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class EnrollmentToken(Base):
    __tablename__ = "enrollment_tokens"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    token_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[ApiRole] = mapped_column(Enum(ApiRole, name="api_role", create_type=False), nullable=False)
    namespaces: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list, server_default=text("ARRAY['default']::TEXT[]")
    )
    name_hint: Mapped[str | None] = mapped_column(Text, nullable=True)
    max_uses: Mapped[int] = mapped_column(nullable=False, default=1, server_default=text("1"))
    times_used: Mapped[int] = mapped_column(nullable=False, default=0, server_default=text("0"))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Event(Base):
    __tablename__ = "events"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    namespace: Mapped[str] = mapped_column(Text, ForeignKey("namespaces.name", ondelete="CASCADE"), nullable=False)
    project_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("entities.id", ondelete="SET NULL"))
    agent_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("entities.id", ondelete="SET NULL"))
    type: Mapped[EventType] = mapped_column(
        Enum(EventType, name="event_type", create_type=False),
        nullable=False,
        default=EventType.note,
        server_default=EventType.note.value,
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str | None] = mapped_column(Text)
    context_snippet: Mapped[str | None] = mapped_column(Text)
    session_id: Mapped[str | None] = mapped_column(Text)
    is_archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("false"))
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list, server_default=text("ARRAY[]::TEXT[]"))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.embedding_dim), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
