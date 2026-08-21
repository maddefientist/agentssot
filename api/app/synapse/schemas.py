from datetime import datetime

from pydantic import BaseModel, Field


class SessionRegister(BaseModel):
    session_id: str = Field(min_length=1)
    host: str = Field(min_length=1)
    cwd: str = Field(min_length=1)
    repo: str | None = None
    agent: str = Field(min_length=1)
    current_file: str | None = None
    current_op: str | None = None


class SessionHeartbeat(BaseModel):
    session_id: str = Field(min_length=1)
    current_file: str | None = None
    current_op: str | None = None


class EventCreate(BaseModel):
    session_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    file: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    payload: dict | None = None


class SessionOut(BaseModel):
    session_id: str
    host: str
    cwd: str
    repo: str | None
    agent: str
    started_at: datetime
    last_seen: datetime
    current_file: str | None
    current_op: str | None
    # Computed convenience fields (no DB migration: both source timestamps
    # already exist). None on plain `model_validate(row)` construction --
    # routes.py sets them explicitly after validation.
    age_seconds: float | None = None
    idle_seconds: float | None = None

    model_config = {"from_attributes": True}


class EventOut(BaseModel):
    id: int
    session_id: str
    ts: datetime
    kind: str
    file: str | None
    line_start: int | None
    line_end: int | None
    payload: dict | None

    model_config = {"from_attributes": True}


class CollisionOut(BaseModel):
    session_id: str
    host: str
    cwd: str
    last_event_ts: datetime
    kind: str
