"""Learning intake distill endpoint (WU3).

POST /api/v1/distill: extract a transcript from a source, distill it into
atomic lessons via the LLM provider, and return provenance + transcript_ref +
lessons. Never imports/calls knowledge.py, schemas.py, /ingest, or a DB session.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from ..intake.distill import Lesson, parse_lessons
from ..intake.extract import IntakeExtractionError, extract
from ..llm import LLMProviderError
from ..security import AuthContext, require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["intake"])

MediaType = Literal["video", "audio", "article", "thread"]
MemoryType = Literal["skill", "decision", "fact"]


class DistillRequest(BaseModel):
    source_url: str | None = None
    text: str | None = None
    media_type: MediaType
    title: str | None = None

    @model_validator(mode="after")
    def require_source_or_text(self) -> "DistillRequest":
        if self.media_type in ("video", "audio"):
            if not self.source_url or not self.source_url.strip():
                raise ValueError("source_url is required for video/audio media")
        else:  # article / thread
            if not self.text or not self.text.strip():
                raise ValueError("text is required for article/thread media")
        return self


class Provenance(BaseModel):
    source_url: str | None = None
    media_type: MediaType
    captured_at: str
    title: str | None = None
    duration: float | None = None


class LessonResponse(BaseModel):
    claim: str
    citation: str
    memory_type: MemoryType = "skill"
    confidence: float = Field(ge=0.0, le=1.0)


class DistillResponse(BaseModel):
    provenance: Provenance
    transcript_ref: str
    lessons: list[LessonResponse]


@router.post("/distill", response_model=DistillResponse)
def distill_source(
    payload: DistillRequest,
    request: Request,
    auth: AuthContext = Depends(require_api_key),
) -> DistillResponse:
    settings = request.app.state.settings
    try:
        transcript, provenance = extract(
            payload.source_url,
            payload.text,
            payload.media_type,
            stt_url=settings.voice_stack_stt_url,
        )

        if payload.title:
            provenance["title"] = payload.title

        raw = request.app.state.llm_provider.distill(transcript)

        lessons = parse_lessons(raw)

        transcript_ref = "sha256:" + hashlib.sha256(
            transcript.encode("utf-8")
        ).hexdigest()
    except IntakeExtractionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("unexpected error during /distill")
        raise HTTPException(status_code=500, detail="internal error") from exc

    return DistillResponse(
        provenance=Provenance(**provenance),
        transcript_ref=transcript_ref,
        lessons=[LessonResponse(**lesson) for lesson in lessons],
    )