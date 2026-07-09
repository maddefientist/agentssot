# Learning Intake Pipeline — Implementation Plan (chain dispatch)

*Status: ready to dispatch · Planner: gpt-orchestrate (GPT-5.5, thinking=high) · Reviewed by: Opus (architect) · 2026-07-09*
*Spec: `docs/superpowers/specs/2026-07-09-learning-intake-pipeline-design.md`*

## Architect review (Opus) — verified before endorsing

Verified against live repo @ `0333b7a`:
- ✅ **Settings pattern correct.** `api/app/settings.py` = pydantic `BaseSettings` + `Field(default=…, alias="ENV_VAR")`. GPT-5.5's `voice_stack_stt_url` field matches house style.
- ✅ **Docker→host networking already solved.** `docker-compose.yml` already has `extra_hosts: "host.docker.internal:host-gateway"` (for Ollama). The default `http://host.docker.internal:8402/transcribe` reaches voice-stack from inside the container with **no compose change**. (My original brief's `127.0.0.1:8402` would have been unreachable from the container — GPT-5.5's correction stands.)
- ✅ `api/requirements.txt` exists, `httpx` present; only `yt-dlp` needs adding.

Accepted deviations / notes for the coders:
- **`transcript_ref` = sha256 of transcript, NOT stored** (spec §Components said "stored, so we can re-distill later"). Accepted for v1 — re-distill re-extracts from `source_url` in provenance. If persistent transcripts are wanted later, that's a v2 follow-up.
- **`python-multipart` dependency is UNNECESSARY** — it's for *server-side* form parsing; `/distill` takes JSON and the STT client uses `httpx` multipart (client-side, no dep needed). Coder may drop it; harmless if kept.
- **Dispatch one-liners are illustrative.** The `chain.sh glm-implement "WU1 distill core"` form in the ROUTING section is shorthand — at dispatch each coder brick MUST receive the FULL WU brief section below, not the one-liner, or it will starve.

Escalation triggers to Opus (unchanged): sync `/distill` too slow for 30-min video → async job/poll redesign; any `LLMProvider` call-site breakage from the new `distill()` signature.

---

## DECISIONS (from GPT-5.5)

- Add `LLMProvider.distill(transcript, model=None) -> str` beside `summarize`; mirrors JSON-lines `synthesize_concepts`.
- `voice_stack_stt_url` in `api/app/settings.py`; no hardcoded service URL.
- `intake_router` self-prefixed `APIRouter(prefix="/api/v1")` + bare `app.include_router(intake_router)`; avoids double prefix.
- `/distill` returns parsed lessons only; never imports `knowledge.py`/ingest types; preserves Corra promotion boundary.
- Extraction/distillation errors → typed non-2xx; no silent empty successes.

---

## WU1 — Distill core
**Brick pairing:** `glm-implement -> kimi-review` · **Branch:** `feat/learning-intake-distill`
**Touches only:** `api/app/llm/base.py`, `api/app/llm/ollama_provider.py`, `api/app/llm/openai_provider.py`, `api/app/intake/__init__.py`, `api/app/intake/distill.py`, `api/tests/test_intake_distill.py`

**Signatures:**
- `base.py` abstract: `def distill(self, transcript: str, model: str | None = None) -> str: raise NotImplementedError`
- both providers implement `def distill(self, transcript, model=None) -> str`
- `api/app/intake/distill.py`:
```python
from __future__ import annotations
import json
from typing import Any, Literal, TypedDict
MemoryType = Literal["skill", "decision", "fact"]
class Lesson(TypedDict):
    claim: str
    citation: str
    memory_type: MemoryType
    confidence: float
def parse_lessons(raw: str) -> list[Lesson]: ...
```

**Literal distill system prompt:**
```text
You are a learning-intake distiller. Extract atomic, reusable lessons from the transcript.

Output ONLY valid JSON lines, one JSON object per line. No markdown, no commentary.
Each object MUST contain:
- claim: concise actionable lesson or factual takeaway
- citation: source anchor supporting the claim; use a timestamp for audio/video when available, otherwise a quote or paragraph anchor
- memory_type: "skill" | "decision" | "fact"; default to "skill" for best-practices
- confidence: number from 0.0 to 1.0

Rules:
- Prefer specific best-practices over generic summaries.
- Each lesson must stand alone and cite evidence from the source.
- Do not invent claims not supported by the transcript.
- If no useful lessons exist, output an empty line.
```

**Implementation requirements:**
1. `OllamaLLMProvider.distill` mirrors `summarize`: POST `f"{self.base_url}/api/chat"`, `stream: False`, `model: model or self.model`, system prompt above, user=transcript.
2. Include `options: {"num_ctx": 8192}` for Ollama.
3. `OpenAILLMProvider.distill` POSTs `https://api.openai.com/v1/chat/completions`, `temperature: 0.2`.
4. Provider failures raise `LLMProviderError`.
5. `parse_lessons` tolerant JSON-lines: skip malformed lines/objects, never raise for one bad line.
6. Missing/invalid `memory_type` → `"skill"`.
7. `confidence` clamped to `0.0..1.0`, default `0.5` if missing/invalid.
8. Drop lessons missing non-empty `claim` or `citation`.

**Acceptance:** (1) canned JSON-lines → `Lesson` dicts; (2) malformed line skipped, no raise; (3) missing memory_type → skill; (4) invalid confidence → 0.5; (5) provider `distill` returns stripped content, raises `LLMProviderError` on bad response.

**DO NOT TOUCH:** `knowledge.py`, `schemas.py`, `ingest_tiered`, dedup gate, `synthesis/`, reranker/embedding config. `/distill` must not import/call ingest.

**Verify:**
```bash
pytest api/tests/test_intake_distill.py -q
git diff -- api/app/routers/knowledge.py api/app/schemas.py --exit-code
grep -R "ingest_tiered\|/api/v1/ingest" api/app/intake api/app/llm || true
```

---

## WU2 — Extraction adapter
**Brick pairing:** `glm-implement -> kimi-review` · run after WU1 (avoid package-init conflict on `intake/`)
**Touches only:** `api/app/intake/extract.py`, `api/app/settings.py`, `api/requirements.txt`, `api/Dockerfile`, `api/tests/test_intake_extract.py`

**Signatures:**
```python
# api/app/intake/extract.py
from __future__ import annotations
import logging, shutil, subprocess, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
import httpx
MediaType = Literal["video", "audio", "article", "thread"]
class IntakeExtractionError(RuntimeError): pass
def validate_source_url(source_url: str | None) -> str: ...
def extract(source_url: str | None, text: str | None, media_type: MediaType, *, stt_url: str) -> tuple[str, dict]: ...
```

**Settings change (`api/app/settings.py`):**
```python
voice_stack_stt_url: str = Field(default="http://host.docker.internal:8402/transcribe", alias="VOICE_STACK_STT_URL")
```

**Subprocess argv (list only, never shell=True):**
```python
yt_dlp_argv = ["yt-dlp", "-f", "bestaudio/best", "-o", str(download_path), source_url]
ffmpeg_argv = [ffmpeg_bin, "-y", "-i", str(download_path), "-ar", "16000", "-ac", "1", str(wav_path)]
```

**Dependency changes:** add `yt-dlp>=2025.1.15` to `api/requirements.txt` (drop the `python-multipart` line GPT-5.5 suggested — unnecessary, see architect note). `api/Dockerfile` apt install must include `ffmpeg`.

**Implementation requirements:**
1. `article`/`thread`: require non-empty `text`, return `(text.strip(), provenance)`.
2. `video`/`audio`: require `source_url`, validate scheme ∈ {http, https} (SSRF guard).
3. `shutil.which("yt-dlp")`, `shutil.which("ffmpeg") or "/usr/bin/ffmpeg"`; fail loudly if missing.
4. subprocess `shell=False`, list argv, `check=False`, capture stdout/stderr, timeout.
5. non-zero return → `IntakeExtractionError` with stderr excerpt.
6. STT multipart: `httpx.post(stt_url, files={"file": ("audio.wav", fh, "audio/wav")}, timeout=300)`.
7. STT JSON keys tried in order: `"text"`, `"transcript"`, `"result"`; none non-empty → raise.
8. provenance includes `source_url`, `media_type`, `captured_at` (ISO8601 UTC).
9. no silent empty transcript.

**Acceptance:** (1) text passthrough returns stripped transcript+provenance; (2) invalid scheme raises; (3) missing yt-dlp/ffmpeg raises typed; (4) mocked subprocess+httpx success returns transcript; (5) mocked failures raise typed; (6) no real network/GPU in unit tests.

**DO NOT TOUCH:** `knowledge.py`, `schemas.py`, ingest path, Corra wiring, async job queue.

**Verify:**
```bash
pytest api/tests/test_intake_extract.py -q
git diff -- api/app/routers/knowledge.py api/app/schemas.py --exit-code
grep -R "shell=True" api/app/intake && exit 1 || true
```

---

## WU3 — Endpoint + wiring
**Brick pairing:** `glm-implement -> kimi-review -> gpt-review` · depends on WU1+WU2
**Touches only:** `api/app/routers/intake.py`, `api/app/main.py`, `api/tests/test_distill_endpoint.py`
**⚠ main.py is the ONLY shared-file edit — no other WU may touch it; sequence WU3 last.**

**Router + models:**
```python
# api/app/routers/intake.py
from __future__ import annotations
import hashlib, logging
from datetime import datetime, timezone
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, model_validator
from ..intake.distill import Lesson, parse_lessons
from ..intake.extract import IntakeExtractionError, extract
from ..llm import LLMProviderError
from ..security import AuthContext, require_api_key

router = APIRouter(prefix="/api/v1", tags=["intake"])
MediaType = Literal["video", "audio", "article", "thread"]
MemoryType = Literal["skill", "decision", "fact"]

class DistillRequest(BaseModel):
    source_url: str | None = None
    text: str | None = None
    media_type: MediaType
    title: str | None = None
    @model_validator(mode="after")
    def require_source_or_text(self) -> "DistillRequest": ...

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
def distill_source(payload: DistillRequest, request: Request, auth: AuthContext = Depends(require_api_key)) -> DistillResponse: ...
```

**`main.py` wiring:** `from .routers.intake import router as intake_router` (next to other `.routers.<name>` imports); `app.include_router(intake_router)` (after the `knowledge_router` include).

**Implementation requirements:**
1. `DistillRequest` validation: video/audio require `source_url`; article/thread require non-empty `text`.
2. `settings = request.app.state.settings`; `transcript, provenance = extract(payload.source_url, payload.text, payload.media_type, stt_url=settings.voice_stack_stt_url)`.
3. add `title` to provenance if provided.
4. `raw = request.app.state.llm_provider.distill(transcript)`.
5. `lessons = parse_lessons(raw)`.
6. `transcript_ref = "sha256:" + hashlib.sha256(transcript.encode("utf-8")).hexdigest()` (not stored — see architect note).
7. Error mapping: `IntakeExtractionError` → 502; `LLMProviderError` → 502; unexpected → log + 500.
8. ZERO lessons = valid 200 with `lessons: []`; never fabricate.
9. never import/call `knowledge.py`, `schemas.py`, `/ingest`, or DB session.

**Acceptance:** (1) auth required via `X-API-Key`; (2) text request (mocked extract/provider) returns provenance+transcript_ref+lessons; (3) zero lessons → 200 empty list; (4) extraction failure → non-2xx; (5) distill failure → non-2xx; (6) `main.py` registers `/api/v1/distill` exactly once.

**DO NOT TOUCH:** ingest route, hive writes, Corra bot, DB models, synthesis, embeddings/reranker.

**Verify:**
```bash
pytest api/tests/test_distill_endpoint.py -q
pytest api/tests/test_intake_distill.py api/tests/test_intake_extract.py api/tests/test_distill_endpoint.py -q
git diff -- api/app/routers/knowledge.py api/app/schemas.py --exit-code
grep -R "ingest_tiered\|knowledge_router\|/api/v1/ingest" api/app/routers/intake.py api/app/intake && exit 1 || true
```

---

## WU4 — Manual live acceptance harness (optional, if budget)
**Brick pairing:** `glm-quick -> kimi-review` · depends on WU1-WU3
**Touches only:** `api/tests/smoke/test_live_intake_distill.py`

1. skipped unless `RUN_LIVE_INTAKE=1`.
2. uses `SSOT_TEST_URL`, `SSOT_API_KEY`, `LIVE_INTAKE_URL`.
3. do NOT hardcode the Anthropic video URL — take from env; default only if operator-provided.
4. POST `{"source_url": live_url, "media_type": "video", "title": "Live intake smoke"}`, header `{"X-API-Key": api_key}`.
5. assert 200, `transcript_ref` starts `sha256:`, `lessons` present, each has claim/citation/memory_type/confidence.

**Acceptance:** skips by default in CI; does not call ingest; works against live `:8088` with env set; failure output includes response-body excerpt.

**Verify:**
```bash
pytest api/tests/smoke/test_live_intake_distill.py -q   # skips by default
git diff -- api/app/routers/knowledge.py api/app/schemas.py --exit-code
```

---

## Dispatch sequence
Sequential (`&&`) — each coder brick receives its FULL WU section above as the brief, NOT a one-liner:
1. `deepseek-plan` — sanity-check this plan (file-collision discipline, SSRF/error handling). No edits.
2. `glm-implement` WU1 → `kimi-review` WU1
3. `glm-implement` WU2 → `kimi-review` WU2
4. `glm-implement` WU3 → `kimi-review` WU3 → `gpt-review` WU3 (external-input/SSRF seam)
5. `glm-quick` WU4 → `kimi-review` WU4 (if budget)

After all coder bricks: run `state-return-check.sh` on each `progress.md`, then Opus verifies (read diffs, run full `pytest api/tests -q`), then land on `main`.

## Follow-on (separate dispatch, NOT this plan)
Corra-side (moni, branch `corra`): on a source, call `/api/v1/distill`, map each lesson → `memory_ingest` with provenance into `corra` namespace → owner `/promote` to `claude-shared`.
