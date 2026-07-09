# Learning Intake Pipeline — Design

*Status: approved for planning · Author: Opus (agentssot session) · 2026-07-09*

## Problem

Valuable **external** content — a 30-min Anthropic memory video, articles, X/IG
threads — should flow into AgentSSOT as **distilled, cited lessons** automatically,
instead of depending on the operator hand-teaching them. The storage/typing/summarize
machinery already exists and is reusable; the gap is a front-end that turns an
arbitrary source into a transcript, distills atomic *lessons* (not a prose summary),
and hands them into the existing ingest path with provenance.

## What already exists (reuse, do not rebuild)

Verified against `/opt/agentssot` @ `f63e771`:

| Capability | Location | Reuse as |
|---|---|---|
| `POST /ingest` → `ingest_tiered()` | `api/app/routers/knowledge.py:131` | Final sink for lessons |
| Semantic dedup / near-dup review gate (0.985 collapse) | `api/app/routers/knowledge.py:111` | Idempotent re-ingest |
| Loud fail on embedding-backend error (503) | `api/app/routers/knowledge.py:162` | Reused as-is |
| Auto-classify on ingest | `api/app/llm/classifier.py:134` | Final say on lesson typing |
| Layer compute (abstract/summary) | `api/app/llm/layer_compute.py:43` | Reused as-is |
| `summarize(transcript)` LLM abstraction | `api/app/llm/base.py:17` (ollama + openai impls) | Base for a new *distill* prompt variant |
| Background synthesis loop | `api/app/synthesis/loop.py` | Downstream, untouched |
| Whisper STT | voice-stack `:8402` (hari, 3090) — **separate, not wired** | Called by the extractor |
| **CORRABOX / Corra** knowledge partner | moni VM, branch `corra`, `@corraboxbot` | **Orchestrator + curation/promotion gate** |

**Corra already provides** (partner layer, 2026-06-25): `web-fetch`, `read-shared-links`,
`capture-conclusions`, `memory_ingest`, an owner **promotion gate** to `claude-shared`
(`/knowledge` → `/promote N` / `/kedit` / `/kreject`, provenance tag `corra:candidate:<id>`,
post-before-mark, idempotent), interest signals (`/more`,`/less`), and `corra_supersede`.
Her hive key `corra-agent` is writer on `[corra, claude-shared]`. She is on **moni**;
voice-stack + AgentSSOT are on **hari**.

## Approach (decided)

- **Wiring direction:** *Corra orchestrates.* She owns the workflow, the Telegram
  trigger, and the promotion gate; AgentSSOT exposes a new stateless `/distill`
  endpoint she calls. (Alternative — a self-contained AgentSSOT `/learn` route —
  rejected because it duplicates Corra's fetch + curation.)
- **v1 acceptance target:** *the video path.* The 30-min Anthropic memory video is
  the headline acceptance test, so A/V → STT is in v1 scope (not deferred).

## Architecture

Responsibility boundary (two hard invariants):

1. **Corra never touches ffmpeg/STT.** For media she passes the URL onward; for
   text she may pass already-fetched text.
2. **`/distill` never writes to hive.** It returns lessons; Corra ingests them
   through her existing promotion gate.

```
Telegram / URL ─► Corra (moni)  ── orchestrate + curate ──┐
                    │ POST /distill {source_url|text, media_type}
                    ▼
        AgentSSOT /distill (hari, co-located w/ voice-stack)
          media:  yt-dlp ─► ffmpeg -ar 16000 -ac 1 ─► POST :8402/transcribe
          text:   passthrough
                    │  transcript
                    ▼
          distill prompt (new summarize() variant) ─► lessons[] + provenance
                    │ returns to Corra
                    ▼
        Corra memory_ingest ─► POST /ingest (ingest_tiered)
          → dedup gate → classify → corra namespace
                    │
                    ▼  owner: /knowledge → /promote N
              claude-shared (recallable, cited)
```

Net-new code = **one endpoint (`/distill`) + one extraction adapter + one distill
prompt.** Everything downstream is reused untouched.

## Components

### 1. Extraction adapter (`api/app/intake/extract.py`, new)
- `extract(source_url|text, media_type) -> transcript_text`
- media (`video`/`audio`): `yt-dlp` download → `ffmpeg -ar 16000 -ac 1` → WAV →
  `POST http://127.0.0.1:8402/transcribe` (multipart `file=@out.wav`).
- text (`article`/`thread`): passthrough (Corra already fetched/parsed).
- Depends on: `yt-dlp`, `/usr/bin/ffmpeg`, voice-stack `:8402` reachable.

### 2. Distill prompt (new variant in `api/app/llm/`)
- Distinct from the session-summary prompt. Extracts **atomic best-practices /
  insights, each with a source anchor** (timestamp for A/V, quote/paragraph for text).
- Returns structured lessons, not prose.

### 3. `/distill` endpoint (`api/app/routers/intake.py`, new)
```
POST /distill
  { source_url?: str, text?: str,
    media_type: "video"|"audio"|"article"|"thread", title?: str }
  →
  { provenance: { source_url, media_type, captured_at, title, duration? },
    transcript_ref,                    // stored so we can re-distill later
    lessons: [ { claim: str,
                 citation: str,        // "12:30" | quote | paragraph anchor
                 memory_type: "skill"|"decision"|"fact",
                 confidence: float } ] }
```
- Stateless; auth via existing API key dep. Does **not** write to hive.
- Lesson default `memory_type = "skill"` ("when X do Y" best-practice); distiller
  may override per item; AgentSSOT `classify()` still gets final say at ingest.

### 4. Corra side (branch `corra`, moni)
- New action: on a source, call `/distill`, then feed each lesson to `memory_ingest`
  carrying provenance, landing in the `corra` namespace → owner promotes to
  `claude-shared` via the existing gate. No new promotion machinery.

## Data flow / provenance

Every lesson carries `source_url + media_type + captured_at + citation` into
`/ingest`, so recall can cite *where* and *when* a lesson came from. Re-ingesting the
same video is idempotent via the existing dedup gate (≥0.985 collapse, near-dup →
review row), so bulk intake will not bloat the review queue.

## Error handling — design against "confirms dispatch, never arrival"

- STT down / ffmpeg fail / yt-dlp fail → `/distill` returns a **hard error**; Corra
  reports it in Telegram. No silent empty-lesson 200.
- Zero lessons extracted → explicit "distilled 0 lessons", not a fake success.
- Embedding backend down at ingest → already fails loud (503, `knowledge.py:162`).
- Long videos → `/distill` is synchronous for v1; if STT latency on the 30-min video
  is unacceptable, promote to a job-id/poll shape in v2 (out of scope now).

## Testing / acceptance

- **Unit:** distill prompt on a canned transcript → asserts N atomic *cited* lessons.
- **Adapter:** short known clip → ffmpeg → `:8402` → non-empty transcript.
- **Acceptance (headline):** the 30-min Anthropic memory video URL → Corra →
  `/distill` → lessons appear in Corra's `/knowledge` queue → `/promote` →
  recallable from `claude-shared` **with a citation back to a video timestamp.**

## Out of scope (v1)

- Watched-folder / hands-free auto-intake (later).
- PDF adapter (article/thread/A/V only for v1).
- Async job/poll shape for very long media (synchronous v1).
- Semantic interest scoring, voice input to Corra (her deferred gaps, unrelated).

## Flagged risk (deferred, not bundled)

**Classifier ReadTimeouts** (fleet-audit rot): `classify()` on ingest can time out
because the qwen classifier model isn't resident. Bulk video intake will hammer it.
*Not* in v1 core scope, but recorded here: mitigation is keeping the classifier model
warm on the idle 3090. File separately if intake surfaces it in the acceptance run.
