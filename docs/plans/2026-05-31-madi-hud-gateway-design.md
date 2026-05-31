# Madi HUD + Gateway — Design

**Date:** 2026-05-31
**Status:** Design (approved in brainstorm; pending spec review)
**Repo:** `/opt/agentssot` (hari)
**Surface:** new hero surface in the existing Cortex FastAPI app (`api/app`)

---

## 1. Context & vision

We are assembling the experiential layer on top of the system already built (hive/agentSSOT memory, the 13-host fleet, chain model-routing, voice infra). The reference is larossa_tech's "Building Jarvis" reel — but the takeaway from decoding it was that **we have already built the hard 80% (memory, sprawl, compute, routing); what is missing is the face, the proactive life-layer, and ambient embodiment.**

The whole system reads as one body:

| Organ | System | Status |
|---|---|---|
| Mind / memory | hive: short-term (working) + long-term (knowledge/concepts) + **synapse** (associative) + cross-platform | built / upgrading |
| Nervous system | **`madi-gateway`** — routing, session state, reflexes | **this project** |
| Brain regions | executor ladder (Opus → deepseek-v4-pro → flash → local) | exists (chains) |
| Bodies & senses | **HUD** (sight) · Telegram (text) · voice/glasses (speech) | **HUD this project**; others later |
| Limbs / reach | fleet + hari-core dispatch | exists |
| Autonomic | proactive scheduled agents (the briefing) | deferred thread |

**This spec covers two organs only: the nervous system (`madi-gateway`) and the first body (the HUD).** The proactive briefing, voice adapter, and Telegram adapter are explicitly deferred — but the interfaces here are built to accept them without rework.

## 2. Goals & non-goals

**Goals**
1. A standing, always-on Madi presence with an interactive command center (HUD) as its first body.
2. Reliability that does **not** depend on any single model or on the flaky Hairy runtime — strength in architecture, models swappable.
3. Opus-class orchestration when needed, with a transparent deepseek fallback ladder.
4. One continuous memory across channels (HUD now, Telegram/voice later) via hive-backed session state.
5. Zero new orphans: every overlapping surface gets an explicit absorb/keep/retire verdict, enforced.

**Non-goals (this iteration)**
- Proactive/scheduled briefing generation (deferred thread; HUD renders a briefing artifact when one exists, but does not generate it here).
- Voice and Telegram channel adapters (interface-ready, not built).
- Replacing the existing Cortex admin UI (it stays; HUD is a new sibling surface / new landing).

## 3. Architecture

Five independently-testable units. The gateway is a **new module inside the existing FastAPI app** (`api/app/gateway/`), reusing its process, auth, and hive connection — not a parallel service.

### 3.1 `madi-gateway` (the nervous system)
Deterministic plumbing, **no model baked in**. Responsibilities:
- Accept channel connections (WebSocket for HUD; same protocol for future channels).
- Maintain conversational/session state **in hive**, not in process memory — a restart loses nothing.
- Run the **hybrid intent router** (§3.2).
- Dispatch to an executor (§3.3) and stream results back.
- Apply the fallback ladder and surface typed errors to the channel (never crash a session).

### 3.2 Hybrid intent router (front door)
1. **Deterministic rules first** — known patterns (slash-commands, `recall …`, `scan fleet`, `build …`, `briefing`) match a rules table → executor. Instant, no model.
2. **Cheap local model fallback** — freeform natural language goes to a small local Ollama model that returns an executor + structured args only (classification, not generation).
Front door stays fast and model-light; classification never blocks on a cloud model.

### 3.3 Executors (swappable brain regions)
Single stable interface: `execute(intent, ctx) -> AsyncStream[Event]`. Adding/swapping never touches gateway or HUD.

| Executor | Backend | Use |
|---|---|---|
| `chat-local` | local Ollama (qwen3.5) | casual chat — cheap, instant |
| `hive-tool` | direct hive calls | recall/teach/status — **deterministic, no LLM** |
| `orchestrate` | **Opus → deepseek-v4-pro → flash → local** | real reasoning / tool-use / orchestration |
| `dispatch` | hari-core / `chain.sh` | builds, scans, chains, fleet jobs |

Fallback ladder is **gateway config**, not code. Failures are logged visibly (mirroring `fallovers.log`). Hairy, if ever wired, is one optional executor wrapped in timeout+fallback — never the critical path.

### 3.4 HUD frontend ("Obsidian Terminal")
Single-page hero surface in `api/app/ui/` following existing `_nav.html` + cache-bust shared-asset patterns.
- **Command bar** → WebSocket into the gateway; streamed responses render live.
- **Status panels** → SSE (§3.5).
- **Two modes, one surface (morphing):**
  - *Ambient* — Madi presence ring + today's briefing (if present) as hero card + memory/synapse feed + agent/executor health. Glanceable.
  - *Active* — on engage, centre becomes a working console (streamed conversation/output, live model shown); briefing docks aside.

### 3.5 Status feeders
SSE endpoints in the FastAPI app aggregating: hive activity (recalls/teaches/feedback, synapse links), executors-online + health (which model is live; **you see when Opus is down and it dropped to deepseek**), fleet health (reuse fleet-dashboard data on :9105 — do not rebuild), active chain runs, active projects.

## 4. Visual identity — "Obsidian Terminal"
One near-black palette, one warm brass/amber accent, two type registers.
- **Palette:** base `#0a0a0c`, panel `#101013`, line `#26241f`, accent (brass/amber) `#d9a441`, accent-dim `#8a6e3a`, ink `#ece7dd`, dim `#7d776c`.
- **Type:** monospace (`ui-monospace`) for the machine layer — chrome, status, data, command line; serif (Georgia) for the reading layer — the briefing/editorial content.
- **Texture/motion:** subtle phosphor scanline overlay on stage zones; minimal brass breathing ring for Madi presence; restrained glow. Operator-honest + premium. Explicitly **not** Stark cyan.

## 5. Data flow
- **Command:** HUD → WS → gateway → hybrid router → executor → token/event stream → HUD renders live.
- **Shared memory:** gateway reads/writes conversation context to hive → a thread started elsewhere (future Telegram/voice) continues on the HUD. One continuous Madi.
- **Status:** feeders subscribe to sources → SSE → panels update live.

## 6. Consolidation Ledger (no orphans)
**Rule (enforced in plan's Definition of Done):** a component is "absorbed" only when (1) the new path works, (2) old files are deleted, (3) all references repoint. No "clean up later."

| Surface | Verdict | Note |
|---|---|---|
| harivoice | **ABSORB** | becomes the voice channel adapter (later); delete old files once voice works |
| OmniVoice / HariScribe | **KEEP** | likely the STT/TTS engine behind the voice adapter — audit confirms |
| air Telegram bot | **ABSORB** | repoint at gateway as Telegram adapter (later); delete standalone brain |
| Hariha / madi-core / Pi | **KEEP** | `orchestrate` executor backend — audit confirms boundary |
| fleet-dashboard (:9105) | **KEEP** | data source for status panels — do not rebuild |
| flightdeck | **RETIRE if HUD supersedes** | audit confirms not load-bearing elsewhere |
| Hairy | **KEEP (optional)** | optional executor only, off critical path |
| repo root: `firebase-debug.log`, `after/`, `dropbox/` | **AUDIT → likely RETIRE** | orphan artifacts to verify + remove |

**Plan step 1 is a live audit** to confirm file lists and verdicts before any deletion.

## 7. Endpoints / data model (to detail in plan)
- `WS /gateway/ws` — channel protocol: `{type, intent?, text, session_id}` in; `{type: token|event|error|done, …}` out.
- `GET /gateway/sse/status` — multiplexed status stream (or per-panel SSE).
- Session state stored in hive under a `madi-session` namespace/key scheme (TBD in plan).
- HUD served at a new route (candidate: `/hud`, possibly the new default landing).

## 8. Reliability & error handling
- Executors return structured results or **typed errors**; gateway catches → surfaces in HUD with retry/alternate-executor option; session survives.
- `orchestrate` applies fallback ladder on error/unavailability; every fallover logged.
- Each executor has a liveness probe rendered in the HUD health panel.
- Gateway is stateless-in-process (state in hive) → crash/restart is non-destructive.

## 9. Testing
- Intent router: unit tests on the rules table + mocked local-model fallback (assert correct executor + args).
- Executor contract tests with fake executors (interface conformance, streaming, typed errors).
- Fallback ladder: simulate Opus failure → assert deepseek-v4-pro used, fallover logged.
- Status feeders: SSE integration test (event shape, NULL-graceful per existing display rule).
- HUD: WS round-trip + mode-morph render test.

## 10. Out of scope (this iteration)
Proactive briefing generation; voice + Telegram adapters; multi-user/auth hardening beyond existing app; mobile/glasses clients.

## 11. Open questions for the plan
1. Gateway as a module in `api/app` (shared process) vs a sibling service — leaning shared module; confirm against current app structure.
2. HUD route: new `/hud` vs becoming the default landing over the current Cortex index.
3. Exact local model for chat + intent classification (qwen3.5:4b vs other).
4. Session namespace/key scheme in hive.

## 12. Definition of done (this iteration)
- Gateway module live in `api/app`; WS + SSE endpoints working.
- HUD ambient + active modes functioning against real hive/fleet/chain data.
- `orchestrate` fallback ladder verified (Opus→deepseek) with visible logging.
- All tests green.
- Consolidation Ledger audit completed; any RETIRE artifacts deleted; ABSORB items either done or explicitly scheduled with old paths still intact (not yet deleted) and tracked.
