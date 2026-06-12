# Scope Sheet — Cortex Cohesion Pass

**Date:** 2026-06-08
**Author:** Madi (Opus, on-box at `hari:/opt/agentssot`)
**Trigger:** Operator visited the running Cortex UI; "not intuitive, can't tell if it works, not cohesive."
**Companion to:** `AUDIT-2026-06-04-cortex-learning-loops.md`, `PLAN-2026-06-04-cortex-control-panel-and-learning-loops.md`

> **Purpose of this file:** declare the intended final state of this work so any auditing
> agent (or a resumed session) knows what is in-flight, what "done" looks like, and what is
> deliberately out of scope — even if I am interrupted mid-change. Update the STATUS column
> as each item lands. Delete this file once all items are ✅ and merged to master.

---

## 0. Pre-work already done (this session, before this sheet)

| # | Change | Status |
|---|---|---|
| 0a | **Killed runaway drain loop** (host PID 3549561, ran 2d11h) hammering `/admin/review-queue/reclassify`, saturating the sync threadpool → every page timed out. This was the entire "can't tell if it works" symptom. Server recovered: `/health` 14s→1.5ms, all pages 200. | ✅ done (process kill, no code) |
| 0b | Consolidated 3 overlapping keepUp digests → canonical `~/knowledge-base/agentssot.md`; deleted stale dup; repointed README index. | ✅ done, pushed |
| 0c | Taught hive the runaway-loop diagnosis (rule id 00983eda). | ✅ done |

---

## 1. HUD as the front door

**Problem:** `/` serves the legacy Cortex admin (`index.html`); the cohesive Obsidian Terminal HUD is a separate page at `/hud`, buried in the nav. Visitors land on the old admin and never see the hero surface. This is the #1 "not cohesive" cause.

**Intended final state:**
- `GET /` serves the **HUD** (full-bleed, cache-busted) — same body `/hud` serves today.
- `GET /hud` still works (alias / kept for muscle memory).
- The legacy admin index moves to `GET /classic` (rendered with cortex nav, `active="home"`).
- The cortex nav (`_nav.html`) gains a link to `/classic` so the admin surface is still reachable; HUD nav link points at `/`.
- No data/endpoint changes — pure routing + one nav link.

**Files touched:** `api/app/main.py` (routes `/`, `/hud`, add `/classic`; extract a `_render_hud()` helper used by both `/` and `/hud`), `api/app/ui/_nav.html` (add Classic link).

**Done when:** `curl -s localhost:8088/ | grep -c 'hud.js'` ≥ 1; `/classic` returns the old admin (200, contains nav); `/hud` still 200. Visual: landing is the Obsidian Terminal.

**STATUS:** ✅ done

---

## 2. Reclassify convergence guard (the bug that enabled the 3-day loop)

**Problem:** `services/review_queue.py::reclassify_low_conf` — when the classifier returns low confidence, the row stays `pending` and **nothing is stamped**. Items that can't be auto-typed (here: 5) re-enter the pool on every pass, so any drain loop runs forever. Root cause of pre-work item 0a.

**Intended final state:**
- `review_queue` gains an `attempts INTEGER NOT NULL DEFAULT 0` column (migration `db/init/008_review_attempts.sql`, additive).
- New setting `RECLASSIFY_MAX_ATTEMPTS` (default 3) in `settings.py`.
- `reclassify_low_conf` per row: increment `attempts`; if confident → type + resolve (existing); **elif `attempts >= max` → dismiss as terminal** with reason `"unclassifiable after N attempts"`; else leave pending. Always `commit` the attempts bump even when not dry_run-typed.
- Return dict gains `dismissed_unclassifiable`.
- A drainer now provably converges: every low_conf row is either typed, or dismissed after ≤ max attempts. No infinite loop possible.

**Files touched:** `db/init/008_review_attempts.sql` (new), `api/app/models.py` (`ReviewQueueItem.attempts`), `api/app/settings.py` (`reclassify_max_attempts`), `api/app/services/review_queue.py` (`reclassify_low_conf`).

**Done when:** running reclassify repeatedly drives `low_conf` count to 0 (typed or dismissed) and then `scanned=0`; no row survives > `RECLASSIFY_MAX_ATTEMPTS` passes. Verified live against the current 5 stuck items.

**Out of scope:** `reclassify_untyped` already converges (stamps `last_classified_at`, attempted once) — left as-is.

**STATUS:** ✅ done

---

## 3. Malformed entity_refs

**Problem:** Callers send entity **names** (`'unraid'`, `'jellyfin'`) in `entity_refs`; ingest stores them verbatim though the schema says "Entity UUIDs (as strings)". On every recall, `_safe_uuids` tries `UUID()`, fails, and logs a `warning` per bad ref — continuous log spam and a real contract drift.

**Correction found during implementation:** the dominant polluter is NOT the caller — it's
the **classifier**. `classify()` returns `entity_mentions` as human *names*
(`['unraid','jellyfin','qbittorrent']`), which were written straight into `entity_refs`.
The original repeating warnings were classifier-sourced. Fix therefore resolves BOTH caller
and classifier refs through one resolver.

**Intended final state:**
- **Ingest (write path):** caller-supplied AND classifier-extracted `entity_refs`/`entity_mentions`
  are run through `_resolve_entity_refs` — matched by `(namespace, slug)` then case-insensitive
  `name`. Resolved → store the entity UUID. Unresolved → dropped from `entity_refs` (never stored
  as a fake ref) and preserved as `entity:<name>` tags. `entity_refs` is now UUID-only; the
  supersession scan (`new_entity_refs`) uses the resolved UUIDs.
- **Read path:** `_safe_uuids` downgrades the per-item miss from `warning` to `debug` (legacy
  rows still exist; no need to spam a hot path).
- Net: new writes never pollute `entity_refs` with non-UUIDs; existing rows stop spamming logs;
  dropped names stay searchable via tags.

**Files touched:** `api/app/routers/knowledge.py` (`ingest_tiered` entity-ref resolution; `_safe_uuids` log level), small helper `_resolve_entity_refs(session, namespace, names)`.

**Done when:** ingesting an item with `entity_refs:["unraid"]` against a namespace where an `unraid` entity exists stores that entity's UUID; against one where it doesn't, stores no ref and adds tag `entity:unraid`; recall no longer emits `warning`-level `skipping malformed entity_ref`.

**Out of scope (offered, not done unless asked):** a corpus cleanup script to rewrite/strip the ~existing legacy non-UUID refs. New writes are clean; legacy rows are harmless once the warning is downgraded.

**STATUS:** ✅ done

---

## 4. Wire the dead ambient panels

**Problem:** `gateway/wiring.py` passes `fleet=None, chains=None` to `snapshot_status`; the HUD's FLEET slot shows "—" and there's no live-activity panel. SSE confirmed `fleet:null, chains:null`.

**External reality (constrains this item):**
- **fleet-dashboard (:9105) is currently DOWN** — `curl localhost:9105` returns 000 even from the host. The container *can* reach `host.docker.internal` (it reaches Ollama), so a fleet feeder will auto-light when that service runs, but it returns null right now.
- **chains** live in `.chain/` on the **host filesystem**, which is **not mounted** into the api container. The container cannot enumerate chain runs. Wiring this properly needs a volume mount or a host-side chain-run API — deferred.
- The container **can** see Postgres, so `synapse_session` (live agent activity: host/cwd/repo/file) is real, verifiable data available now.

**Intended final state:**
- `snapshot_status` extended with a `synapse` slot.
- New `_fleet_status` source: GET `FLEET_DASHBOARD_URL` (setting, default `http://host.docker.internal:9105/api/state`), 2s timeout, return compact summary or None. Wired into `fleet=`. Auto-populates when :9105 is up.
- New `_synapse_activity` source: count `synapse_session` rows seen in the last 10 min + latest cwd/host. Wired into the new `synapse=` slot. **Real data now.**
- `hud.js`: render `snap.fleet` into `#fleet` (host count or "—") and `snap.synapse` into `#synapse-row` (e.g. "3 active · /opt/agentssot") + flip `#dot-synapse` on only when activity > 0.
- `chains` slot stays `None` with an inline comment documenting the mount/API requirement.

**Files touched:** `api/app/gateway/feeders.py` (`synapse` param), `api/app/gateway/wiring.py` (`_fleet_status`, `_synapse_activity`, pass them), `api/app/settings.py` (`fleet_dashboard_url`), `api/app/ui/hud.js` (render fleet + synapse).

**Done when:** SSE `/gateway/sse/status` shows a non-null `synapse` slot reflecting live sessions; HUD `#synapse-row` shows live activity. `fleet` remains null **only because :9105 is down** — verified the feeder returns the dashboard payload when pointed at a reachable URL (documented; not asserting fleet is populated).

**STATUS:** ✅ done (synapse live; fleet wired, null until :9105 runs; chains deferred-documented)

---

## Deployment / verification

- Migrations are raw SQL applied by `db/init` on container init **and** idempotently re-applied; for a running DB, 008 is applied via `docker exec agentssot-db psql` (additive `ADD COLUMN IF NOT EXISTS`).
- Rebuild: `docker compose up -d --build api`.
- Smoke after each item per its "Done when".
- Final: `./scripts/postdeploy-isolation-check.sh` (namespace isolation gate must stay green).

## Verification results (2026-06-08)

| Item | Evidence |
|------|----------|
| #1 HUD front door | `/` serves HUD (`hud.js` present), `/hud` 200, `/classic` renders admin nav. |
| #2 Reclassify guard | Live drain: `scanned=5 → typed_and_resolved=5, low_conf→0` in one pass; response now reports `max_attempts:3` + `dismissed_unclassifiable`. Terminal-dismiss path code-verified. |
| #3 entity_refs | Ingest with `entity_refs:["unraid","madeup-qxz"]` + content mentioning jellyfin/qbittorrent → stored `entity_refs` = 3 entity UUIDs; only `entity:madeup-qxz` became a tag. Recall no longer logs `warning`. |
| #4 panels | SSE `/gateway/sse/status` now emits a `synapse` slot with live DB data (`active`,`cwd`,`host`); `fleet` wired (null only because :9105 is down); HUD renders both. |
| Tests | `tests/gateway` 66 passed. Pre-existing/stale failures in `tests/test_typed_memory.py` (enum expanded 2026-04-24, test not updated) are unrelated to this work. |
| Isolation gate | 100% — 6 denied, 0 leaks. Also fixed the gate's `python`→`python3` bug (`scripts/postdeploy-isolation-check.sh`). |
| Migration | `008_review_attempts.sql` applied live (`review_queue.attempts` int default 0). |

## Out of scope for this pass
- Generating the proactive briefing artifact (deferred thread).
- Voice/Telegram channel adapters.
- Repairing the synthesis clustering (separate PLAN-2026-06-04 effort).
- Legacy entity_ref corpus rewrite (offered).
- Retiring the legacy admin entirely (kept at `/classic`).
