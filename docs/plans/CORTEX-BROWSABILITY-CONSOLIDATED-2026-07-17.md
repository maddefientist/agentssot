# Cortex Browsability — Consolidated Findings & Implementation Plan

**Date:** 2026-07-17 · **Status:** REVIEW COMPLETE — no application code changed · **Branch:** `main`
**This file supersedes and merges three source docs** (now folded in here; delete-safe):
- `PI-FINDINGS-2026-07-15-cortex-browsability-review.md` (Pi lead — live-DB review, GPT-5.6 Terra/Luna/Sol + Grok 4.5)
- `HANDOFF-2026-07-16-obsidian-eval-and-graph-layer.md` (Opus/Claude Code — audit + spec, fable-heavy + gpt-5.6-sol)
- `ENTITY-KI-GRAPH-LAYER-2026-07-16.md` (Opus/Claude Code — original graph-layer chain brief)

**Next agent: read THIS file. It is the single reconciled source.** The original brief's graph-first
approach is **superseded** by the phased plan in §4 (see §2 for why). Technical anchors from all three
are preserved in §5.

Provenance note: two independent agents reviewed the same question with different tool access. Where
they conflicted, **Pi's live production-DB measurements win over Opus's static-read inferences** — that
reconciliation is done below, not left to the reader.

---

## 1. Shared verdict (both agents agree)

- **Do NOT replace Hive with Claude/Obsidian.** PostgreSQL stays the source of truth.
- **Do NOT build a read-only Markdown/Obsidian projection now.** It bypasses namespace/RBAC and
  retains memories after decay/correction/deletion (no tombstone semantics), and captures the
  pattern's weakest attribute (passive viewing) while dropping its strongest (human authoring).
  Reconsider only if a hard offline/grep/git requirement is ever *demonstrated* — one-way,
  namespace-scoped, secret-aware, clearly non-canonical. **Deferred, not "confirmed dead"** (Pi's
  softening of Opus's original framing — accepted).
- **The real gap is a UI/rendering problem, not a memory-architecture problem.** `cortex-v2.html`
  is already a three.js graph browser; it renders the concept layer only.
- **CORRECTION (both agents initially wrong):** "synapse" (`synapse/routes.py`) is **live-session
  presence** (who's editing which file now), NOT the knowledge graph. Do not wire it as memory edges.
- **Nothing ships before the P0 security fix in §3.** No code changed this session; no commits.

## 2. Where the two reviews conflicted — and the reconciliation

| # | Opus brief (`ENTITY-KI-GRAPH-LAYER`) | Pi review (live DB) | RECONCILED DECISION |
|---|---|---|---|
| Product order | Entity **graph** layer first | Concept→evidence **drawer** first | **Concept-evidence drawer first** (Pi). It's the smallest high-value feature — concept payload already carries `evidence_ids` and the panel already shows the count. |
| Graph reuse | Wrap entities in concept-shaped objects, reuse `buildNodes`/`buildGlobalEdges` | Avoidable coupling in a 3,186-line monolith; use entity **page/detail** UI | **Entity detail page, not graph mode** (Pi). Revisit a local subgraph only if list/panel journeys prove insufficient. |
| Auth path | New `/cortex/*` endpoints to get reader-key behavior (entities router is writer/admin) | That's a workaround for a **bug**; fix the router's missing namespace auth + allow reader keys | **Fix the entities router** (Pi). Do not create duplicate `/cortex/*` entity APIs to dodge role behavior. |
| Data coverage | Assumes `entity_refs` sufficient | Only **2,756 / 7,914 KIs** have a resolved entity ref (~35%) | **Disclose coverage / consider backfill** before an entity graph — it would underrepresent the corpus. |
| Evidence provenance | Filter to `status='active'` | Concept evidence must NOT silently drop inactive/superseded provenance — return status + missing counts | **Split contracts:** entity-KI browsing may filter active; concept-evidence must surface status/orphans so the operator can audit *why a concept exists*. |
| Effort | "~1 day, low risk" | **3–5 days** tested (concept+entity), after RBAC fix | **3–5 days.** Opus's estimate was too optimistic. |
| Security gate | Not identified | **P0 cross-namespace entity leak** must be fixed first | **Hard gate.** See §3 P0. |

Net: the *strategic* conclusion (extend the dashboard, not Obsidian) holds; the *implementation
shape* is Pi's — security-gated, concept-evidence-first, entity-detail-second, global graph deferred.

## 3. Consolidated findings (verified against live prod, commit `140d289`)

### Verified current state
- **Graph renders concepts only, and not even all of them.** `/cortex/data` (`main.py:646-664`) →
  `crud.list_concepts(..., limit=1000)` hard-capped at 1,000 by confidence (`crud.py:1346-1365`).
  Prod has 1,385 concepts / 1,368 non-superseded → **≥368 active concepts are absent from the graph.**
  (Opus's "graph shows ~1,385 nodes" conflated corpus total with rendered nodes — corrected.)
- **KI browsing exists but is search-oriented.** Knowledge Browser tab (`cortex-v2.html:835-856`,
  impl `:2972-3185`) needs a query, fetches ≤50, client-slices. `/console` + `/query` also expose
  KIs but `/query` mixes entities/requirements/KIs/events, caps at 100, no cursor
  (`main.py:1116-1138`, `crud.py:248-347`). **No concept→evidence pivot anywhere.**
- **Entity list exists, entity drill does not.** `/entities` → `entities.html` is a flat searchable
  table (name/type/ref-count/IPs/CWD hints, `:19-50`); no row detail, no linked-KI/concept list, no
  server pagination.
- **Three distinct relations (do not conflate):** `Concept.evidence_ids` = UUID array of evidence KIs
  (`models.py:326-345`); `KnowledgeItem.entity_refs` = JSONB entity-UUID strings (`models.py:260-263`);
  `ConceptLink` = concept→concept only (`models.py:363-372`). Separate API contracts.

### Live corpus measurements (Pi, read-only aggregates, namespace `claude-shared`; counts grow with ingest)
| Metric | Count | Metric | Count |
|---|---:|---|---:|
| Knowledge items | 7,914 | Concepts | 1,385 |
| KIs w/ non-empty `entity_refs` | 2,955 | Concepts w/ evidence | 1,385 |
| KIs w/ ≥1 resolved entity ref | 2,756 | Concept-evidence refs | 56,292 |
| KIs w/ legacy `entity_id` | 25 | Distinct KIs used as evidence | 4,978 |
| KIs w/ legacy `project_id` | 6 | KIs in no concept | ~2,948 |
| Entities | 672 | Orphan evidence refs | 136 |
| Entities referenced ≥1× | 671 | Avg evidence refs/concept | 40.64 |
| Entity-ref values (resolved/unresolved) | 4,520 / 297 | 95p / max evidence refs | 202 / 446 |
| | | Concepts w/ >50 evidence refs | 313 |

**Implications:** (1) ~35% KI entity-ref coverage → entity graph underrepresents corpus unless
disclosed/backfilled. (2) Nearly every entity is referenced → entity detail is useful. (3) 313
concepts exceed 50 evidence items (max 446) → **concept evidence REQUIRES server pagination.** (4)
Resolution must report missing/orphan refs, not silently drop. (5) ~3,000 KIs belong to no concept →
concept browsing ≠ complete KI browser.

### Ranked findings
- **P0 — entity endpoint lacks namespace authorization.** `GET /api/v1/entities/`
  (`routers/entities.py:16-29`) accepts a caller-controlled namespace and checks only writer/admin
  role — **no `ensure_namespace_access`.** A key scoped to one namespace can read another's entity
  metadata/counts. **Fix + regression-test before any new browsing surface.** After the fix, pure
  entity reads should allow **reader** keys. Do NOT work around it with duplicate `/cortex/*` APIs.
- **P1 — concept evidence is the smallest high-value feature.** Payload already has `evidence_ids`;
  detail panel shows the count (`cortex-v2.html:1436-1464`) but never resolves/renders the KIs. The
  single-KI expand endpoint (`routers/knowledge.py:757-784`) is unfit for loading a concept's
  evidence one-at-a-time (N+1). Needs a bounded, paginated concept-evidence endpoint.
- **P1 — KIs/entities should NOT become global graph nodes yet.** `buildNodes`/`buildGlobalEdges`
  (`cortex-v2.html:1149-1223`, `:1560-1606`) assume concept scope/type/confidence/evidence sizing.
  Wrapping entities as fake concepts raises coupling. `InstancedMesh` cuts draw calls but doesn't
  solve legibility/interaction/accessibility/layout stability. Evidence → list in detail panel;
  entity → page/panel; local subgraph only if those prove insufficient.
- **P1 — explicit snapshot/pagination contracts.** API returns `total`/`rendered` but UI doesn't
  surface the distinction; concept links can reference nodes omitted by the 1,000 cap. Concept-evidence
  response shape (minimum):
  ```json
  { "concept_id": "uuid", "total": 123, "items": [], "missing_ids": [], "next_cursor": "opaque-or-null" }
  ```
  Every concept/KI lookup must enforce the authorized namespace **in the DB query**, not by trusting
  the caller's requested namespace after a UUID fetch.
- **P2 — node sizing bug.** Frontend reads `c.evidence_count` (`cortex-v2.html:1185-1192`, `:1241-1245`)
  but the serializer returns `evidence_ids`, not `evidence_count` (`crud.py:1367-1387`) → "Size =
  evidence" legend is inert; nodes fall to min size. Fix: use `(c.evidence_ids||[]).length` or add a
  tested serialized `evidence_count`.
- **P2 — fragmented auth inside Cortex.** Shared shell stores a canonical key (`cortex-shell.js`) but
  several tabs (incl. Knowledge Browser) read a separate `#admin-api-key` (`cortex-v2.html:2559-2609`,
  `:2983-2984`) → silent empty states for a valid reader key from the header pill. New browsing must
  use `window.cortexFetch`; migrate existing read-only tabs to it.
- **P2 — stale tests give false safety.** `api/tests/test_cortex.py:172-198` calls `/cortex/links`
  without headers though prod requires `require_api_key` (`main.py:668-678`). No coverage of entity
  namespace isolation, reader access, evidence pagination, orphan handling, cross-namespace link
  targets, or the 1,000-node cap.
- **P2 — CDN deps broader than three.js.** three.js+addons from jsDelivr (`cortex-v2.html:7-11`);
  entity page pulls Alpine from unpkg (`entities.html:9`). Blocks reliable offline use; the browser
  stores the API key in `localStorage` (`cortex-shell.js:15-37`) → supply-chain exposure. Vendor if
  offline/LAN-only or stronger key isolation is needed.

## 4. Reconciled implementation plan (security-gated, phased)

### Phase 0 — security & correctness (BLOCKING; do first)
1. Add `ensure_namespace_access` to entity reads; permit authorized **reader** keys.
2. Verify **both** endpoints of each concept link belong to the authorized namespace.
3. RBAC regression tests: reader/writer/admin × authorized/unauthorized namespaces.
4. Fix evidence-derived node sizing (P2 above).
5. Surface rendered-concepts vs corpus-total in the UI.
6. Migrate existing read-only Cortex tabs to `window.cortexFetch`.
7. Fix stale Cortex tests so authenticated routes are actually exercised.

### Phase 1 — concept → evidence drawer (smallest high-value)
Reader-authorized, namespace-scoped endpoint, e.g.:
```
GET /api/v1/concepts/{concept_id}/evidence?namespace=claude-shared&limit=50&cursor=...
```
Behavior: resolve concept in authorized namespace; preserve `evidence_ids` order (or document a
deterministic alternate); return KI summary (source/provenance, memory_type, tags, status, staleness,
entity_refs, timestamps); full content only on explicit expand; **report missing/orphan IDs**;
deterministic pagination; **do NOT silently exclude superseded/inactive evidence — return its status**
so the operator can audit why a concept exists. Wire into the existing concept detail panel; make
"N evidence items" actionable with loading/empty/partial/forbidden/retry/orphan states.

### Phase 2 — entity detail & linked KIs (after Phase 0/1)
Extend the existing `/entities` page (NOT a global graph mode). Reader-authorized, namespace-scoped:
```
GET /api/v1/entities/{entity_id}/knowledge?namespace=claude-shared&status=active&limit=50&cursor=...
```
Behavior: match canonical `entity_refs`; optionally include legacy `entity_id`/`project_id` with a
relation-source label; de-dup KIs reached via multiple paths; paginate/filter explicitly; show entity
metadata + linked KIs + linked concepts; **disclose the ~35% entity-ref coverage limitation.** Make
entity rows clickable; visible loading/error/forbidden states; never render an empty table on auth
failure.

### Phase 3 — usability hardening
Accessible list/search for concepts (incl. those omitted by the 1,000 cap); keyboard nav; responsive
drawers; reduced-motion; non-WebGL fallback; stabilize graph layout across refreshes. Revisit local
subgraph expansion only if operator testing shows lists/panels insufficient.

## 5. Technical reference (anchors preserved from all three source docs)

**Auth/wiring facts.** Graph browser fetches via `window.cortexFetch` → `X-API-Key` cortex read key
(`cortex-v2.html:2580`). Existing concept-edge endpoint (mirror-of-record): `main.py:668`
`@app.get("/cortex/links")` → `crud.list_concept_links(session, ns, limit, min_weight)`. Router mounts
`main.py:254-264` (entities at `/api/v1/entities`, knowledge at `/api/v1`). **Do NOT** add entity APIs
under `/cortex/*` to dodge role behavior — fix the entities router (Phase 0.1).

**Frontend seams** (`cortex-v2.html`): `buildNodes(concepts)` @1149, `updateNodes` @1226,
`loadLinks`/`buildGlobalEdges` @1549/1560, `showFocusEdges` @~1618, `showDetailPanel` @1436,
`focusOnNode` @1411, drawer tabs @610, Knowledge Browser @835 / impl @2972-3185, evidence count in
panel @1436-1464, node sizing @1185-1192 & @1241-1245.

**Models** (`models.py`): `Concept.evidence_ids` UUID[] @326-345; `KnowledgeItem` @192 (`id`,
`namespace`, `content`, `entity_id`/`project_id` FK entities nullable, `entity_refs` JSONB @260-263,
`memory_type`, `status`, `summary`/`abstract`); `Entity` @147 (`id`,`slug`,`type`(EntityType:
project/person/agent/document/integration/other),`name`,`description`,`meta`(JSONB col `metadata`));
`ConceptLink` @363. Reusable jsonb pattern proven in `routers/entities.py list_entities`:
`func.jsonb_array_elements_text(entity_refs)`, `func.jsonb_exists_any(entity_refs, cast(ids, ARRAY(TEXT)))`.

**Superseded from the original brief:** its `/cortex/entity-graph` + `/cortex/entity-items` endpoints
and the `Concepts|Entities` three.js toggle. Replaced by Phase-1/2 endpoints on the proper routers and
entity **detail-page** UI. The brief's negative-space constraints still hold (read-only, no migration,
no write paths, no embeddings/>200-char content to browser, status handling per §2 provenance split).

## 6. Explicit deferrals
- **Global entity/KI graph layer** — defer until concept-evidence + entity-detail journeys are used;
  data coverage (~35%) and graph semantics don't justify first-class global entity/KI nodes yet.
- **Markdown/Obsidian projection** — do not build now (see §1). One-way, namespace-scoped,
  secret-aware, non-canonical *if* ever demonstrated necessary.

## 7. Go / No-Go
- **NO-GO:** dispatch the original `ENTITY-KI-GRAPH-LAYER` brief unchanged (graph-first, `/cortex/*`
  workaround, ~1-day estimate).
- **GO:** Phase 0 (security/correctness) → Phase 1 (concept-evidence drilldown).
- **GO later:** Phase 2 entity detail once endpoint semantics + data coverage are explicit.
- **NO-GO for now:** global KI nodes, full entity co-reference graph, Obsidian projection.

## 8. Pickup checklist (next agent, in order)
1. This document (single reconciled source).
2. `api/app/routers/entities.py` — fix P0 first.
3. `api/app/main.py:646-678` and concept routes ~`:1599-1629`.
4. `api/app/crud.py:1346-1518`.
5. `api/app/ui/cortex-v2.html:1149-1606` and `:2972-3185`.
6. `api/app/ui/entities.html`; `api/app/ui/cortex-shell.js`.
7. `api/tests/test_cortex.py` (stale — see P2).
8. hive: recall "obsidian integration" / "cortex graph browser gap" (3 KIs logged 2026-07-16).

Then implement Phase 0 → 1 → 2. Verify acceptance yourself (curls, RBAC tests) — worker output is
summary, not proof. Gate: browsing need is asserted, not yet evidenced; Phase 0 (the security fix) is
worth doing regardless.

---
*Consolidated by Opus 4.8 (Claude Code) on 2026-07-17 from the Pi review (2026-07-15) and the Opus
audit+brief (2026-07-16). Two independent agents, cross-vendor auditors on both sides.*
