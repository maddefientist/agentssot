# Recall / Ingest Remediation Plan — 2026-07-09

**Author:** Opus (synthesis) from independent GPT-5.5 (`gpt-orchestrate`) + Grok-4.5 (`pi`) plans.
**Status:** DRAFT — awaiting operator go before chain dispatch.
**Predecessor:** logit reranker scoring is LIVE in prod (`reranker_scoring_mode=logit`), A/B-gated. Done.

## Key architectural decision (grok-4.5, verified)

The brief over-gated coverage recovery behind latency infra. **It shouldn't be.**
Per-candidate rerank is prefill-bound (~20 docs/s global) — parallelizing tiers does *not*
multiply throughput. But it *does* kill inter-tier serial dead-time, and combined with **low
per-tier caps** on the new content tiers, that is enough to re-land the tier widening **without**
the 4B swap or TEI/vLLM spike. 4B and TEI become optional *latency polish*, not coverage
prerequisites.

**Spine:** parallelize per-tier recall → widen tiers *with caps* → measure → 4B/TEI only if budget still misses.

## Live ground-truth (probed today)

- `reranker_scoring_mode=logit` ✓, `ollama_reranker_model=8B:Q8_0`, `RERANKER_CANDIDATE_MULTIPLIER=2`.
- Type distribution (n=9345): **NULL 2289 (24.5%)**, episodic 1922, skill 1813, fact 1007, doctrine 802,
  rule 627, decision 296, entity 236, session_summary 200, correction 68, command 39, preference 28, reference 18.
  NULL is *growing* (was ~2069) → confirms untyped writes accumulate → item C is urgent.
- Synthesis loop is **ALIVE**: 1519 concepts, 86 in last 7d, last run today 03:29 (matches `synthesis_schedule_hour=3`),
  concept_feedback flowing. → item E is *verify/instrument only*, not a rebuild. Note: synthesis builds a
  `concepts` graph, NOT episodic→knowledge_items promotions (a nuance for later).
- Deferred classify machinery exists (`review_queue.py:229,304`, `reclassify_max_attempts`, migration 008) →
  C and D reuse it rather than build new.

## Critical shared hazard (both planners)

`asyncio.gather` over tiers sharing one SQLAlchemy `Session` is **not safe**. Mitigation options for WU1:
one session/connection per tier task, OR serialize the DB candidate fetch and parallelize only the
reranker HTTP calls. Global rerank semaphore stays (≤8) so 9 tiers don't open 72 workers.

---

## Work units

### WU1 — Parallelize per-tier bucketed recall  [item A, unblocks B]  **DO FIRST**
- **Goal:** Replace the sequential `for tier in tiers:` (`knowledge.py` ~633) with bounded `asyncio.gather`,
  so tier wall-clock ≈ max(tier) + shared embed, not sum(tiers).
- **Deps:** none.
- **Bricks:** `gemma-scout` (inventory session/rerank call sites) → `deepseek-plan` (file-specific plan) →
  `glm-implement` → `kimi-review` (session-safety). Escalate to `gpt-review` if the session-concurrency
  design is contested.
- **Negative space:** per-tier session isolation (see hazard); keep global rerank semaphore ≤8; preserve
  `rerank_ms`/`candidates_per_tier` diagnostics, per-tier RRF, output sanitization, and per-tier
  vector-score fallback on reranker error.
- **Accept:** warm 5-tier `hive_recall` p50 ≤ current baseline; 9-tier synthetic ≤ ~1.3× 5-tier (not ~1.8×);
  unit tests for gather + session isolation + fallback; result-ID parity vs serial (tie-order tolerant).

### WU2 — Re-land tier widening **with caps**  [item B]  — gate: WU1
- **Goal:** Recover dark ~60% coverage. Re-apply the `10c61450` surfaces (`mcp_server.py` `top_per_tier`;
  `schemas.py` `DEFAULT_RECALL_TIERS`/`DEFAULT_TOP_PER_TIER`/`MemoryTierLiteral`) but with **low caps**:
  `fact:2, doctrine:2, correction:1, preference:1`.
- **Deps:** WU1 only. *Not* A1 (4B), *not* A2 (TEI).
- **Bricks:** `glm-implement` (known diff + caps) → `kimi-review`.
- **Negative space:** no uncapped 9-tier defaults; do NOT add episodic/session_summary to the default MCP set;
  NULL-typed stays unreachable until WU4 (expected). Latency guard: warm p50 ≤ 9s or tighten caps further.
- **Accept:** MCP recall returns non-empty fact/doctrine buckets on seeded queries; warm p50 ≤ 9s; overlap@k
  smoke shows no regression on the original 5 tiers' top results.

### WU3 — Ingest classify off the hot path  [item C]  **DO FIRST (parallel with WU1)**
- **Goal:** Stop awaiting `gemma4:31b-cloud` classify on untyped ingest (`knowledge.py:176`); return fast,
  resolve via the existing `review_queue` drainer.
- **Deps:** none.
- **Bricks:** `deepseek-plan` (write-path contract) → `glm-implement` → **`gpt-review` (escalation:
  correctness-critical contract)** → `kimi-review` optional.
- **Contract (no waffling):** explicit `memory_type` → unchanged sync path. Untyped → immediate type = **NULL**
  (no fake provisional type), enqueue/leave for the drainer, `last_classified_at` unset until it runs.
  **Secret-scan + semantic-dedup (0.92) stay on the hot path.** Layers (`abstract`/`summary`) via cheap local
  `compute_layers` without classifier; classifier enriches later. Do NOT change the response API shape unless the
  model already allows added metadata.
- **Accept:** untyped ingest returns without a classify call (unit test, target <500ms excl. embed); item readable
  immediately; drainer assigns type within one cycle; explicit-type path + secrets + dedup tests green;
  inserted untyped row has `memory_type IS NULL` and `last_classified_at IS NULL`.

### WU4 — NULL backfill sweep  [item D]  — gate: WU3
- **Goal:** Drive ~2289 NULL rows through the existing reclassify machinery in resumable, throttled batches.
- **Deps:** WU3 (shared classify contract/load).
- **Bricks:** `minimax-bulk` (batch/rate plan + reporting) → `glm-quick` (thin admin/cron wrapper if missing).
- **Negative space:** confirm the *actual* reclassify endpoint name (don't assume); batch ≤50, sleep between
  batches, dry-run first; idempotent via `last_classified_at`; local/cheap classifier only if it clears a
  quality gate on n≈50 sample vs gemma4:31b-cloud; no unreviewed type flips on high-value namespaces.
- **Accept:** dry-run pool report; live run drops NULL count ≥80% of classifiable set; second run scans ~0 new.

### WU5 — Synthesis liveness proof  [item E]  — parallel, low-priority
- **Goal:** Already confirmed alive (1519 concepts); produce written evidence + light instrumentation.
- **Bricks:** `gemma-scout` → `minimax-bulk` (query/digest evidence). No rebuild. Manual trigger only non-prod.
- **Accept:** evidence doc: task running, last-run ts, N concepts last 7d — or a filed defect if found dead.

### WU6 — Larger-n logit confirm (n≈200)  [item F]  — parallel, non-blocking
- **Goal:** Run existing `benchmarks/longmemeval/` at ~200 Q; freeze logit permanently.
- **Deps:** after WU2 preferred (measure real prod tier set), not blocking.
- **Bricks:** `minimax-bulk` (run + analyze). Do NOT flip scoring back; do NOT block B/C on this.
- **Accept:** recall@k table (logit ≥ generate) committed under `benchmarks/longmemeval/results/`.

### WU7 — 4B fast-model gate  [item A1]  — OPTIONAL, operator-auth-gated
- **Trigger:** only if post-WU2 warm p50 still >8s. A/B 4B vs 8B on ≥100 Q via hot-key only; explicit operator
  auth before any prod flip; rollback = hot-key back to 8B. Verify live config actually applies (stale-settings hazard).
- **Bricks:** `glm-quick` (runbook/hot-key) → `minimax-bulk` (score).
- **Accept:** recall@k delta within tolerance AND ≥1.5× latency win, else no flip.

### WU8 — TEI/vLLM prefix-cache spike  [item A2]  — LATER, not critical path
- **Goal:** Scoped spike doc (+ optional sidecar): TEI or vLLM reranker with shared-prefix cache; shadow A/B vs
  Ollama; rollback plan. Watch GPU-RAM contention with synth 27b.
- **Bricks:** `deepseek-plan` → human/infra; `gpt-review` if adopting.
- **Accept:** spike report with p50/p95 + quality parity + rollback → go/no-go, not a forced migration.

### WU9 — Retire legacy flat path (C2)  [item G]  — after WU2
- **Goal:** Inventory callers (`main.py` ~1100/1142/1149, `gateway/wiring.py:159`, `crud.py:88/586`); migrate
  gateway → bucketed; delete or hard-deprecate `crud.recall`/`crud.ingest_batch`.
- **Bricks:** `gemma-scout` (call graph) → `glm-implement` → `kimi-review`.
- **Negative space:** keep a shim (410 or one-release proxy) for any external client on legacy routes; gateway
  teach/recall must not break.
- **Accept:** no prod callers of the flat path; gateway tests green; dead code removed or quarantined.

---

## Execution order

```
Track 1 (latency→coverage):  WU1 → WU2 → (WU7 only on latency miss) → WU8 later
Track 2 (write path):        WU3 → WU4
Track 3 (verify/cleanup):    WU5 ∥ WU6 ∥ WU9   (WU9 after WU2)
```

- **Launch now, in parallel:** WU1 + WU3 + WU5.
- **Serial gates:** WU1→WU2 ; WU3→WU4 ; WU2→WU7 (only if latency misses).
- **Highest leverage / do-first two:** WU1 (parallel tiers) + WU2 (capped widen) = the product win
  (recovers ~60% dark memory). WU3 stops the ingest bleed independently.
- **Do NOT** spend frontier/paid tokens on WU8 or WU6 before WU1–3 land. **Do NOT** treat 4B/TEI as coverage prerequisites.

## Cost posture

All implementation on free cloud chains (glm/deepseek/kimi/minimax/gemma). Paid escalation reserved:
`gpt-review` on WU3 (write-path contract) and WU1 (only if session-concurrency design is contested). No Anthropic
frontier tokens except this Opus synthesis + per-WU brief authoring + post-run verification.
