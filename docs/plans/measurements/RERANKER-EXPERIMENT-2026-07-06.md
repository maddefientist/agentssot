# Reranker On/Off — Live Latency Experiment

**Date:** 2026-07-06 · **Service:** hari `:8088` · **Deployed:** `main @ 0fa67b4`
(first deploy carrying P0–P4 + reranker-parallelization `86603a7`; prior live image was 07-02 code).
**Namespace probed:** claude-shared · **probe-count:** 20 · **harness:** `scripts/recall_quality_report.py --probe`

## Latency (live `/api/v1/knowledge/recall`)

| config | vec_ms p50 | vec_ms p95 | rerank_ms p50 | total p50 | total p95 | reranker_used |
|--------|-----------|-----------|---------------|-----------|-----------|---------------|
| reranker **ON**  (`ollama`, qwen3-reranker-8b) | 22.0 | 39.1 | 4,567.5 | **4,644.7** | 4,826.3 | qwen3-reranker-8b ×20 |
| reranker **OFF** (`none`)                       | 26.5 | 40.2 | 0.0     | **63.5**    | 78.9      | none ×20 |

**Delta: ~73× faster total (4,645 → 63.5ms).** Retrieval (vector track) is ~25ms; the reranker
added ~4.5s of pure LLM-judge overhead per recall — *after* the ThreadPool parallelization fix
(was 11–28s before). RRF fuses vector + FTS in the bucketed path with no reranker.

## Flip mechanism (worked, no restart)
`POST /admin/config {"key":"reranker_provider","value":"none"}` (admin key `claude-sync-admin`)
→ `set_override` + `_reload_runtime_overrides` + `rebuild_providers` → live. HTTP 200,
`reranker_deep`/`reranker_fast` both `provider disabled`.

## Production-signal caveat (why latency ≠ the whole decision)
Volume/feedback stats are too sparse to judge *quality* from prod: claude-shared window had
only 3 recalls/14d; lifetime cross-check = 41,502 recalls · 7,459 items · 0.11% feedback.
**Latency parity is proven; QUALITY parity is NOT yet measured.** The C2 retirement of the
legacy `crud.recall`/`ingest_batch` stack is gated on quality parity — see decision below.

## Quality A/B — recall@k (the real C2 gate)

**Harness note:** `benchmarks/longmemeval/runner.py` is INCOMPATIBLE with the current API
and produced `recall@*=0.0` (a false zero, not a signal): it reads `results` (endpoint now
returns `buckets`) and matches `source_ref` (absent from `BucketedRecallItem`), and never
requests the `episodic`/`fact` tiers the content classifies into (`DEFAULT_RECALL_TIERS` =
the 5 governance types; **episodic excluded by default**). This is why the 2nd agent's
`vector_only.json` was a null stub. Measured instead with a standalone bucketed-path probe
(`scratchpad/rerank_ab_probe.py`): queries `tiers=[episodic,fact]`, merges buckets by score,
maps returned `id`→`source_ref` via a DB map. n=40 (first 40 oracle Qs), ns=device-hari-private.

| config | recall@1 | recall@5 | recall@10 | probe lat p50 |
|--------|----------|----------|-----------|---------------|
| reranker **OFF** (`none`)              | 0.550 | 0.850 | 0.900 | 114ms |
| reranker **ON — 8B deep** (Qwen3-Reranker-8B:Q8_0) | 0.750 | 0.900 | 0.925 | 4,064ms |
| reranker **ON — 4B fast** (Qwen3-Reranker-4B:Q4_K_M) | **0.825** | **0.950** | **0.950** | **2,403ms** |

**The reranker is NOT dead weight** — it adds +20 pts recall@1 (0.55→0.75) at the 8B, and the
**4B fast model DOMINATES the 8B on this set: higher quality AND ~40% lower latency** (0.825 @1,
2.4s vs 0.75 @1, 4.1s). This **refutes the "LLM-judge hack / remove it" premise** — the decision
is which reranker + how to cut its latency, NOT whether to keep one.

Immediate low-risk win (hot key, zero code): switch the deep reranker model to the 4B — better
recall AND faster. Then the batched single-forward-pass fix (one `/api/generate` instead of N)
to push sub-second. Verify on a larger sample before making 4B the prod default.

**Diagnostics label bug (minor, real):** `diagnostics.reranker_used` is a STATIC string from
`reranker/router.py` (`"qwen3-reranker-8b"` for the deep slot) — it does NOT reflect the actual
`ollama_reranker_model`. The 4B run was mislabeled "8b"; the 4,064→2,403ms latency drop (both
tight distributions) is the proof the 4B model actually loaded. Fix the label so telemetry is honest.

**Caveats (honest):** n=40 is small/noisy (recall@1 delta = 8 questions); content is episodic/fact
in a bench ns on non-default tiers; oracle set has answer_session_ids==haystack_session_ids so
this measures "does the right session rank high" — exactly what reranking helps. A 200-question
run would tighten it, but direction is clear.

## Adjacent finding — ~200× ingest write-path regression
The prior committed benchmark (`results/vector_only.json`, 2026-04-14) ingested **940 sessions in
48s (~0.05s/item)**. Today the same `/api/v1/knowledge/ingest` runs at **~10s/item** (measured live,
0.1 items/s) — a **~200× slowdown**, dominated by the per-item cloud classifier (`gemma4:31b-cloud`)
+ semantic-dedup search added since April. This is a real efficiency regression on the WRITE path,
independent of the reranker (read path). Options: batch/async classify, local fast classifier for
ingest, or make classification lazy/off-path. Flag for its own investigation.

*(Prior 500Q flat-path baseline for reference: vector-only recall@1=0.286 / @10=0.664 — NOT directly
comparable to today's numbers: different namespace, n, and the flat vs bucketed path.)*

## Decision state
- [x] Deployed landed main, healthy, clean startup.
- [x] Baseline (ON) + re-measure (OFF) latency captured (~73× latency delta).
- [x] **QUALITY A/B measured** via standalone bucketed probe (runner.py was broken).
- [x] Result: reranker EARNS its keep on ranking precision (+20 @1). **Do NOT retire it.**
- [ ] **C2 REVISED:** don't retire the reranker. Legacy `crud.recall`/`ingest_batch` retirement is
      a *separate* question (path parity), unblocked by this — but the reranker stays.
- [ ] **New P-item:** fix reranker latency (batched logits / 4B-fast) so quality is kept at ~sub-second.
- [ ] **Architectural flag:** default bucketed recall EXCLUDES episodic + omits `fact` from
      `DEFAULT_RECALL_TIERS` — so default `hive_recall` returns nothing for conversational/factual
      memory. Intended (governance-first) or a gap vs the "living memory" vision? Needs a decision.
- [x] Current live state: reranker **ON** (restored prior prod default; no silent quality regression).
- [ ] Fix `runner.py` to the bucketed schema so future runs aren't false-zero.
