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

## 2026-07-07 follow-up — code fixes deployed + the recall-coverage finding

Three fixes landed (`e8d6636`) and deployed to live `:8088` (verified via probe):
1. **Honest reranker telemetry** — `pick_reranker` now returns the provider's actual
   `.model` instead of the static `"qwen3-reranker-8b"`. Live probe confirms
   `diagnostics.reranker_used = "dengcao/Qwen3-Reranker-8B:Q8_0"` (was mislabeled).
2. **`source_ref` provenance in recall** — `BucketedRecallItem` now carries `source_ref`
   (populated from the ORM row). Live probe confirms `source_ref="answer_e5131a1b_1"` etc.
   This is a product win (recall now says WHERE a memory came from) and unbreaks the harness.
3. **`runner.py` rewired to the bucketed schema** — reads `buckets`, requests a broad tier
   set + `exclude_episodic:false`, matches on `source_ref`. No more false `recall@0.0`.
   `purge_namespace` no longer POSTs the phantom `/admin/delete-by-tag` (no such endpoint
   exists — documents the manual DB purge instead).

### THE headline finding — `hive_recall` can only see ~40% of the hive
`hive_recall` (MCP, `plugin/mcp_server.py:149`) hard-codes `top_per_tier` to
`{command, rule, skill, entity, decision}` and never sets `tiers`/`exclude_episodic`, so
tier resolution collapses to `DEFAULT_RECALL_TIERS` (those same 5 governance types). Every
semantic recall an agent does searches ONLY those 5 tiers. Live DB, `claude-shared` (n=7490):

| reachable by hive_recall (semantic) | count | | NOT semantically reachable | count |
|---|---|---|---|---|
| skill | 1800 | | episodic | 1844 |
| fact* (*not even added by exclude_episodic:false) | — | | fact | 994 |
| rule | 596 | | doctrine | 676 |
| decision | 294 | | NULL memory_type | 711 |
| entity | 236 | | session_summary | 192 |
| command | 39 | | correction / preference / reference | 108 |
| **total ~2965 (~40%)** | | | **total ~4525 (~60%)** | |

**~60% of the production hive — all episodic conversation, all facts, doctrine, and every
NULL-typed item — is invisible to semantic `hive_recall`.** It is still reachable by keyword
`hive_query` (FTS `/query` applies no tier filter), but NOT by natural-language similarity —
which is the whole point of an embedding memory and exactly the path agents use to "remember."

Mechanism confirmed at `knowledge.py:636`: `KnowledgeItem.memory_type == tier`. A NULL
memory_type matches no tier string, so **NULL-typed items (2069/9087 = 23% of ALL memory)
are unreachable via the bucketed path for ANY tier request** — keyword-only, permanently,
until typed.

This is the "living memory" north-star risk in one number: you teach it a fact or a
preference or a correction, and the primary recall interface will never semantically surface
it. Two fixes to weigh (operator decision — it changes retrieval semantics):
  a) widen `hive_recall`'s tier set (add episodic/fact/etc. — more recall, more latency/noise);
  b) backfill NULL memory_types + lean on the synthesis loop to promote raw episodic/fact
     into governance tiers (keeps recall lean but depends on promotion keeping up).

### Ingest ~200× regression — root cause CONFIRMED (not the dedup)
Traced `ingest_tiered`: the per-item cost is the **synchronous `classify()` call**
(`llm/classifier.py:134`) → `httpx.post` to Ollama `/api/generate` with
`classifier_model = gemma4:31b-cloud` (a 31B CLOUD model), on the write hot-loop, for every
item lacking an explicit abstract/summary. That is the ~10s/item. The semantic-dedup scan is
**index-backed** (`idx_knowledge_items_embedding_hnsw` exists, `ENABLE_HNSW_INDEX=true`) so it
is NOT the culprit — theory retired. April's 0.05s/item predates the T2.3 auto-classify feature.
Fix options (operator decision — affects how memory is typed, which drives the tier gap above):
fast local classifier / async off-path classify / batch classify.

## 2026-07-08 — tier-widening experiment: coverage WORKS, latency KILLS it (reverted)

Shipped the widened recall set (`fact/correction/preference/doctrine` added to
`DEFAULT_RECALL_TIERS` + `hive_recall`, `10c6145`), deployed, measured on live
`claude-shared`, then **reverted (`57d2d72`)**. What the measurement showed:

- **Coverage: confirmed.** Every new tier returned a full bucket — `fact:5,
  correction:5, preference:5, doctrine:5` on real queries. The ~23% dark slice
  becomes semantically recallable exactly as predicted.
- **Latency: unacceptable.** Wide (9-tier) recall p50 = **~12.9s** (42s cold),
  `rerank_ms ≈ 12.7s`. Reverting to 5 tiers restored p50 = **~7.4s**
  (`rerank_ms ≈ 7.2s`). That is **~1.4s PER TIER, sequential** — the per-tier
  rerank loop (`knowledge.py:633`) awaits each tier's rerank in turn.

**The reframe:** rerank is **GPU-bound sequential scoring**, so *baseline* recall
was ALREADY ~7s — widening just makes an existing latency problem worse linearly.
The plan's premise ("4B keeps widening affordable") is **falsified**: 4B is ~40%
faster per call (→ ~8s wide), still unacceptable, and the flip was correctly
blocked (needs larger-sample confirm anyway). Client-side parallelism won't help
either — the GPU serializes the forward passes.

**Corrected sequence — latency fix is the PREREQUISITE, not a companion:**
1. **Batched rerank** (score all candidates across all tiers in ONE forward pass,
   or a true batch endpoint) — this is the real unlock. Drops *baseline* recall
   from ~7s toward sub-2s and makes tier-count nearly free. Provider-level change
   in `reranker/ollama_provider.py`.
2. **THEN re-land the widening** (the diff is proven correct; just reverted) — now
   affordable. Optionally 4B on top, larger-sample confirmed.
3. Coverage recovery (~23%) is real and waiting; it is **gated on step 1**, not
   abandoned.

Prod is back to the exact prior state: 5 governance tiers, reranker ON / 8B, ~7.4s.
No net change shipped from this experiment except the knowledge that widening is a
latency problem wearing a coverage-problem costume.

## 2026-07-09 — "batched rerank" investigation: the premise is empirically dead

Went to design the batched-rerank latency fix. Instrumented the live 8B reranker
(`dengcao/Qwen3-Reranker-8B:Q8_0`) on hari's Ollama directly. Findings, all measured:

- **`OLLAMA_NUM_PARALLEL=1`** on the shared server — but raising it is **useless**:
  amortized cost/doc is **FLAT from 4→32 concurrent** (~75ms/doc np=5). The GPU is
  throughput-saturated at ~8 concurrent; more parallel slots don't raise throughput.
  Infra lever ruled OUT (and it would cost shared-fleet VRAM for nothing).
- **`num_predict` reduction is a mirage.** np=1 returns empty; the text-gen scorer
  needs ~5 tokens to emit `"0.95"`. Rerank is **prefill-bound**, not decode-bound, so
  cutting decode tokens barely helps. Ruled OUT.
- **Document length has ZERO effect** — 49ms/doc from 100 to 1200 chars. The per-
  candidate cost is a fixed ~50ms floor (model + system/template prefill overhead),
  not doc tokens. Truncation ruled OUT.
- **Net: rerank throughput is ~20 docs/sec (8B), fixed per candidate.** There is **no
  batching/config trick** for this Ollama serving setup. The ~7s baseline = 75
  candidates (multiplier 3 × top_k 5 × 5 tiers) ÷ ~20 docs/sec.

**So the ONLY real latency levers are:**
1. **Fewer candidates** — `reranker_candidate_multiplier` 3→2 (75→50 docs → ~2.5s).
   Direct rerank-quality tradeoff; must be measured, not assumed.
2. **The 4B model** — smaller = faster prefill (~2×). Auth-gated (correctly).
3. **A different serving layer** — TEI/vLLM with **prefix caching** would amortize the
   shared system+instruct prefill across candidates (the fixed ~50ms floor) and do true
   continuous batching. This is the real structural fix, but it's an infra project.

**The one unambiguous WIN found — logit scoring (quality, not speed):** Ollama 0.20.0
exposes top-level `logprobs`. Using the **official Qwen3-Reranker chat template + `raw:true`
+ `num_predict:1`** and reading `P(yes)` vs `P(no)` from `top_logprobs` gives clean,
continuous, correctly-ordered scores (validated HIGH `yes@-0.1` ≫ MED `no@-0.71` ≫ LOW).
The current provider instead prompts for a "0-1" float and regex-parses it — which returns
COARSE, near-binary scores (mostly `0.0` or `0.95`). Logit scoring is the correct way to
use a cross-encoder reranker: ~equal speed, materially better ranking signal. In-repo,
no infra. **This is worth building — but it changes the core ranking path, so it needs a
recall@k quality gate (via the now-fixed harness) before it goes live, not a blind swap.**

## 2026-07-09 (cont.) — logit scoring BUILT + recall@k gate PASSED

Built logit-based scoring behind a `reranker_scoring_mode` hot-key (`generate`|`logit`,
default `generate` — deployed behavior-neutral). Then ran the recall@k gate out-of-band
(no prod flip): fetch each query's full candidate pool from the live endpoint, score the
SAME candidates both ways, compare ranking. `device-hari-private`, n=25:

| scorer | recall@1 | recall@5 | recall@10 |
|--------|----------|----------|-----------|
| generate (current prod) | 0.720 | 0.880 | 0.920 |
| **logit (new)** | **0.800** | **0.960** | **0.960** |

**Logit wins on all three k** (+8pts @1, +8pts @5, +4pts @10). The gate clears logit as
the better scorer. It is a **quality** win, not a latency one — still ~7s baseline (the
prefill wall stands). n=25 is directional but the margin is consistent across every k.

**Recommendation:** flip the prod default to `logit` (hot-key
`reranker_scoring_mode=logit`, reversible) — free recall-quality gain, zero latency cost.
Latency remains a separate decision (4B model / TEI serving).

Landed: `a5bc2d7` (logit provider + toggle). Activation is one hot-key, pending operator OK.

## Decision state
- [x] **2026-07-07:** telemetry + source_ref + runner fixes landed (`e8d6636`) & deployed; both prod fixes verified live.
- [x] **2026-07-07:** ingest regression root-caused to synchronous per-item `gemma4:31b-cloud` classify (dedup exonerated — HNSW-indexed).
- [x] **2026-07-07:** quantified the recall-coverage gap — `hive_recall` reaches ~40% of claude-shared; 23% of all memory is NULL-typed and bucket-unreachable.
- [x] Deployed landed main, healthy, clean startup.
- [x] Baseline (ON) + re-measure (OFF) latency captured (~73× latency delta).
- [x] **QUALITY A/B measured** via standalone bucketed probe (runner.py was broken).
- [x] Result: reranker EARNS its keep on ranking precision (+20 @1). **Do NOT retire it.**
- [ ] **C2 REVISED:** don't retire the reranker. Legacy `crud.recall`/`ingest_batch` retirement is
      a *separate* question (path parity), unblocked by this — but the reranker stays.
- [ ] **New P-item:** fix reranker latency (batched logits / 4B-fast) so quality is kept at ~sub-second.
- [ ] **Architectural decision (QUANTIFIED above):** `hive_recall` reaches only ~40% of the hive;
      ~60% (episodic/fact/doctrine/NULL/...) is semantically dark; 23% of all memory is NULL-typed.
      Widen recall tiers vs. backfill types + trust the synthesis/promotion loop? Operator call.
- [ ] **Ingest-regression fix (root-caused):** move the per-item `gemma4:31b-cloud` classify off
      the write hot-loop (fast-local / async / batch). Operator call — it changes how memory is typed.
- [x] Current live state: reranker **ON** (restored prior prod default; no silent quality regression).
- [x] **Fixed `runner.py`** to the bucketed schema (`e8d6636`); source_ref matching verified live.
- [ ] Full 500-Q bucketed benchmark still pending — needs classified data (bench_longmemeval's 940
      April items are NULL-typed → unrecallable); re-ingest is gated on the slow classify path above.
